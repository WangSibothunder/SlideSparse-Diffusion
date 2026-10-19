"""One Quant+Lift kernel and one direct-BF16 sparse GEMM per projection."""
from pathlib import Path
import torch
import triton
import triton.language as tl
from .packing import group_size, lifted_k, slide_weight
from .backend import load_native, sha
from .profile import execution_profile, known_configurations, LEGACY_CORE_SHA


@triton.jit
def _quant_lift(X, Q, S, M: tl.constexpr, K: tl.constexpr, PK: tl.constexpr,
                SX: tl.constexpr, L: tl.constexpr, WINDOWS: tl.constexpr,
                VALID: tl.constexpr, B: tl.constexpr, RELOAD: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * SX + col, mask=(row < M) & (col < K), other=0).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(x)), 1e-12)
    scale = tl.maximum(maximum / 127., 1. / (127. * 512.))
    inv = 1. / scale
    if L == 4:
        q = tl.clamp(tl.extra.cuda.libdevice.rint(x * inv), -128., 127.).to(tl.int8)
    else:
        src = (col // (4 * WINDOWS)) * L + 2 * ((col // 4) % WINDOWS) + col % 4
        lifted = tl.load(X + row * SX + src,
                         mask=(row < M) & (col < VALID) & (src < K), other=0).to(tl.float32)
        q = tl.clamp(tl.extra.cuda.libdevice.rint(lifted * inv), -128., 127.).to(tl.int8)
        q = tl.where(col < VALID, q, 0)
    tl.store(Q + row * PK + col, q, mask=col < PK)
    tl.store(S + row, tl.where(row < M, scale, 1.))


@triton.jit
def _quant_register(X, Q, S, M: tl.constexpr, K: tl.constexpr, PK: tl.constexpr,
                    SX: tl.constexpr, L: tl.constexpr, WINDOWS: tl.constexpr,
                    G: tl.constexpr):
    row = tl.program_id(0)
    groups = tl.arange(0, G)
    lanes = tl.arange(0, 8)
    src = groups[:, None] * L + lanes[None, :]
    x = tl.load(X + row * SX + src,
                mask=(row < M) & (src < K) & (lanes[None, :] < L), other=0).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.max(tl.abs(x), axis=1), axis=0), 1e-12)
    scale = tl.maximum(maximum / 127., 1. / (127. * 512.))
    q = tl.clamp(tl.extra.cuda.libdevice.rint(x * (1. / scale)), -128., 127.).to(tl.int32) & 255
    # Statically rearrange the eight columns using Triton3.1 split/join. No
    # dynamic gather, re-reading original X, or global intermediate Q matrix.
    even, odd = tl.split(tl.reshape(q, (G, 4, 2)))
    e04, e26 = tl.split(tl.reshape(even, (G, 2, 2)))
    o15, o37 = tl.split(tl.reshape(odd, (G, 2, 2)))
    q0, q4 = tl.split(e04)
    q2, q6 = tl.split(e26)
    q1, q5 = tl.split(o15)
    q3, q7 = tl.split(o37)
    w0 = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24)
    w1 = q2 | (q3 << 8) | (q4 << 16) | (q5 << 24)
    if L == 6:
        values = tl.join(w0, w1)
        win = tl.arange(0, 2)
    else:
        w2 = q4 | (q5 << 8) | (q6 << 16) | (q7 << 24)
        zero = tl.full((G,), 0, tl.int32)
        values = tl.reshape(tl.join(tl.join(w0, w2), tl.join(w1, zero)), (G, 4))
        win = tl.arange(0, 4)
    dst = groups[:, None] * WINDOWS + win[None, :]
    words = Q.to(tl.pointer_type(tl.int32)) + row * (PK // 4)
    # Padding is written here too; valid groups contain zero padded input lanes.
    tl.store(words + dst, values, mask=(dst < PK // 4) & (win[None, :] < WINDOWS))
    tl.store(S + row, tl.where(row < M, scale, 1.))


def quantize_lift(x, mode="2:4", strategy="register"):
    """Dynamic per-token quantization; quantize AND dequantize use same floor.

    Input must be [M,K] with unit column stride; row stride may be noncontiguous.
    Padding and the sliding expansion are produced inside this single kernel.
    """
    l = group_size(mode)
    if strategy not in ("register", "reload"):
        raise ValueError("unknown Quant+Lift strategy")
    m, k = x.shape
    valid, pk = lifted_k(k, l)
    pm = triton.cdiv(m, 32) * 32
    q = torch.empty((pm, pk), dtype=torch.int8, device=x.device)
    s = torch.empty(pm, dtype=torch.float32, device=x.device)
    b = triton.next_power_of_2(max(k, pk))
    if l != 4 and strategy == "register":
        g = triton.next_power_of_2(max(triton.cdiv(k, l), triton.cdiv(pk // 4, l // 2 - 1)))
        _quant_register[(pm,)](x, q, s, M=m, K=k, PK=pk, SX=x.stride(0),
                              L=l, WINDOWS=l // 2 - 1, G=g, num_warps=4 if b <= 2048 else 8)
    else:
        _quant_lift[(pm,)](x, q, s, M=m, K=k, PK=pk, SX=x.stride(0),
                           L=l, WINDOWS=l // 2 - 1, VALID=valid, B=b,
                           RELOAD=True, num_warps=4 if b <= 2048 else 8)
    return q, s


class PackedLinear(torch.nn.Module):
    """Frozen packed projection, supporting shared-input QKV/KV concatenation.

    Prepare with concatenate([q,k,v]) along output channels, using identically
    ordered scales/bias. Split the final output by widths outside this module.
    Different-input cross-Q must use a different invocation.
    """
    def __init__(self, packed, metadata, scales, bias, k, n, mode, config="auto",
                 strategy="register", identity=None, frontend="auto"):
        super().__init__()
        self.k, self.n = int(k), int(n)
        self.mode = group_size(mode)
        self.config, self.strategy = ("auto" if config == "auto" else int(config)), strategy
        self.frontend = frontend
        self._graphs = {}
        self.in_features, self.out_features = self.k, self.n
        self.identity = identity or {}
        for name, value in (("packed", packed), ("metadata", metadata),
                            ("scales", scales), ("bias", bias)):
            self.register_buffer(name, value)

    @classmethod
    def prepare(cls, qweight, scales, bias=None, mode="2:4", config="auto",
                strategy="register", frontend="auto"):
        if config not in (0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, "auto") or strategy not in ("register", "reload"):
            raise ValueError("unsupported execution configuration")
        n, k = qweight.shape
        scales = scales.detach().cpu().float().contiguous()
        bias = torch.empty(0, dtype=torch.float32) if bias is None else bias.detach().cpu().float().contiguous()
        if scales.shape != (n,) or not torch.isfinite(scales).all() or torch.any(scales <= 0):
            raise ValueError("finite positive per-channel scales required")
        if bias.numel() and (bias.shape != (n,) or not torch.isfinite(bias).all()):
            raise ValueError("finite per-channel bias required")
        lifted = slide_weight(qweight, mode)
        _, pk = lifted_k(k, mode)
        packed, metadata = torch.ops.sdacc_dualscale_sm80.pack_cpu(lifted, pk)
        import hashlib
        identity = dict(qweight_sha=hashlib.sha256(qweight.contiguous().numpy().tobytes()).hexdigest(),
                        k_physical=pk, k_lifted=lifted.shape[1],
                        kernel_source_sha=sha(Path(__file__).parent / "csrc/dualscale_sm80.cu"))
        if frontend not in ('triton','native','scalar','packed16','graph_native','graph_packed16','graph1_native','graph1_packed16','direct2_native','direct2_packed16','auto'):
            raise ValueError('unknown frontend')
        return cls(packed, metadata, scales, bias, k, n, mode, config, strategy, identity, frontend)

    def configure(self, has_addend=False):
        """Set shared-memory attributes and inspect resources before timing."""
        configs = known_configurations(self.mode,self.k,self.n) if self.config == "auto" else [self.config]
        resources = {c:torch.ops.sdacc_dualscale_sm80.configure(self.packed,c,has_addend) for c in configs}
        if not has_addend:
            from .profile import _PROFILES
            rows = {m for (l,m,k,n),(c,f) in _PROFILES.items()
                    if l == self.mode and k == self.k and n == self.n and f.startswith(('graph','direct2_'))}
            for m in rows:
                c,f = self.configuration_for(m)
                if f.startswith(('graph','direct2_')) and m not in self._graphs:
                    # Explicit-config callers may override the published profile.
                    torch.ops.sdacc_dualscale_sm80.configure(self.packed,c,False)
                    self._graphs[m] = (torch.ops.sdacc_prepared_projection_sm80.prepare_direct2 if f.startswith('direct2_') else torch.ops.sdacc_prepared_projection_sm80.prepare_singlepatch if f.startswith('graph1_') else torch.ops.sdacc_prepared_projection_sm80.prepare)(
                        self.packed,self.metadata,self.scales,self.bias,self.k,m,self.n,
                        self.mode,c,f.endswith('packed16'))
        return resources if self.config == "auto" else resources[self.config]

    def configure_for(self, m):
        """Offline preparation of a shape-specific two-node graph."""
        config, frontend = self.configuration_for(m)
        self.configure()
        if frontend.startswith(('graph','direct2_')) and m not in self._graphs:
            self._graphs[m] = (torch.ops.sdacc_prepared_projection_sm80.prepare_direct2 if frontend.startswith('direct2_') else torch.ops.sdacc_prepared_projection_sm80.prepare_singlepatch if frontend.startswith('graph1_') else torch.ops.sdacc_prepared_projection_sm80.prepare)(
                self.packed,self.metadata,self.scales,self.bias,self.k,m,self.n,
                self.mode,config,frontend.endswith('packed16'))
        return self._graphs.get(m)

    def close_graphs(self):
        for handle in self._graphs.values():
            torch.ops.sdacc_prepared_projection_sm80.release(handle)
        self._graphs.clear()

    def _apply(self, fn, recurse=True):
        if hasattr(self, '_graphs') and self._graphs:
            self.close_graphs()
        return super()._apply(fn,recurse)

    def configuration_for(self, m):
        """Use only frozen matching shapes; unmatched shapes retain cfg5/Triton."""
        config, frontend = execution_profile(self.mode,m,self.k,self.n)
        if self.config != "auto":config = self.config
        if self.frontend != "auto":frontend = self.frontend
        return config,frontend

    def forward(self, x, output_addend=None):
        if x.dtype != torch.bfloat16 or not x.is_cuda or x.shape[-1] != self.k:
            raise ValueError("BF16 CUDA input with matching last dimension required")
        flat = x.reshape(-1, self.k)
        if flat.shape[0] == 0:
            return x.new_empty((*x.shape[:-1], self.n))
        config,frontend = self.configuration_for(flat.shape[0])
        if output_addend is not None:
            if output_addend.shape != (*x.shape[:-1], self.n) or output_addend.dtype != torch.bfloat16 or output_addend.device != x.device:
                raise ValueError("output_addend must match BF16 output shape/device")
            output_addend = output_addend.reshape(-1, self.n)
            if output_addend.stride(-1) != 1:
                output_addend = output_addend.contiguous()
        if frontend.startswith(('graph','direct2_')):
            if output_addend is not None:
                # Addend workloads retain the established native path.
                return torch.ops.sdacc_mild_projection_sm80.projection(
                    x,self.packed,self.metadata,self.scales,self.bias,self.mode,self.n,config,output_addend)
            handle = self._graphs.get(flat.shape[0])
            if handle is None:
                raise RuntimeError('configure_for(M) is required before graph forward')
            return torch.ops.sdacc_prepared_projection_sm80.run(x,handle,self.n)
        if frontend == 'packed16':
            return torch.ops.sdacc_mild_projection_sm80.projection_packed16(
                x,self.packed,self.metadata,self.scales,self.bias,self.mode,self.n,config,output_addend)
        if frontend == 'native':
            return torch.ops.sdacc_mild_projection_sm80.projection(
                x,self.packed,self.metadata,self.scales,self.bias,self.mode,self.n,config,output_addend)
        if frontend == 'scalar':
            return torch.ops.sdacc_mild_projection_sm80.projection_scalar(
                x,self.packed,self.metadata,self.scales,self.bias,self.mode,self.n,config,output_addend)
        if flat.stride(-1) != 1:
            flat = flat.contiguous()
        q, sa = quantize_lift(flat, self.mode, self.strategy)
        y = torch.ops.sdacc_dualscale_sm80.run(q, self.packed, self.metadata,
                                              sa, self.scales, self.bias,
                                              flat.shape[0], self.n, config, output_addend)
        return y.reshape(*x.shape[:-1], self.n)

    def save(self, path):
        """Portable tensor payload; no handles, searched plans, or Python pickle."""
        torch.save(dict(version=1, k=self.k, n=self.n, mode=self.mode, config=self.config,
                        strategy=self.strategy, identity=self.identity, frontend=self.frontend,
                        **{name: tensor.detach().cpu() for name, tensor in self.named_buffers()}), path)

    @classmethod
    def load(cls, path, device="cpu", optimized=True):
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob.pop("version") != 1:
            raise ValueError("unsupported packed payload version")
        current=sha(Path(__file__).parent / "csrc/dualscale_sm80.cu")
        if blob["identity"]["kernel_source_sha"] not in (current,"2787082a7feed4f7d9287578f7b7cac80857770dc4d8d704cdf888dca4a625a4",LEGACY_CORE_SHA,"33f19700a1d9e6e01390cdc19627837ecfdeca2fffdd45e65ca1e7e7cac0c12d","600cbf9ffdd7a863b9d04a0afdab6279caea70d3a705ea8cf92e26ab8c9f0b7a"):
            raise ValueError("packing/kernel source identity differs")
        if optimized:
            blob["identity"]=dict(blob["identity"],loaded_execution_original=dict(config=blob['config'],frontend=blob.get('frontend','triton')))
            blob['config'],blob['frontend']='auto','auto'
        return cls(**blob).to(device)
