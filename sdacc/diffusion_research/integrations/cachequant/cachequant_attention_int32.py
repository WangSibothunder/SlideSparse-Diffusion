"""Native asymmetric attention grid deployment; actual CUDA admission pending.

Q/K and softmax-probability/V use signed INT8 GEMMs and I32 zero correction.
Native divide/round/zero rules are imported unchanged from the Dense caller.
This caller approximates the native BF16-dequantized einsums; it does not
claim bit equivalence to those einsums or to original fused SDPA. Every actual
table/caller requires its own numeric, reload, trace and whole-caller gate.
No symmetric substitution, FP8, float-GEMM fallback, sampling or online fit.
"""
import torch
from torch.nn import functional as F
from .cachequant_dense_int32 import encode_native, native_round


SIDES = ('q', 'k', 'probability', 'v')


def validate_tables(tables):
    if set(tables) != set(SIDES):
        raise ValueError('all four original native per-tensor attention grids required')
    dtype = None
    for side in SIDES:
        if set(tables[side]) != {'delta', 'zero'}:
            raise ValueError('scale/zero only; no replacement fit or implicit defaults')
        scale, zero = tables[side]['delta'], tables[side]['zero']
        if (scale.shape != (20,) or zero.shape != (20,) or scale.dtype != zero.dtype
            or scale.dtype not in (torch.bfloat16, torch.float32)
            or scale.device.type != 'cpu' or zero.device.type != 'cpu'
            or scale.requires_grad or zero.requires_grad
            or not torch.isfinite(scale).all() or not (scale > 0).all()
            or not torch.isfinite(zero).all()):
            raise ValueError('frozen original-dtype finite native20 CPU tables required')
        if dtype is None:
            dtype = scale.dtype
        if dtype != scale.dtype:
            raise ValueError('original attention quantizer dtypes must agree')
        rounded = native_round(zero).double()
        if not ((rounded >= -32768) & (rounded <= 32767)).all():
            raise ValueError('native zero exceeds explicit integer deployment bound')
    return dtype


def integer_bounds(k, left_zero, right_zero):
    """Conservative bounds cover EACH sequential I32 intermediate, not only final.

    Encoded signed operands are in [-128,127]. Correction first subtracts the
    right shift times left row sum, then the left shift times right column sum,
    then adds K*left_shift*right_shift. Bounds are per actual timestep pair.
    """
    if type(k) is not int or not 1 <= k <= 32768:
        raise ValueError('declared positive logicalK <=32768 required')
    a = (native_round(left_zero).double() - 128).abs()
    b = (native_round(right_zero).double() - 128).abs()
    bound = k * (128 * 128 + 128 * b + 128 * a + a * b)
    if not torch.isfinite(bound).all() or (bound > torch.iinfo(torch.int32).max).any():
        raise ValueError('I32 dot/row/column/constant intermediates not proven safe')
    return int(bound.max())


def correct_integer(raw, left_sum, right_sum, left_zero, right_zero, k):
    sa = native_round(left_zero).to(torch.int32) - 128
    sb = native_round(right_zero).to(torch.int32) - 128
    return raw - sb * left_sum[..., None] - sa * right_sum[..., None, :] + k * sa * sb


@torch.library.custom_op('sigma_orthogonal_native_attention_i32::matmul', mutates_args=())
def matmul(left: torch.Tensor, right: torch.Tensor, left_scale: torch.Tensor,
           left_zero: torch.Tensor, right_scale: torch.Tensor,
           right_zero: torch.Tensor, transpose_right: bool) -> torch.Tensor:
    # Quantize in each original native operand layout before any transpose or
    # integer padding. The entire probability tensor is consumed, no sampling.
    a = (encode_native(left, left_scale, left_zero).to(torch.int16) - 128).to(torch.int8)
    b = (encode_native(right, right_scale, right_zero).to(torch.int16) - 128).to(torch.int8)
    if transpose_right:
        b = b.transpose(1, 2)
    batch, m, k = a.shape
    n = b.shape[2]
    left_sum, right_sum = a.sum(2, dtype=torch.int32), b.sum(1, dtype=torch.int32)
    a = F.pad(a, (0, (-k) % 32, 0, (-m) % 8)).contiguous()
    b = F.pad(b, (0, (-n) % 8, 0, (-k) % 32)).contiguous()
    outputs = []
    for h in range(batch):
        raw = torch._int_mm(a[h], b[h])[:m, :n]
        corrected = correct_integer(raw, left_sum[h], right_sum[h], left_zero, right_zero, k)
        # Scale only AFTER exact I32 zero correction. Round to native output
        # dtype before the original attention scale/mask/softmax operations.
        outputs.append((corrected.float() * (left_scale.float() * right_scale.float())).to(left.dtype))
    return torch.stack(outputs)


@matmul.register_fake
def fake_matmul(left, right, left_scale, left_zero, right_scale, right_zero, transpose_right):
    return torch.empty((left.shape[0], left.shape[1], right.shape[1] if transpose_right else right.shape[2]),
                       device=left.device, dtype=left.dtype)


def scale_mask_softmax(sim, scale, mask=None):
    # Same native Sigma operation boundary: BF16 einsum output, multiply the
    # ORIGINAL head scale, add original prepared additive mask, then softmax.
    sim = sim * scale
    if mask is not None:
        if (mask.dtype == torch.bool or mask.ndim != 3 or mask.shape[0] != sim.shape[0]
            or mask.shape[-1] != sim.shape[-1] or mask.shape[-2] not in (1, sim.shape[-2])):
            raise ValueError('original Bhead additive mask/broadcast semantics required')
        sim = sim + mask
    return sim.softmax(dim=-1)


class NativeAttentionInt32(torch.nn.Module):
    """Fixed slots for FOUR native grids, 8 explicit D2D copies in Timer.select.

    Constructor is format validation, not actual table/producer acceptance.
    Geometry is declared explicitly; native calibration and warm kernel/caller
    identities must be checked independently before any model deployment.
    """
    def __init__(self, tables, geometry, scale, device='cpu'):
        super().__init__()
        self.native_dtype = validate_tables(tables)
        if set(geometry) != {'batch_heads', 'queries', 'context', 'head_dim'} or any(
            type(v) is not int or v <= 0 for v in geometry.values()):
            raise ValueError('original explicit Bhead/query/context/head geometry required')
        if not isinstance(scale, (int, float)) or not __import__('math').isfinite(scale) or scale <= 0:
            raise ValueError('original finite positive softmax scale required')
        self.geometry, self.scale = dict(geometry), float(scale)
        self.bound_qk = integer_bounds(geometry['head_dim'], tables['q']['zero'], tables['k']['zero'])
        self.bound_pv = integer_bounds(geometry['context'], tables['probability']['zero'], tables['v']['zero'])
        for side in SIDES:
            for field in ('delta', 'zero'):
                value = tables[side][field].detach().clone()
                self.register_buffer(side + '_' + field + '_table', value.to(device))
                self.register_buffer(side + '_' + field, value[0].clone().to(device))
        self.actual_native_tables_numeric_trace_cost_pending = True

    def select(self, step):
        if type(step) is not int or not 0 <= step < 20:
            raise ValueError('declared20 step only')
        destinations, size = [], 0
        for side in SIDES:
            for field in ('delta', 'zero'):
                dst = getattr(self, side + '_' + field)
                dst.copy_(getattr(self, side + '_' + field + '_table')[step])
                destinations.append(dst.data_ptr())
                size += dst.element_size()
        return dict(parameter_copy_count=8, parameter_copy_bytes=size,
                    destination_addresses=destinations, required_timer='Timer.select')

    def _require(self, value, shape):
        if (value.device.type != 'cuda' or value.device != self.q_delta.device
            or value.dtype != self.native_dtype or tuple(value.shape) != shape):
            raise ValueError('actual CUDA/native dtype and declared original attention layout required')

    def qk(self, q, k):
        g = self.geometry
        self._require(q, (g['batch_heads'], g['queries'], g['head_dim']))
        self._require(k, (g['batch_heads'], g['context'], g['head_dim']))
        return matmul(q, k, self.q_delta, self.q_zero, self.k_delta, self.k_zero, True)

    def pv(self, p, v):
        g = self.geometry
        self._require(p, (g['batch_heads'], g['queries'], g['context']))
        self._require(v, (g['batch_heads'], g['context'], g['head_dim']))
        return matmul(p, v, self.probability_delta, self.probability_zero, self.v_delta, self.v_zero, False)

    def forward(self, q, k, v, mask=None):
        # V quantization happens only after softmax probability quantization,
        # matching original NativeSigmaAttention's Q/K -> P -> V order.
        p = scale_mask_softmax(self.qk(q, k), self.scale, mask)
        return self.pv(p, v)


class NativeIntegerSigmaAttention:
    """Exact host order reused from033; integer core has separate admission.

    No implicit grid selection here: the model's registered Timer.select must
    select all 8 native parameters before the compiled forward. This processor
    is not installed by CPU preparation or a synthetic diagnostic.
    """
    def __init__(self, name, core):
        self.name, self.core = name, core

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        if args or kwargs or temb is not None:
            raise ValueError('unregistered attention kwargs/temb need an explicit native caller')
        if (hidden_states.ndim != 3 or attn.spatial_norm is not None
            or attn.group_norm is not None or attn.norm_q is not None or attn.norm_k is not None):
            raise ValueError('declared Sigma BNC attention without extra normalization required')
        residual = hidden_states
        batch, n, _ = hidden_states.shape
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if context.ndim != 3 or context.shape[0] != batch:
            raise ValueError('same ordered CFG batch for attention context required')
        if encoder_hidden_states is not None and attn.norm_cross:
            context = attn.norm_encoder_hidden_states(context)
        q, k, v = attn.to_q(hidden_states), attn.to_k(context), attn.to_v(context)
        if q.shape[-1] % attn.heads or k.shape[-1] != q.shape[-1] or v.shape[-1] != q.shape[-1]:
            raise ValueError('original equal native Q/K/V projection widths required')
        d = q.shape[-1] // attn.heads
        g = self.core.geometry
        if (g != dict(batch_heads=batch * attn.heads, queries=n, context=context.shape[1], head_dim=d)
            or self.core.scale != attn.scale):
            raise ValueError('original model geometry/scale changed; no padded-scale substitution')
        def fold(x):
            return x.reshape(batch, -1, attn.heads, d).permute(0, 2, 1, 3).reshape(batch * attn.heads, -1, d)
        mask = None
        if attention_mask is not None:
            mask = attn.prepare_attention_mask(attention_mask, context.shape[1], batch)
            if mask.ndim == 4:
                mask = mask.reshape(batch * attn.heads, mask.shape[-2], mask.shape[-1])
        y = self.core(fold(q), fold(k), fold(v), mask)
        y = y.reshape(batch, attn.heads, n, d).permute(0, 2, 1, 3).reshape(batch, n, attn.heads * d)
        y = attn.to_out[1](attn.to_out[0](y))
        if attn.residual_connection:
            y = y + residual
        return y / attn.rescale_output_factor
