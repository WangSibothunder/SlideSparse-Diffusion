"""Stride-explicit FLUX preparation; fixed upstream -0.1 similarity threshold.

Mean pairwise cosine equals squared norm of the mean normalized vector.
Compute that reduction without materializing the 128x128 Gram matrix.
Quantization, BF16 key centering, top-k selection and LUT semantics are retained.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def pool_quant(X, Mean, Pool, Sim, Quant, Scale,
               N: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
               XB: tl.constexpr, XH: tl.constexpr, XN: tl.constexpr,
               QB: tl.constexpr, QH: tl.constexpr, QN: tl.constexpr,
               PB: tl.constexpr, PH: tl.constexpr, PN: tl.constexpr, PD: tl.constexpr,
               SB: tl.constexpr, SH: tl.constexpr, SN: tl.constexpr,
               CB: tl.constexpr, CH: tl.constexpr, CN: tl.constexpr,
               MB: tl.constexpr, MH: tl.constexpr, MD: tl.constexpr,
               CENTER: tl.constexpr):
    b, h, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    H, NB = tl.num_programs(1), tl.num_programs(2)
    rows = block * BS + tl.arange(0, BS)
    cols = tl.arange(0, D)
    x = tl.load(X + b * XB + h * XH + rows[:, None] * XN + cols[None, :],
                mask=rows[:, None] < N, other=0)
    if CENTER:
        mean = tl.load(Mean + b*MB + h*MH + cols*MD)
        # The pinned upstream subtraction rounds to BF16 before FP32 reductions.
        x = (x - mean).to(tl.bfloat16)
    xf = x.to(tl.float32)
    count = tl.minimum(BS, N - block * BS)
    pooled = tl.sum(xf, 0) / count
    norm = tl.sqrt(tl.sum(xf * xf, 1))
    normalized = (xf / norm[:, None]).to(tl.float16).to(tl.float32)
    average = tl.sum(normalized, 0) / count
    similarity = tl.sum(average * average, 0) > -0.1
    # NaNs from zero norms propagate to the comparison, as in the original Gram.
    tl.store(Pool + b*PB + h*PH + block*PN + cols*PD, pooled)
    tl.store(Sim + b*SB + h*SH + block*SN, similarity)
    scale = tl.max(tl.abs(xf)) / 127. + 0.0000001
    codes = xf / scale
    codes += 0.5 * tl.where(codes >= 0, 1, -1)
    tl.store(Quant + b * QB + h * QH + rows[:, None] * QN + cols[None, :],
             codes.to(tl.int8), mask=rows[:, None] < N)
    tl.store(Scale + b*CB + h*CH + block*CN, scale)


def pooled_quantized(x, mean, block_size):
    b, h, n, d = x.shape
    blocks = triton.cdiv(n, block_size)
    pool = torch.empty((b, h, blocks, d), dtype=x.dtype, device=x.device)
    similar = torch.empty((b, h, blocks), dtype=torch.bool, device=x.device)
    quant = torch.empty_strided((b, h, n, d), (h*n*d, n*d, d, 1),
                               dtype=torch.int8, device=x.device)
    scale = torch.empty((b, h, blocks), dtype=torch.float32, device=x.device)
    pool_quant[(b, h, blocks)](x, mean, pool, similar, quant, scale,
        N=n, D=d, BS=block_size, XB=x.stride(0), XH=x.stride(1), XN=x.stride(2),
        QB=quant.stride(0), QH=quant.stride(1), QN=quant.stride(2),
        PB=pool.stride(0), PH=pool.stride(1), PN=pool.stride(2), PD=pool.stride(3),
        SB=similar.stride(0), SH=similar.stride(1), SN=similar.stride(2),
        CB=scale.stride(0), CH=scale.stride(1), CN=scale.stride(2),
        MB=0 if mean is None else mean.stride(0),
        MH=0 if mean is None else mean.stride(1),
        MD=0 if mean is None else mean.stride(3),
        CENTER=mean is not None, num_warps=8 if block_size == 128 else 4)
    return pool, similar, quant, scale


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@torch.library.custom_op('sdacc_flux_sparge::block_rank', mutates_args=())
def block_rank(qp: torch.Tensor, kp: torch.Tensor, ksim: torch.Tensor) -> torch.Tensor:
    # Keep the pinned eager BF16 rounding and CUDA sort tie behavior together.
    # Inductor otherwise removes BF16 round trips and uses a different sorter,
    # which changes top-k membership even when the float scores are close.
    scores = (qp @ kp.transpose(-1, -2)) * (128**-.5)
    key_ok = ksim.unsqueeze(-2).expand(-1, -1, 12, -1)
    scores[~key_ok] = -torch.inf
    scores = scores.softmax(-1)
    return torch.sort(scores, dim=-1, descending=True).indices


@block_rank.register_fake
def fake_block_rank(qp, kp, ksim):
    return torch.empty((*qp.shape[:-1], kp.shape[-2]), dtype=torch.int64, device=qp.device)


@triton.jit
def selected_lut(Indices, QSim, KSim, Lut, Valid,
                 IH: tl.constexpr, IQ: tl.constexpr, IK: tl.constexpr,
                 QH: tl.constexpr, QQ: tl.constexpr,
                 KH: tl.constexpr, KK: tl.constexpr,
                 LH: tl.constexpr, LQ: tl.constexpr, LK: tl.constexpr,
                 VH: tl.constexpr, VQ: tl.constexpr, SELECT: tl.constexpr):
    # Fixed B=1 geometry. Store every unused LUT entry as zero, matching upstream.
    h, q = tl.program_id(0), tl.program_id(1)
    key = tl.arange(0, 32)
    ranked = tl.load(Indices+h*IH+q*IQ+key*IK, mask=key<24, other=-1)
    chosen = tl.sum(((key[:, None] == ranked[None, :]) &
                     (tl.arange(0, 32)[None, :] < SELECT)).to(tl.int32), 1) > 0
    qok = tl.load(QSim+h*QH+q*QQ)
    kok = tl.load(KSim+h*KH+key*KK, mask=key<24, other=True)
    keep = (key<24) & (chosen | (kok == 0) | (qok == 0))
    prefix = tl.cumsum(keep.to(tl.int32), 0)
    previous = tl.max(tl.where(keep[None, :] & (key[None, :] < key[:, None]),
                               key[None, :], 0), 1)
    base = Lut+h*LH+q*LQ
    tl.store(base+key*LK, 0, mask=key<24)
    tl.debug_barrier()
    tl.store(base+(prefix-1)*LK, key-previous, mask=(keep != 0))
    tl.store(Valid+h*VH+q*VQ, tl.sum(keep.to(tl.int32), 0))


def prepare(q, k, v, topk):
    if q.shape != (1, 1536, 24, 128) or topk not in (.5, 1.):
        raise ValueError('only the authorized FLUX geometry and strengths are supported')
    qh, kh = q.transpose(1, 2), k.transpose(1, 2)
    vh = v.transpose(1, 2).to(dtype=torch.float16, memory_format=torch.contiguous_format)
    km = kh.mean(-2, keepdim=True)
    qp, qsim, qi, qs = pooled_quantized(qh, None, 128)
    kp, ksim, ki, ks = pooled_quantized(kh, km, 64)
    indices = block_rank(qp, kp, ksim)
    lut = torch.empty((1, 24, 12, 24), dtype=torch.int32, device=q.device)
    valid = torch.empty((1, 24, 12), dtype=torch.int32, device=q.device)
    selected_lut[(24, 12)](indices, qsim, ksim, lut, valid,
        IH=indices.stride(1), IQ=indices.stride(2), IK=indices.stride(3),
        QH=qsim.stride(1), QQ=qsim.stride(2), KH=ksim.stride(1), KK=ksim.stride(2),
        LH=lut.stride(1), LQ=lut.stride(2), LK=lut.stride(3),
        VH=valid.stride(1), VQ=valid.stride(2), SELECT=int(topk*24), num_warps=4)
    pv = torch.full((24,), 50., device=q.device)
    return qi, ki, vh, lut, valid, pv, qs, ks
