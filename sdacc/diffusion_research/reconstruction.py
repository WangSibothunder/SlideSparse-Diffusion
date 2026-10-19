"""Fixed-order, shared-curvature pruning with optional joint INT8 rounding.

Derived from OBS-Diff fasterprune (f0b5fd175a9686839f606b6d77e05710163f8080).
The joint variant implements the existing SparseGPT+GPTQ-style error rule;
it is neither official ExactOBC nor a reproduction of full OBR.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Curvature:
    upper: torch.Tensor
    dead: torch.Tensor
    damping: float


@torch.inference_mode()
def factor_hessian(hessian, damping=.01):
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("Hessian must be square")
    h = hessian.detach().float().clone()
    if not torch.isfinite(h).all() or damping < 0:
        raise ValueError("invalid Hessian or damping")
    if not torch.allclose(h, h.T, rtol=1e-5, atol=1e-6):
        raise ValueError("Hessian is not symmetric")
    dead = h.diag() == 0
    if dead.any() and torch.count_nonzero(h[dead]):
        raise ValueError("zero-diagonal Hessian has nonzero cross terms")
    h[dead, dead] = 1
    amount = damping * h.diag().mean()
    h.diagonal().add_(amount)
    # Match OBS-Diff's shared upper Cholesky of the inverse. A non-SPD
    # input fails explicitly; no adaptive damping or hidden +I fallback.
    lower = torch.linalg.cholesky(h)
    inverse = torch.cholesky_inverse(lower)
    upper = torch.linalg.cholesky(inverse, upper=True)
    return Curvature(upper, dead, float(amount))


@torch.inference_mode()
def channel_scales(weight):
    """Freeze the same signed symmetric grid for both comparison methods.

    This is the OBC symmetric 8-bit grid: offset128, integer range[-128,127],
    width255. Entirely zero rows use scale1 rather than a zero denominator.
    """
    maximum = weight.detach().float().abs().amax(dim=1)
    return torch.where(maximum > 0, 2 * maximum / 255, torch.ones_like(maximum))


def round_signed(value, scales):
    return torch.round(value / scales).clamp(-128, 127)


@torch.inference_mode()
def compress(weight, curvature, group, *, method="joint", blocksize=192,
             scales=None, allow_tail=False, progress=None):
    """All output rows share one factor; column blocks are updated by GEMM.

    group4/6/8 means exactly two selected zeros in each complete group.
    The diagnostic prune_then_rtn and joint use identical initial scales.
    Updates to not-yet-finalized columns are temporary. A selected zero is
    always finalized as zero, including after final quantization/export.
    """
    if method not in ("joint", "prune_then_rtn") or group not in (4, 6, 8):
        raise ValueError("unsupported method or sparsity group")
    if blocksize <= 0 or blocksize % group:
        raise ValueError("column block would split an N:M group")
    if weight.ndim != 2 or not weight.numel() or not torch.isfinite(weight).all():
        raise ValueError("invalid weight")
    rows, columns = weight.shape
    if columns % group and not allow_tail:
        raise ValueError("full model groups required; tail handling is explicit")
    if (curvature.upper.shape != (columns, columns)
            or curvature.upper.device != weight.device):
        raise ValueError("curvature input width/device mismatch")
    w = weight.detach().float().clone()
    w[:, curvature.dead] = 0
    scales = channel_scales(weight) if scales is None else scales.detach().float().clone()
    if (scales.shape != (rows,) or scales.device != weight.device
            or not torch.isfinite(scales).all() or not (scales > 0).all()):
        raise ValueError("invalid per-output-channel scales")
    structural_zero = torch.zeros_like(w, dtype=torch.bool)
    losses = torch.zeros(rows, device=w.device)
    upper = curvature.upper
    for start in range(0, columns, blocksize):
        stop = min(start + blocksize, columns)
        local = w[:, start:stop].clone()
        finalized = torch.zeros_like(local)
        errors = torch.zeros_like(local)
        mask = torch.zeros_like(local, dtype=torch.bool)
        factor = upper[start:stop, start:stop]
        for index in range(stop - start):
            if index % group == 0:
                width = min(group, stop - start - index)
                scores = local[:, index:index+width].square() / factor.diag()[index:index+width].square()
                # Missing lanes are zero padding in the public packer's last
                # group. Count them before removing any real input column.
                remove = max(0, 2 - (group - width))
                removed = torch.topk(scores, remove, dim=1, largest=False).indices
                mask.scatter_(1, index + removed, True)
            current = local[:, index].clone()
            final = (round_signed(current, scales) * scales
                     if method == "joint" else current.clone())
            final.masked_fill_(mask[:, index], 0)
            finalized[:, index] = final
            error = (current - final) / factor[index, index]
            losses.add_(error.square() / 2)
            local[:, index:].sub_(error[:, None] @ factor[index, index:][None, :])
            errors[:, index] = error
        w[:, start:stop] = finalized
        structural_zero[:, start:stop] = mask
        if stop < columns:
            w[:, stop:].sub_(errors @ upper[start:stop, stop:])
        if progress is not None:
            progress(finalized_columns=stop, total_columns=columns)
    structural_zero[:, curvature.dead] = True
    q = round_signed(w, scales[:, None]).to(torch.int8)
    q.masked_fill_(structural_zero, 0)
    if not torch.isfinite(w).all() or not torch.isfinite(scales).all():
        raise RuntimeError("nonfinite compensation")
    if not torch.equal(q.float() * scales[:, None], w) and method == "joint":
        raise RuntimeError("joint output left the frozen quantization grid")
    complete = columns // group * group
    if complete and not (q[:, :complete].reshape(rows, -1, group).count_nonzero(-1) <= group - 2).all():
        raise RuntimeError("structural sparsity violation")
    return dict(qweight=q, scales=scales, zeros=structural_zero,
                compensated=w, proxy_loss=losses, method=method,
                group=group, blocksize=blocksize,
                weight_grid="signed[-128,127]; scale=2*Dense-row-absmax/255",
                curvature_damping=curvature.damping,
                activation_error_compensated=False)
