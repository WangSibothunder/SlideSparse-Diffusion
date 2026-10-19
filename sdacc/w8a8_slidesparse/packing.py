"""Exact, offline conversion of (L-2):L into overlapping hardware 2:4."""
import torch
from torch.nn import functional as F

FORMATS = {"2:4": 4, "4:6": 6, "6:8": 8}


def group_size(mode):
    try:
        return FORMATS[mode] if isinstance(mode, str) else {4: 4, 6: 6, 8: 8}[mode]
    except KeyError as e:
        raise ValueError("format must be 2:4, 4:6, or 6:8") from e


def lifted_k(k, mode, alignment=128):
    if k <= 0 or alignment <= 0 or alignment % 4:
        raise ValueError("positive K and a multiple-of-four alignment required")
    l = group_size(mode)
    valid = (k + l - 1) // l * (l // 2 - 1) * 4
    return valid, (valid + alignment - 1) // alignment * alignment


@torch.no_grad()
def slide_weight(qweight, mode):
    """Preserve every nonzero exactly once; never prune or requantize here.

    Follow official SlideSparse's ordered greedy allocation: increasing window,
    then increasing lane. Magnitude-based window allocation is NOT equivalent.
    Structural zeros, including quantized-to-zero retained weights, stay zero.
    """
    if qweight.device.type != "cpu" or qweight.dtype != torch.int8 or qweight.ndim != 2:
        raise TypeError("packing requires a CPU INT8 [out_features,in_features] tensor")
    n, k = qweight.shape
    if n <= 0 or k <= 0:
        raise ValueError("empty weights are unsupported")
    l = group_size(mode)
    groups = F.pad(qweight, (0, (-k) % l)).reshape(n, -1, l)
    if torch.any(torch.count_nonzero(groups, dim=-1) > l - 2):
        raise ValueError("weight violates the frozen (L-2):L structure")
    windows = l // 2 - 1
    allocated = torch.zeros_like(groups, dtype=torch.bool)
    out = torch.zeros((*groups.shape[:2], windows, 4), dtype=torch.int8)
    for w in range(windows):
        start = 2 * w
        values = groups[..., start:start + 4]
        pending = (values != 0) & ~allocated[..., start:start + 4]
        keep = pending & (pending.cumsum(-1) <= 2)
        out[..., w, :] = torch.where(keep, values, 0)
        allocated[..., start:start + 4] |= keep
    if torch.any((groups != 0) & ~allocated):
        raise RuntimeError("slide failed to allocate a retained weight")
    return out.reshape(n, -1).contiguous()
