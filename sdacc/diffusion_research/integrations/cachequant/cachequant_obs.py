"""Offline native-grid OBS-Diff compensation; no caller/speed admission.

Reuse the accepted fixed-order curvature factor and block error propagation.
Start at CacheQuant's already quantized BF16/FP32 weights, retain its exact
division/round/zero-point/dequantization boundaries, then choose two deleted
columns per group. A fresh own Hessian from the actual quantized-input chain
is required by the model producer; old symmetric banks are not inputs here.
"""
import importlib.util
from pathlib import Path
import torch
from .evidence import load, sha


def accepted_factor_module(provenance):
    """Load the byte-preserved previous implementation, never modify it."""
    provenance = Path(provenance)
    declaration = load(provenance / 'SOURCE.json')
    source = provenance / 'obsdiff_joint.py'
    if sha(source) != declaration['sha256']:
        raise ValueError('accepted OBS-Diff factor source SHA changed')
    spec = importlib.util.spec_from_file_location('sigma_private_obs_factor', source)
    module = importlib.util.module_from_spec(spec)
    # dataclass resolves its module during import.
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _native_grid(grid, device):
    codes, delta, zero = (grid[k] for k in ('codes', 'delta', 'zero'))
    if (codes.dtype != torch.uint8 or codes.ndim != 2 or not codes.numel()
        or delta.dtype not in (torch.float32, torch.bfloat16)
        or delta.dtype != zero.dtype or tuple(delta.shape) != (codes.shape[0], 1)
        or tuple(zero.shape) != tuple(delta.shape)
        or grid['native_dtype'] != str(delta.dtype)
        or grid['original_shape'] != list(codes.shape)):
        raise ValueError('actual native per-output Linear grid before format conversion required')
    if (not torch.isfinite(delta).all() or not (delta > 0).all()
        or not torch.isfinite(zero).all()
        or not torch.equal(zero, zero.round())
        or not ((zero >= 0) & (zero <= 255)).all()):
        raise ValueError('unchanged native grid must encode real zero; no symmetric or epsilon replacement')
    return codes.to(device), delta.to(device), zero.to(device)


def native_encode(value, delta, zero):
    """Native BF16 division and round must precede adding the native zero."""
    dtype = delta.dtype
    return (torch.round(value.to(dtype) / delta) + zero).clamp(0, 255).to(torch.uint8)


def native_decode(codes, delta, zero):
    return (codes.to(delta.dtype) - zero) * delta


@torch.inference_mode()
def compress_native_grid(grid, curvature, group, *, blocksize=192, progress=None):
    """Fixed-column shared-curvature OBS on an unchanged asymmetric grid.

    Two removals in each complete group4/6/8 mean 2:4/4:6/6:8. Sigma has no
    partial groups; reject tails. Natural quantized zeros and dead-curvature
    columns are reported separately from the selected structural budget.
    This offline builder is not an INT8 kernel or a complete projection.
    """
    if group not in (4, 6, 8) or type(blocksize) is not int or blocksize <= 0 or blocksize % group:
        raise ValueError('explicit supported complete groups and aligned column block required')
    upper = curvature.upper
    q, delta, zero = _native_grid(grid, upper.device)
    rows, columns = q.shape
    if columns % group:
        raise ValueError('full model grouping required; no implicit tail pruning')
    if (upper.dtype != torch.float32 or upper.shape != (columns, columns)
        or curvature.dead.dtype != torch.bool or curvature.dead.shape != (columns,)
        or curvature.dead.device != upper.device or not torch.isfinite(upper).all()
        or not (upper.diag() > 0).all()
        or torch.count_nonzero(torch.tril(upper, diagonal=-1))):
        raise ValueError('accepted finite upper inverse-Cholesky and exact dead-column plan required')
    w = native_decode(q, delta, zero).float().clone()
    w[:, curvature.dead] = 0
    result = torch.empty_like(q)
    structural = torch.zeros_like(q, dtype=torch.bool)
    losses = torch.zeros(rows, device=w.device)
    for start in range(0, columns, blocksize):
        stop = min(start + blocksize, columns)
        local = w[:, start:stop].clone()
        errors = torch.zeros_like(local)
        factor = upper[start:stop, start:stop]
        mask = torch.zeros_like(local, dtype=torch.bool)
        for index in range(stop - start):
            if index % group == 0:
                scores = local[:, index:index + group].square() / factor.diag()[index:index + group].square()
                if not torch.isfinite(scores).all():
                    raise ValueError('nonfinite OBS selection score')
                removed = torch.topk(scores, 2, dim=1, largest=False).indices
                mask.scatter_(1, index + removed, True)
            current = local[:, index].clone()
            final_codes = native_encode(current, delta[:, 0], zero[:, 0])
            final_codes = torch.where(mask[:, index], zero[:, 0].to(torch.uint8), final_codes)
            final = native_decode(final_codes, delta[:, 0], zero[:, 0]).float()
            result[:, start + index] = final_codes
            error = (current - final) / factor[index, index]
            losses.add_(error.square() / 2)
            local[:, index:].sub_(error[:, None] @ factor[index, index:][None, :])
            errors[:, index] = error
        structural[:, start:stop] = mask
        if stop < columns:
            w[:, stop:].sub_(errors @ upper[start:stop, stop:])
        if progress is not None:
            progress(finalized_columns=stop, total_columns=columns)
    # Curvature-dead inputs contribute no error and have explicit native zero.
    result[:, curvature.dead] = zero.to(torch.uint8)
    effective = structural | curvature.dead[None, :]
    if (not (structural.reshape(rows, -1, group).sum(-1) == 2).all()
        or not torch.equal(result[effective], zero.expand_as(result).to(torch.uint8)[effective])):
        raise AssertionError('selected removal does not encode native zero point')
    if not torch.isfinite(losses).all():
        raise ValueError('nonfinite compensation; no silent fallback')
    return dict(codes=result, delta=delta.clone(), zero=zero.clone(), structural_zero=structural,
        dead_columns=curvature.dead.clone(), effective_forced_zero=effective,
        compensated=native_decode(result, delta, zero), proxy_loss=losses,
        group=group, blocksize=blocksize, curvature_damping=curvature.damping,
        native_dtype=grid['native_dtype'], original_shape=list(q.shape),
        starting_codes_SHA_required_by_producer=True,
        weights_quantized_before_OBS=True, unchanged_native_grid=True,
        no_grid_refit_or_symmetric_substitution=True, actual_INT8_numeric_trace_cost_pending=True,
        actual_model_calibration_and_quality_not_proven=True)
