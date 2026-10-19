"""Native uint8 codes with original wide zero, for unpruned modules only.

Sigma's patch Conv has a native zero=-31 row. Retain it; the original codes
remain uint8 while q-z can reach286, exactly represented by signed INT8 planes.
This storage/oracle adapter does not admit a CUDA caller. Sparse grids still
require representable real zero and cannot use this adapter for deletion.
"""
import torch
from .cachequant_calibration import make_quantizer, _grid
from .cachequant_weight_journal import validate_weight_grid
from .storage import tensor_sha


@torch.no_grad()
def dense_weight_grid(weight,native_class):
    if weight.ndim not in (2,4) or not torch.isfinite(weight).all():raise ValueError('original finite Linear/Conv required')
    q=make_quantizer('weight',native_class);native=q(weight)
    delta,zero=q.delta.detach(),q.zero_point.detach();_grid(delta,zero)
    if delta.shape!=zero.shape or delta.shape!=(weight.shape[0],*([1]*(weight.ndim-1))):raise ValueError('native per-output grid required')
    codes=(torch.round(weight/delta)+zero.round()).clamp(0,255)
    if not torch.equal((codes-zero.round())*delta,native):raise ValueError('native rounding/dequantization changed')
    return dict(codes=codes.to(torch.uint8).cpu(),delta=delta.cpu().clone(),zero=zero.cpu().clone(),
        native_dtype=str(weight.dtype),original_shape=list(weight.shape),gradients=False,optional_reconstruction=False,
        source='native original full weight, unpruned Dense-format export; no zero-point clamp',
        native_uint8_codes_with_original_extended_zero=True,real_INT8_output_and_OBS_numeric_pending=True)


def centred_planes(codes,zero):
    z=zero.round()
    if not torch.isfinite(z).all() or not ((z>=-32768)&(z<=32767)).all():
        raise ValueError('native zero outside explicit two-plane format bounds; independent format needed')
    d=codes.long()-z.long();low=((d+128)%256-128).to(torch.int8);high=(d-low.long())//256
    if not ((high>=-128)&(high<=127)).all():raise ValueError('centred native weight exceeds declared INT8-plane format')
    high=high.to(torch.int8)
    if not torch.equal(d,low.long()+256*high.long()):raise AssertionError('native integer plane loss')
    return low,high


def validate_dense_or_sparse_grid(grid,declaration):
    z=grid['zero']
    if torch.isfinite(z).all() and ((z>=0)&(z<=255)).all():return validate_weight_grid(grid,declaration)
    if declaration.get('sparse_member') is not None:
        raise ValueError('wide native zero cannot authorize uint8 N:M deletion; no symmetric substitution')
    # Apply all original geometry/dtype/finiteness/device rules through a
    # separate validation-only zero tensor; actual zero is never changed.
    proxy=dict(grid,zero=torch.zeros_like(z));fmt=validate_weight_grid(proxy,declaration)
    if not torch.isfinite(z).all() or not torch.equal(z,z.round()):raise ValueError('finite native rounded zero required')
    low,high=centred_planes(grid['codes'],z)
    fmt.update(zero_sha=tensor_sha(z),centred_min=int((grid['codes'].long()-z.long()).min()),
        centred_max=int((grid['codes'].long()-z.long()).max()),two_signed_INT8_planes_exact=True,
        native_wide_zero_unpruned_only=True,Sparse_uint8_zero_not_encodable=True,
        real_INT8_fullcaller_numeric_trace_cost_pending=True)
    return fmt
