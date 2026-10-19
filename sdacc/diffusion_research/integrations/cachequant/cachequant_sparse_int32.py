"""Native-grid Sparse I32 frontend, pending actual-bank CUDA acceptance.

Reuse the unchanged private CUTLASS I32 kernel, original sliding packer and
native BF16 encoder. Prove both signed integer planes reconstruct the complete
logical weight before packing. No grid refit, dense masked fallback or online
fit. Every dynamic activation/combined-affine copy is explicit in select().
"""
import torch
from .cachequant_weight_journal import validate_weight_grid
from .cachequant_dense_grid import centred_planes
from .cachequant_dense_int32 import encode_native,integer_correct,validate_activation_tables
from .native_sparse import activation_indices,lift_integer


def compose_affine(sparse_a,sparse_b,dec_a,dec_b):
    values=(sparse_a,sparse_b,dec_a,dec_b)
    if (any(not isinstance(v,torch.Tensor) or v.device.type!='cpu' or v.dtype!=torch.float32
            or v.requires_grad or not torch.isfinite(v).all() for v in values)
            or any(v.shape!=sparse_a.shape for v in values)):
        raise ValueError('explicit finite same-boundary CPU FP32 coefficients required')
    a=dec_a*sparse_a;b=dec_a*sparse_b+dec_b
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('nonfinite composed recovery')
    return a,b


@torch.inference_mode()
def sparse_format(grid,declaration,structural,group,slide_weight,lifted_k):
    validate_weight_grid(grid,declaration)
    if len(declaration['shape'])!=2:raise ValueError('actual eligible Linear only, not patch Conv')
    n,k=grid['codes'].shape
    if (group not in (4,6,8) or k%group or k>32768 or structural.device.type!='cpu'
            or structural.dtype!=torch.bool or structural.shape!=(n,k)
            or not (structural.reshape(n,-1,group).sum(-1)==2).all()):
        raise ValueError('complete original logical groups/two structural deletions required')
    z=grid['zero'].expand(n,k).to(torch.uint8)
    if not torch.equal(grid['codes'][structural],z[structural]):
        raise ValueError('native structural deletion must encode original weight zero point')
    low,high=centred_planes(grid['codes'],grid['zero'])
    indices=activation_indices(k,group);valid,padded=lifted_k(k,group)
    if len(indices)!=valid or (indices>=k).any() or padded<valid or padded%128:
        raise ValueError('original full-group lift/128-K padding identity changed')
    planes=[]
    for original in (low,high):
        slid=slide_weight(original,group)
        if slid.device.type!='cpu' or slid.dtype!=torch.int8 or slid.shape!=(n,valid) or not slid.is_contiguous():
            raise ValueError('original ordered CPU signed sliding plane required')
        if not (slid.reshape(n,-1,4).count_nonzero(-1)<=2).all():
            raise ValueError('hardware sliding plane exceeds two nonzeros per quartet')
        logical=torch.zeros(n,k,dtype=torch.int64)
        logical.scatter_add_(1,indices[None,:].expand(n,-1),slid.long())
        if not torch.equal(logical,original.long()):
            raise ValueError('integer sliding plane changed a retained logical weight')
        planes.append(slid)
    d=grid['codes'].long()-grid['zero'].long()
    bound=128*(planes[0].long().abs().sum(1)+256*planes[1].long().abs().sum(1))
    if (bound>torch.iinfo(torch.int32).max).any():
        raise ValueError('every partial/combined Sparse I32 accumulator bound must hold')
    return dict(slid_low=planes[0],slid_high=planes[1],indices=indices,valid_lifted_k=valid,padded_lifted_k=padded,
        row_sum=d.sum(1).to(torch.int32),weight_abs_sum=d.abs().sum(1),maximum_absolute_I32_bound=int(bound.max()),
        scale=grid['delta'].float().reshape(-1),native_weight_zero=grid['zero'].clone(),logical_n=n,logical_k=k,
        each_plane_exact_complete_logical_reconstruction=True,integer_zero_physical_tails_not_native_deleted_codes=True)


@torch.library.custom_op('sigma_orthogonal_native_sparse_i32::projection',mutates_args=())
def projection(x:torch.Tensor,low:torch.Tensor,low_meta:torch.Tensor,high:torch.Tensor,
               high_meta:torch.Tensor,indices:torch.Tensor,row_sum:torch.Tensor,scale:torch.Tensor,
               bias:torch.Tensor,asc:torch.Tensor,az:torch.Tensor,a:torch.Tensor,b:torch.Tensor)->torch.Tensor:
    rows=x.reshape(-1,x.shape[-1])
    codes=encode_native(rows,asc,az)
    signed=(codes.to(torch.int16)-128).to(torch.int8)
    lifted=lift_integer(signed,indices)
    op=torch.ops.sigma_orthogonal_sparse_int32_sm80.run
    accum=op(lifted,low,low_meta,len(rows),len(bias))+256*op(lifted,high,high_meta,len(rows),len(bias))
    corrected=integer_correct(accum,row_sum,az)
    raw=corrected.float()*(asc.float()*scale)+bias
    # One FP32 composed recovery boundary, before the only final BF16 cast.
    # Coefficients must later be separately calibrated/gated; identity values
    # are solely the raw-bank acceptance path, not an affine/DEC admission.
    return (raw*a+b).to(torch.bfloat16).reshape(*x.shape[:-1],len(bias))


@projection.register_fake
def fake_projection(x,low,low_meta,high,high_meta,indices,row_sum,scale,bias,asc,az,a,b):
    return torch.empty((*x.shape[:-1],len(bias)),device=x.device,dtype=torch.bfloat16)


class NativeSparseInt32(torch.nn.Module):
    """Stable four parameter slots, explicit offline prepare, no fallback.

    CPU construction/packing/select are format facts only. New CUDA compiled
    table/weight/affine swaps, full logical integer oracle, strict native .01
    numeric, reload/trace and complete caller cost are still mandatory.
    """
    def __init__(self,grid,declaration,structural,group,bias,activation_scales,activation_zeros,
                 slide_weight,lifted_k,affine=None,device='cpu'):
        super().__init__()
        if torch.is_grad_enabled():raise ValueError('explicit no-grad native Sparse installation required')
        fmt=sparse_format(grid,declaration,structural,group,slide_weight,lifted_k)
        validate_activation_tables(activation_scales,activation_zeros,fmt['weight_abs_sum'])
        n,k=fmt['logical_n'],fmt['logical_k']
        if bias is None:bias=torch.zeros(n,dtype=torch.float32)
        if bias.shape!=(n,) or not torch.isfinite(bias).all() or bias.requires_grad:
            raise ValueError('original full output bias required')
        if affine is None:
            aa=torch.ones(20,n,dtype=torch.float32);bb=torch.zeros_like(aa);scope='raw_identity_only'
        else:
            aa,bb=affine;scope='provided_combined_coefficients_not_yet_admitted'
        if (aa.shape!=(20,n) or bb.shape!=(20,n) or aa.device.type!='cpu' or bb.device.type!='cpu'
                or aa.dtype!=torch.float32 or bb.dtype!=torch.float32 or aa.requires_grad or bb.requires_grad
                or not torch.isfinite(aa).all() or not torch.isfinite(bb).all()):
            raise ValueError('frozen CPU FP32 full20/output-width coefficient tables required')
        native=torch.ops.sigma_orthogonal_sparse_int32_sm80
        packed_low,low_meta=native.pack_cpu(fmt['slid_low'],fmt['padded_lifted_k'])
        packed_high,high_meta=native.pack_cpu(fmt['slid_high'],fmt['padded_lifted_k'])
        values=dict(low=packed_low,low_meta=low_meta,high=packed_high,high_meta=high_meta,
            indices=fmt['indices'],row_sum=fmt['row_sum'],scale=fmt['scale'],bias=bias.float().clone(),
            native_weight_zero=fmt['native_weight_zero'],activation_scales=activation_scales.clone(),
            activation_zeros=activation_zeros.clone(),affine_a=aa.clone(),affine_b=bb.clone(),
            current_scale=activation_scales[0].clone(),current_zero=activation_zeros[0].clone(),
            current_a=aa[0].clone(),current_b=bb[0].clone())
        for name,value in values.items():self.register_buffer(name,value.to(device))
        self.logical_n,self.logical_k,self.group=n,k,group
        self.prepared=False;self.resources=None;self.slot_addresses=None;self.selected_step=None;self.selection_ready=False
        self.registration=dict(logical_n=n,logical_k=k,group=group,valid_lifted_k=fmt['valid_lifted_k'],
            padded_lifted_k=fmt['padded_lifted_k'],maximum_absolute_I32_bound=fmt['maximum_absolute_I32_bound'],
            reconstruction_before_packing=True,recovery_scope=scope,expected_D2D_slot_copies=4,
            extra_buffer_bytes=sum(v.numel()*v.element_size() for v in self.buffers()),
            native_asymmetric_grid_unchanged=True,no_BF16_GEMM_or_masked_dense_fallback=True,
            actual_bank_numeric_compiled_swaps_reload_realSparse_trace_completecost_modelquality_pending=True,
            formal_launch_ready=False,scientific_delivery_complete=False)

    def prepare_device(self):
        if self.prepared:raise ValueError('successful offline Sparse resource preparation cannot repeat')
        if self.low.device.type!='cuda':raise ValueError('actual CUDA resources only, CPU format is not prepare')
        self.resources=torch.ops.sigma_orthogonal_sparse_int32_sm80.configure(self.low)
        # CPU format addresses are not the new CUDA allocation epoch. Reset
        # only this offline pointer audit; actual tensor values stay intact.
        self.slot_addresses=None;self.selected_step=None;self.selection_ready=False
        self.prepared=True
        return self.resources

    @torch.inference_mode()
    def select(self,step):
        if type(step) is not int or not 0<=step<20:raise ValueError('declared full20 step required')
        self.current_scale.copy_(self.activation_scales[step]);self.current_zero.copy_(self.activation_zeros[step])
        self.current_a.copy_(self.affine_a[step]);self.current_b.copy_(self.affine_b[step])
        values=(self.current_scale,self.current_zero,self.current_a,self.current_b)
        addresses=tuple(v.data_ptr() for v in values)
        if len(set(addresses))!=4:raise ValueError('native/affine parameter slots alias')
        if self.slot_addresses is None:self.slot_addresses=addresses
        elif self.slot_addresses!=addresses:raise ValueError('native/affine slot addresses changed')
        self.selected_step=step
        self.selection_ready=True
        return dict(parameter_copy_count=4,parameter_copy_bytes=sum(v.numel()*v.element_size() for v in values),
                    destination_addresses=addresses,required_timer='Timer.select',step=step)

    def forward(self,x):
        if (not self.prepared or not self.selection_ready or x.device.type!='cuda' or x.device!=self.low.device
                or x.dtype!=self.current_scale.dtype or x.shape[-1]!=self.logical_k):
            raise ValueError('actual prepared CUDA/native dtype/full logical input and selected slots required')
        return projection(x,self.low,self.low_meta,self.high,self.high_meta,self.indices,self.row_sum,
                          self.scale,self.bias,self.current_scale,self.current_zero,self.current_a,self.current_b)
