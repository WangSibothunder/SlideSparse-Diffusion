"""Private asymmetric-grid N:M INT8 prototype; acceptance is still required.

Two signed Sparse MMA integer planes exactly represent native q_w-z_w. Their
I32 accumulators combine before zero-point correction and the single BF16
output. The original per-timestep activation grid remains inside a functional
boundary. Integer gather/lift, two MMA calls and correction are runtime costs.
"""
import torch
from torch.nn import functional as F

def activation_indices(k, group):
    if group not in (4,6,8) or not 0 < k <= 32768:
        raise ValueError('Sigma integer bound and N:M mode')
    indices=[min(base+2*w+j,k) for base in range(0,k,group)
             for w in range(group//2-1) for j in range(4)]
    return torch.tensor(indices,dtype=torch.int64)

def lift_integer(a, indices):
    # An appended integer0 is the sentinel for physical tails, never a native
    # pruned weight code. Native pruned weights equal their weight zero points.
    source=F.pad(a,(0,1))
    lifted=source.index_select(1,indices)
    return F.pad(lifted,(0,(-lifted.shape[1])%128,0,(-len(lifted))%32)).contiguous()

def correct_accumulator(accum, row_sum, zero):
    # Zero-point subtraction must precede I32->FP32. Converting the two large
    # terms separately can lose integers above2**24 even though their final
    # difference is small. Logical K<=32768 keeps this centered dot in I32.
    return accum-(zero.to(torch.int32)-128)*row_sum

@torch.library.custom_op('sigma_orthogonal_asym_sparse::projection',mutates_args=())
def projection(x:torch.Tensor,low:torch.Tensor,low_meta:torch.Tensor,
               high:torch.Tensor,high_meta:torch.Tensor,indices:torch.Tensor,
               row_sum:torch.Tensor,scale:torch.Tensor,bias:torch.Tensor,
               activation_scale:torch.Tensor,activation_zero:torch.Tensor)->torch.Tensor:
    zero=activation_zero.round()
    q=(torch.round(x.reshape(-1,x.shape[-1])/activation_scale)+zero).clamp(0,255)
    a=(q.to(torch.int16)-128).to(torch.int8)
    lifted=lift_integer(a,indices)
    op=torch.ops.sigma_orthogonal_sparse_int32_sm80.run
    accum=op(lifted,low,low_meta,len(a),len(bias))+256*op(lifted,high,high_meta,len(a),len(bias))
    corrected=correct_accumulator(accum,row_sum,zero)
    y=corrected.float()*(activation_scale.float()*scale)+bias
    return y.to(torch.bfloat16).reshape(*x.shape[:-1],len(bias))

@projection.register_fake
def fake_projection(x,low,low_meta,high,high_meta,indices,row_sum,scale,bias,activation_scale,activation_zero):
    return torch.empty((*x.shape[:-1],bias.shape[0]),device=x.device,dtype=torch.bfloat16)

class NativeSparseLinear(torch.nn.Module):
    def __init__(self,qweight,scale,zero,bias,activation_scales,activation_zeros,group,slide_weight,lifted_k,device='cuda'):
        super().__init__()
        if qweight.device.type!='cpu' or qweight.dtype!=torch.uint8 or qweight.ndim!=2:
            raise ValueError('native unsigned frozen weight grid on CPU')
        n,k=qweight.shape;indices=activation_indices(k,group)
        d=qweight.to(torch.int16)-zero.round().to(torch.int16).reshape(-1,1)
        low=((d.to(torch.int32)+128)%256-128).to(torch.int8)
        high=((d-low.to(torch.int16))//256).to(torch.int8)
        valid,padded=lifted_k(k,group)
        if len(indices)!=valid:raise ValueError('lift index identity differs')
        native=torch.ops.sigma_orthogonal_sparse_int32_sm80
        packed_low,meta_low=native.pack_cpu(slide_weight(low,group),padded)
        packed_high,meta_high=native.pack_cpu(slide_weight(high,group),padded)
        for name,value in dict(low=packed_low,low_meta=meta_low,high=packed_high,high_meta=meta_high,
                indices=indices,row_sum=d.to(torch.int32).sum(-1).to(torch.int32),scale=scale.float().reshape(-1),bias=bias.float(),
                activation_scales=activation_scales,activation_zeros=activation_zeros).items():
            self.register_buffer(name,value.to(device))
        # Only explicit offline installation; no configure in forward.
        self.resources=native.configure(self.low);self.group=group;self.step=0
    def forward(self,x):
        return projection(x,self.low,self.low_meta,self.high,self.high_meta,self.indices,self.row_sum,self.scale,self.bias,
                          self.activation_scales[self.step],self.activation_zeros[self.step])
