"""Private native asymmetric INT8/I32 Dense deployment, admission pending.

Reuse the two signed planes and ATen _int_mm from the accepted006 hardware
prototype; extend unpruned wide-zero weights, physical tails and rank4 Conv.
Correction happens in I32, before FP32 scaling (old006 loses small corrected
integers above2**24). Original native BF16 round_ste operation order remains
inside an opaque custom op. This is not N:M sparse acceleration, full-model
acceptance, or a measured cost class. Every real bank/caller still needs gates.
"""
import torch
from typing import List
from torch.nn import functional as F
from .cachequant_dense_grid import centred_planes, validate_dense_or_sparse_grid


def native_round(value):
    # Pinned CacheQuant quant_layer.round_ste under inference/no gradients.
    return (value.round() - value).detach() + value


def encode_native(value, scale, zero):
    return (native_round(value / scale) + native_round(zero)).clamp(0, 255)


def integer_correct(accum, row_sum, zero):
    return accum - (native_round(zero).to(torch.int32) - 128) * row_sum


def prepare_planes(grid, declaration):
    validate_dense_or_sparse_grid(grid, declaration)
    # Quantization/range search preceded flattening, especially for patchConv.
    codes=grid['codes'].reshape(grid['codes'].shape[0],-1)
    zero=grid['zero'].reshape(-1,1)
    low,high=centred_planes(codes,zero)
    d=codes.long()-zero.long()
    n,k=d.shape
    if k>32768:raise ValueError('native Dense logicalK exceeds explicit integer bound')
    low_abs=low.long().abs().sum(1);high_abs=high.long().abs().sum(1);d_abs=d.abs().sum(1)
    limit=torch.iinfo(torch.int32).max
    if ((128*low_abs>limit).any() or (128*256*high_abs>limit).any() or (128*d_abs>limit).any()):
        raise ValueError('INT8 plane or combined I32 accumulator overflow bound not proven')
    kp=(-k)%32;np=(-n)%8
    return dict(low=F.pad(low,(0,kp,0,np)).T.contiguous(),high=F.pad(high,(0,kp,0,np)).T.contiguous(),
        row_sum=d.sum(1).to(torch.int32),weight_abs_sum=d_abs,logical_n=n,logical_k=k,
        native_weight_zero=grid['zero'].clone(),scale=grid['delta'].float().reshape(-1),
        physical_integer0_tails_not_native_deleted_codes=True,centred_max=int(d.max()),centred_min=int(d.min()))


def validate_activation_tables(scales,zeros,weight_abs_sum):
    if (scales.shape!=(20,) or zeros.shape!=(20,) or scales.dtype!=zeros.dtype
        or scales.dtype not in (torch.bfloat16,torch.float32)
        or scales.device.type!='cpu' or zeros.device.type!='cpu'
        or not torch.isfinite(scales).all() or not (scales>0).all()
        or not torch.isfinite(zeros).all() or scales.requires_grad or zeros.requires_grad):
        raise ValueError('frozen native per-tensor20 CPU tables of original dtype required')
    rounded=native_round(zeros).double()
    if not ((rounded>=-32768)&(rounded<=32767)).all():
        raise ValueError('native activation zero outside explicit deployment bounds; independent format needed')
    distance=torch.maximum(rounded.abs(),(255-rounded).abs())
    if (distance.max()*weight_abs_sum.double()>torch.iinfo(torch.int32).max).any():
        raise ValueError('activation zero correction exceeds explicit I32 bound')


def _dot(a,low,high,row_sum,scale,bias,asc,az):
    rows,k=a.shape;n=len(bias)
    a=F.pad(a,(0,low.shape[0]-k,0,(-rows)%8)).contiguous()
    # These two calls are genuine signed INT8 GEMMs with I32 outputs. Their
    # existence/numerics/reload/compiled stability/CUDA trace need real gates.
    accum=torch._int_mm(a,low)+256*torch._int_mm(a,high)
    corrected=integer_correct(accum[:rows,:n],row_sum,az)
    return (corrected.float()*(asc.float()*scale)+bias).to(torch.bfloat16)


@torch.library.custom_op('sigma_orthogonal_native_dense_i32::linear',mutates_args=())
def linear(x:torch.Tensor,low:torch.Tensor,high:torch.Tensor,row_sum:torch.Tensor,
           scale:torch.Tensor,bias:torch.Tensor,asc:torch.Tensor,az:torch.Tensor)->torch.Tensor:
    q=encode_native(x.reshape(-1,x.shape[-1]),asc,az)
    a=(q.to(torch.int16)-128).to(torch.int8)
    return _dot(a,low,high,row_sum,scale,bias,asc,az).reshape(*x.shape[:-1],len(bias))


@linear.register_fake
def fake_linear(x,low,high,row_sum,scale,bias,asc,az):
    return torch.empty((*x.shape[:-1],len(bias)),device=x.device,dtype=torch.bfloat16)


@torch.library.custom_op('sigma_orthogonal_native_dense_i32::conv2d',mutates_args=())
def conv2d(x:torch.Tensor,low:torch.Tensor,high:torch.Tensor,row_sum:torch.Tensor,
           scale:torch.Tensor,bias:torch.Tensor,asc:torch.Tensor,az:torch.Tensor,
           kernel:List[int],stride:List[int],padding:List[int],dilation:List[int])->torch.Tensor:
    # Native activation quantization precedes original Conv. Unfold/padding
    # and float32 layout conversion are explicit runtime operations/costs.
    q=encode_native(x,asc,az);a=(q.to(torch.int16)-128).to(torch.int8)
    if any(padding):
        z=int(native_round(az).item())
        if not 0<=z<=255:raise ValueError('nonzero Conv padding needs representable native activation zero')
        a=F.pad(a,(padding[1],padding[1],padding[0],padding[0]),value=z-128)
    patches=F.unfold(a.float(),kernel,dilation=dilation,padding=0,stride=stride)
    b,k,l=patches.shape
    packed=patches.transpose(1,2).reshape(b*l,k).to(torch.int8)
    y=_dot(packed,low,high,row_sum,scale,bias,asc,az)
    h=(x.shape[2]+2*padding[0]-dilation[0]*(kernel[0]-1)-1)//stride[0]+1
    w=(x.shape[3]+2*padding[1]-dilation[1]*(kernel[1]-1)-1)//stride[1]+1
    return y.reshape(b,h*w,len(bias)).transpose(1,2).reshape(b,len(bias),h,w).contiguous()


@conv2d.register_fake
def fake_conv2d(x,low,high,row_sum,scale,bias,asc,az,kernel,stride,padding,dilation):
    h=(x.shape[2]+2*padding[0]-dilation[0]*(kernel[0]-1)-1)//stride[0]+1
    w=(x.shape[3]+2*padding[1]-dilation[1]*(kernel[1]-1)-1)//stride[1]+1
    return torch.empty((x.shape[0],len(bias),h,w),device=x.device,dtype=torch.bfloat16)


class NativeDenseInt32(torch.nn.Module):
    """Fixed-buffer Dense frontend, no online MSE or hidden parameter fit.

    select(step) returns both actual D2D copies/bytes. A model integration must
    call it in Timer.select, and demonstrate same-address换值/reload/trace.
    Constructor/H2D and the extra buffers are separately declared, not costs
    inherited from a BF16 oracle or from an equal matrix geometry.
    """
    def __init__(self,grid,declaration,bias,activation_scales,activation_zeros,
                 device='cpu',conv=None):
        super().__init__();planes=prepare_planes(grid,declaration)
        validate_activation_tables(activation_scales,activation_zeros,planes['weight_abs_sum'])
        n,k=planes['logical_n'],planes['logical_k'];self.logical_n,self.logical_k=n,k
        if bias is None:bias=torch.zeros(n,dtype=torch.float32)
        if bias.shape!=(n,) or not torch.isfinite(bias).all():raise ValueError('original full output bias required')
        if len(declaration['shape'])==4:
            if conv is None or conv.get('groups')!=1 or conv.get('padding_mode')!='zeros':
                raise ValueError('original rank4 Conv groups1/zero-padding declaration required')
            self.conv={key:list(conv[key]) for key in ('stride','padding','dilation')}
            self.conv['kernel']=declaration['shape'][2:]
            if any(self.conv['padding']) and not ((native_round(activation_zeros)>=0)&(native_round(activation_zeros)<=255)).all():
                raise ValueError('native activation zero cannot encode Conv real-zero padding')
        else:
            if conv is not None:raise ValueError('Linear cannot acquire a Conv layout')
            self.conv=None
        values={key:planes[key] for key in ('low','high','row_sum','scale')}
        values.update(bias=bias.float(),activation_scales=activation_scales.clone(),activation_zeros=activation_zeros.clone(),
            current_scale=activation_scales[0].clone(),current_zero=activation_zeros[0].clone())
        for name,value in values.items():self.register_buffer(name,value.to(device))
        self.runtime_gate_pending=True

    def select(self,step):
        if type(step) is not int or not 0<=step<20:raise ValueError('declared20 step only')
        self.current_scale.copy_(self.activation_scales[step]);self.current_zero.copy_(self.activation_zeros[step])
        return dict(parameter_copy_count=2,parameter_copy_bytes=self.current_scale.element_size()+self.current_zero.element_size(),
            destination_addresses=(self.current_scale.data_ptr(),self.current_zero.data_ptr()),required_timer='Timer.select')

    def forward(self,x):
        if x.device.type!='cuda' or self.low.device!=x.device or x.dtype!=self.current_scale.dtype:
            raise ValueError('actual CUDA input/native dtype required; no BF16 GEMM fallback')
        if self.conv is None:
            if x.shape[-1]!=self.logical_k:raise ValueError('original Linear logicalK changed')
            return linear(x,self.low,self.high,self.row_sum,self.scale,self.bias,self.current_scale,self.current_zero)
        if x.ndim!=4 or x.shape[1]*self.conv['kernel'][0]*self.conv['kernel'][1]!=self.logical_k:
            raise ValueError('original NCHW Conv logical layout required')
        return conv2d(x,self.low,self.high,self.row_sum,self.scale,self.bias,self.current_scale,self.current_zero,
            self.conv['kernel'],self.conv['stride'],self.conv['padding'],self.conv['dilation'])
