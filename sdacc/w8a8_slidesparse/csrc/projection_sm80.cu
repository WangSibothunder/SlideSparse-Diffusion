// Optional single-C++-call frontend. The qualified GEMM source stays unchanged.
#include "dualscale_sm80.cu"
#include <mutex>
#include <memory>
#include <unordered_map>

namespace sdacc_mild_projection_sm80 {
template<int K,int L,bool Vectorized=true>
__global__ void quant_lift(const at::BFloat16* x,int64_t sx,int8_t* q,float* scales,int m,int pk) {
  constexpr int Windows=L/2-1;
  constexpr int Lift=((K+L-1)/L)*Windows*4;
  constexpr int PK=((Lift+127)/128)*128;
  constexpr int Groups=(PK/4+Windows-1)/Windows;
  constexpr int PerThread=(Groups+127)/128;
  int row=blockIdx.x,t=threadIdx.x,lane=t&31,warp=t>>5;
  float values[PerThread][L];float maximum=0;
  #pragma unroll
  for(int i=0;i<PerThread;++i) {
    int g=t+128*i;
    if constexpr(L==4&&Vectorized) {
      const auto* base=reinterpret_cast<const uint16_t*>(x)+int64_t(row)*sx+g*4;
      if((reinterpret_cast<uintptr_t>(base)&7)==0) {
        uint2 bits=make_uint2(0,0);
        if(row<m&&g<K/4)bits=*reinterpret_cast<const uint2*>(base);
        values[i][0]=__uint_as_float((bits.x&65535u)<<16);
        values[i][1]=__uint_as_float(bits.x&0xffff0000u);
        values[i][2]=__uint_as_float((bits.y&65535u)<<16);
        values[i][3]=__uint_as_float(bits.y&0xffff0000u);
        #pragma unroll
        for(int j=0;j<4;++j)maximum=fmaxf(maximum,fabsf(values[i][j]));
      } else {
        // Unit column stride does not guarantee aligned row starts.
        #pragma unroll
        for(int j=0;j<4;++j) {
          float v=(row<m&&g*4+j<K)?__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(x)[int64_t(row)*sx+g*4+j]):0.f;
          values[i][j]=v;maximum=fmaxf(maximum,fabsf(v));
        }
      }
    } else {
      #pragma unroll
      for(int j=0;j<L;++j) {
        float v=(row<m&&g*L+j<K)?__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(x)[int64_t(row)*sx+g*L+j]):0.f;
        values[i][j]=v;maximum=fmaxf(maximum,fabsf(v));
      }
    }
  }
  #pragma unroll
  for(int d=16;d;d>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,d));
  __shared__ float partial[4];__shared__ float scale,inv;
  if(lane==0)partial[warp]=maximum;
  __syncthreads();
  if(t==0) {
    float a=fmaxf(fmaxf(partial[0],partial[1]),fmaxf(partial[2],partial[3]));
    scale=fmaxf(fmaxf(a,1e-12f)/127.f,1.f/(127.f*512.f));
    inv=1.f/scale;scales[row]=row<m?scale:1.f;
  }
  __syncthreads();
  auto* words=reinterpret_cast<uint32_t*>(q+int64_t(row)*pk);
  #pragma unroll
  for(int i=0;i<PerThread;++i) {
    uint32_t b[L];
    #pragma unroll
    for(int j=0;j<L;++j)b[j]=uint32_t(__float2int_rn(fminf(127.f,fmaxf(-128.f,values[i][j]*inv))))&255;
    uint32_t w0=b[0]|(b[1]<<8)|(b[2]<<16)|(b[3]<<24);
    int g=t+128*i;
    if constexpr(L==4) {
      if(g<pk/4)words[g]=w0;
    } else {
      uint32_t w1=b[2]|(b[3]<<8)|(b[4]<<16)|(b[5]<<24);
      if constexpr(L==6) {
        if(g*2<pk/4)*reinterpret_cast<uint64_t*>(words+g*2)=uint64_t(w0)|(uint64_t(w1)<<32);
      } else {
        uint32_t w2=b[4]|(b[5]<<8)|(b[6]<<16)|(b[7]<<24);
        // Three strided stores would triple sector traffic. Reorder the96
        // words owned by this warp into3 consecutive32-word transactions.
        #pragma unroll
        for(int s=0;s<3;++s) {
          int word=lane+s*32,source=word/3,select=word%3;
          uint32_t a=__shfl_sync(0xffffffffu,w0,source);
          uint32_t b=__shfl_sync(0xffffffffu,w1,source);
          uint32_t c=__shfl_sync(0xffffffffu,w2,source);
          int dst=(warp*32+i*128)*3+word;
          if(dst<pk/4)words[dst]=select==0?a:select==1?b:c;
        }
      }
    }
  }
}

template<int K>
void launch_quant(at::Tensor const& flat,at::Tensor const& q,at::Tensor const& sa,int group) {
  int m=flat.size(0),pm=q.size(0),pk=q.size(1);auto stream=at::cuda::getCurrentCUDAStream();
  if(group==4)quant_lift<K,4><<<pm,128,0,stream>>>(flat.data_ptr<at::BFloat16>(),flat.stride(0),q.data_ptr<int8_t>(),sa.data_ptr<float>(),m,pk);
  else if(group==6)quant_lift<K,6><<<pm,128,0,stream>>>(flat.data_ptr<at::BFloat16>(),flat.stride(0),q.data_ptr<int8_t>(),sa.data_ptr<float>(),m,pk);
  else quant_lift<K,8><<<pm,128,0,stream>>>(flat.data_ptr<at::BFloat16>(),flat.stride(0),q.data_ptr<int8_t>(),sa.data_ptr<float>(),m,pk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor,at::Tensor> quantize(at::Tensor const& input,int64_t group) {
  TORCH_CHECK(input.is_cuda()&&input.scalar_type()==at::kBFloat16&&input.dim()>=1,"BF16 CUDA input required");
  c10::cuda::CUDAGuard guard(input.device());
  int k=input.size(-1);TORCH_CHECK(k==1152||k==1536||k==2432||k==3072||k==4096||k==4608||k==6144||k==9728||k==12288||k==16384,"unsupported native K; use Triton fallback otherwise");
  TORCH_CHECK(group==4||group==6||group==8,"invalid retained sparsity group");
  auto flat=input.reshape({-1,k});if(flat.stride(1)!=1)flat=flat.contiguous();
  int m=flat.size(0),pm=((m+31)/32)*32;
  TORCH_CHECK(m>0,"empty input unsupported by native frontend");
  int lifted=((k+group-1)/group)*(group/2-1)*4,pk=((lifted+127)/128)*128;
  auto q=at::empty({pm,pk},flat.options().dtype(at::kChar));auto sa=at::empty({pm},flat.options().dtype(at::kFloat));
  if(k==1152)launch_quant<1152>(flat,q,sa,group);
else if(k==1536)launch_quant<1536>(flat,q,sa,group);
else if(k==2432)launch_quant<2432>(flat,q,sa,group);
else if(k==3072)launch_quant<3072>(flat,q,sa,group);
else if(k==4096)launch_quant<4096>(flat,q,sa,group);
else if(k==4608)launch_quant<4608>(flat,q,sa,group);
else if(k==6144)launch_quant<6144>(flat,q,sa,group);
else if(k==9728)launch_quant<9728>(flat,q,sa,group);
else if(k==12288)launch_quant<12288>(flat,q,sa,group);
else if(k==16384)launch_quant<16384>(flat,q,sa,group);
  return {q,sa};
}

std::tuple<at::Tensor,at::Tensor> quantize_scalar(at::Tensor const& input) {
  TORCH_CHECK(input.is_cuda()&&input.scalar_type()==at::kBFloat16,"BF16 CUDA input required");
  c10::cuda::CUDAGuard guard(input.device());int k=input.size(-1);
  TORCH_CHECK(k==1152||k==1536||k==2432||k==3072||k==4096||k==4608||k==6144||k==9728||k==12288||k==16384,"unsupported scalar oracle width");
  auto flat=input.reshape({-1,k});if(flat.stride(1)!=1)flat=flat.contiguous();
  int m=flat.size(0),pm=((m+31)/32)*32;
  auto q=at::empty({pm,k},flat.options().dtype(at::kChar));auto sa=at::empty({pm},flat.options().dtype(at::kFloat));auto stream=at::cuda::getCurrentCUDAStream();
  if(k==1152)quant_lift<1152,4,false><<<pm,128,0,stream>>>(flat.data_ptr<at::BFloat16>(),flat.stride(0),q.data_ptr<int8_t>(),sa.data_ptr<float>(),m,k);
  else if(k==1536)quant_lift<1536,4,false><<<pm,128,0,stream>>>(flat.data_ptr<at::BFloat16>(),flat.stride(0),q.data_ptr<int8_t>(),sa.data_ptr<float>(),m,k);
  else if(k==2432)quant_lift<2432,4,false><<<pm,128,0,stream>>>(flat.data_ptr<at::BFloat16>(),flat.stride(0),q.data_ptr<int8_t>(),sa.data_ptr<float>(),m,k);
  else if(k==3072)quant_lift<3072,4,false><<<pm,128,0,stream>>>(flat.data_ptr<at::BFloat16>(),flat.stride(0),q.data_ptr<int8_t>(),sa.data_ptr<float>(),m,k);
  else if(k==4096)quant_lift<4096,4,false><<<pm,128,0,stream>>>(flat.data_ptr<at::BFloat16>(),flat.stride(0),q.data_ptr<int8_t>(),sa.data_ptr<float>(),m,k);
  else if(k==4608)quant_lift<4608,4,false><<<pm,128,0,stream>>>(flat.data_ptr<at::BFloat16>(),flat.stride(0),q.data_ptr<int8_t>(),sa.data_ptr<float>(),m,k);
  else if(k==6144)quant_lift<6144,4,false><<<pm,128,0,stream>>>(flat.data_ptr<at::BFloat16>(),flat.stride(0),q.data_ptr<int8_t>(),sa.data_ptr<float>(),m,k);
  else if(k==9728)quant_lift<9728,4,false><<<pm,128,0,stream>>>(flat.data_ptr<at::BFloat16>(),flat.stride(0),q.data_ptr<int8_t>(),sa.data_ptr<float>(),m,k);
  else if(k==12288)quant_lift<12288,4,false><<<pm,128,0,stream>>>(flat.data_ptr<at::BFloat16>(),flat.stride(0),q.data_ptr<int8_t>(),sa.data_ptr<float>(),m,k);
  else if(k==16384)quant_lift<16384,4,false><<<pm,128,0,stream>>>(flat.data_ptr<at::BFloat16>(),flat.stride(0),q.data_ptr<int8_t>(),sa.data_ptr<float>(),m,k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {q,sa};
}

at::Tensor projection(at::Tensor const& input,at::Tensor const& packed,at::Tensor const& metadata,
    at::Tensor const& scales,at::Tensor const& bias,int64_t group,int64_t n,int64_t config,c10::optional<at::Tensor> const& addend) {
  auto result=quantize(input,group);auto q=std::get<0>(result);auto sa=std::get<1>(result);
  int64_t m=input.numel()/input.size(-1);
  auto y=sdacc_dualscale_sm80::run(q,packed,metadata,sa,scales,bias,m,n,config,addend);
  auto shape=input.sizes().vec();shape.back()=n;return y.reshape(shape);
}
// This include is inside the frontend namespace and follows its base helpers.
#include "scalar_frontend.cuh"
#include "packed16_frontend.cuh"
#include "bridge_frontend.cuh"
}
TORCH_LIBRARY(sdacc_mild_projection_sm80,m) {
  m.def("quantize_packed16(Tensor input,int group)->(Tensor,Tensor)");
  m.def("projection_packed16(Tensor input,Tensor packed,Tensor metadata,Tensor scales,Tensor bias,int group,int n,int config,Tensor? output_addend=None)->Tensor");
  m.def("quantize(Tensor input,int group)->(Tensor,Tensor)");
  m.def("quantize_scalar(Tensor input)->(Tensor,Tensor)");
  m.def("projection(Tensor input,Tensor packed,Tensor metadata,Tensor scales,Tensor bias,int group,int n,int config,Tensor? output_addend=None)->Tensor");
  m.def("projection_scalar(Tensor input,Tensor packed,Tensor metadata,Tensor scales,Tensor bias,int group,int n,int config,Tensor? output_addend=None)->Tensor");
}
TORCH_LIBRARY_IMPL(sdacc_mild_projection_sm80,CUDA,m) {
  m.impl("quantize_packed16",sdacc_mild_projection_sm80::quantize_packed16);
  m.impl("projection_packed16",sdacc_mild_projection_sm80::projection_packed16);
  m.impl("quantize",sdacc_mild_projection_sm80::quantize);
  m.impl("quantize_scalar",sdacc_mild_projection_sm80::quantize_scalar);
  m.impl("projection",sdacc_mild_projection_sm80::projection);
  m.impl("projection_scalar",sdacc_mild_projection_sm80::projection_scalar);
}

#include "graph_projection.cuh"
