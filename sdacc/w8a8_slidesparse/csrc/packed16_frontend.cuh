// Retain BF16 pairs until the row reduction is complete. Finite BF16 absolute
// values have monotone unsigned bit representations; no approximate maximum.
template<int K,int L>
__global__ void quant_lift_packed16(const at::BFloat16* x,int64_t sx,
    int8_t* q,float* scales,int m,int pk) {
  constexpr int Windows=L/2-1;
  constexpr int Lift=((K+L-1)/L)*Windows*4;
  constexpr int PK=((Lift+127)/128)*128;
  constexpr int Groups=(PK/4+Windows-1)/Windows;
  constexpr int PerThread=(Groups+127)/128;
  int row=blockIdx.x,t=threadIdx.x,lane=t&31,warp=t>>5;
  uint32_t values[PerThread][L/2];uint32_t maximum_pair=0;
  #pragma unroll
  for(int i=0;i<PerThread;++i) {
    int g=t+128*i;
    #pragma unroll
    for(int j=0;j<L/2;++j) {
      int col=g*L+j*2;uint32_t raw=0;
      if(row<m&&col<K) {
        const auto* p=reinterpret_cast<const uint16_t*>(x)+int64_t(row)*sx+col;
        if(col+1<K&&(reinterpret_cast<uintptr_t>(p)&3)==0)
          raw=*reinterpret_cast<const uint32_t*>(p);
        else raw=uint32_t(p[0])|((col+1<K?uint32_t(p[1]):0u)<<16);
      }
      values[i][j]=raw;
      uint32_t magnitude=raw&0x7fff7fffu;
      // Same fmax(NaN,finite) convention as the scalar frontend.
      uint32_t lo=magnitude&65535u,hi=magnitude>>16;
      if(lo>0x7f80u)lo=0;if(hi>0x7f80u)hi=0;
      maximum_pair=__vmaxu2(maximum_pair,lo|(hi<<16));
    }
  }
  uint32_t maximum=max(maximum_pair&65535u,maximum_pair>>16);
  #pragma unroll
  for(int d=16;d;d>>=1)maximum=max(maximum,__shfl_down_sync(0xffffffffu,maximum,d));
  __shared__ uint32_t partial[4];__shared__ float scale,inv;
  if(lane==0)partial[warp]=maximum;
  __syncthreads();
  if(t==0) {
    uint32_t a=max(max(partial[0],partial[1]),max(partial[2],partial[3]));
    scale=fmaxf(fmaxf(__uint_as_float(a<<16),1e-12f)/127.f,1.f/(127.f*512.f));
    inv=1.f/scale;scales[row]=row<m?scale:1.f;
  }
  __syncthreads();
  auto* words=reinterpret_cast<uint32_t*>(q+int64_t(row)*pk);
  #pragma unroll
  for(int i=0;i<PerThread;++i) {
    uint32_t b[L];
    #pragma unroll
    for(int j=0;j<L/2;++j) {
      float lo=__uint_as_float((values[i][j]&65535u)<<16);
      float hi=__uint_as_float(values[i][j]&0xffff0000u);
      b[j*2]=uint32_t(__float2int_rn(fminf(127.f,fmaxf(-128.f,lo*inv))))&255;
      b[j*2+1]=uint32_t(__float2int_rn(fminf(127.f,fmaxf(-128.f,hi*inv))))&255;
    }
    uint32_t w0=b[0]|(b[1]<<8)|(b[2]<<16)|(b[3]<<24);int g=t+128*i;
    if constexpr(L==4) {if(g<pk/4)words[g]=w0;}
    else {
      uint32_t w1=b[2]|(b[3]<<8)|(b[4]<<16)|(b[5]<<24);
      if constexpr(L==6) {
        if(g*2<pk/4)*reinterpret_cast<uint64_t*>(words+g*2)=uint64_t(w0)|(uint64_t(w1)<<32);
      } else {
        uint32_t w2=b[4]|(b[5]<<8)|(b[6]<<16)|(b[7]<<24);
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
void launch_quant_packed16(at::Tensor const& flat,at::Tensor const& q,
    at::Tensor const& sa,int group) {
  int m=flat.size(0),pm=q.size(0),pk=q.size(1);auto stream=at::cuda::getCurrentCUDAStream();
  if(group==4)quant_lift_packed16<K,4><<<pm,128,0,stream>>>(flat.data_ptr<at::BFloat16>(),flat.stride(0),q.data_ptr<int8_t>(),sa.data_ptr<float>(),m,pk);
  else if(group==6)quant_lift_packed16<K,6><<<pm,128,0,stream>>>(flat.data_ptr<at::BFloat16>(),flat.stride(0),q.data_ptr<int8_t>(),sa.data_ptr<float>(),m,pk);
  else quant_lift_packed16<K,8><<<pm,128,0,stream>>>(flat.data_ptr<at::BFloat16>(),flat.stride(0),q.data_ptr<int8_t>(),sa.data_ptr<float>(),m,pk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::tuple<at::Tensor,at::Tensor> quantize_packed16(at::Tensor const& input,int64_t group) {
  TORCH_CHECK(input.is_cuda()&&input.scalar_type()==at::kBFloat16&&input.dim()>=1,"BF16 CUDA input required");
  c10::cuda::CUDAGuard guard(input.device());int k=input.size(-1);
  TORCH_CHECK(k==1152||k==1536||k==2432||k==4608||k==6144||k==9728,"packed16 qualified widths:1152/1536/2432/4608/6144/9728");
  TORCH_CHECK(group==4||group==6||group==8,"invalid sparsity group");
  auto flat=input.reshape({-1,k});if(flat.stride(1)!=1)flat=flat.contiguous();
  int m=flat.size(0),pm=((m+31)/32)*32;
  TORCH_CHECK(m>0,"empty input handled by Python module");
  int lifted=((k+group-1)/group)*(group/2-1)*4,pk=((lifted+127)/128)*128;
  auto q=at::empty({pm,pk},flat.options().dtype(at::kChar));auto sa=at::empty({pm},flat.options().dtype(at::kFloat));
  if(k==1152)launch_quant_packed16<1152>(flat,q,sa,group);else if(k==1536)launch_quant_packed16<1536>(flat,q,sa,group);else if(k==2432)launch_quant_packed16<2432>(flat,q,sa,group);else if(k==4608)launch_quant_packed16<4608>(flat,q,sa,group);else if(k==6144)launch_quant_packed16<6144>(flat,q,sa,group);else launch_quant_packed16<9728>(flat,q,sa,group);
  return {q,sa};
}
at::Tensor projection_packed16(at::Tensor const& input,at::Tensor const& packed,
    at::Tensor const& metadata,at::Tensor const& scales,at::Tensor const& bias,
    int64_t group,int64_t n,int64_t config,c10::optional<at::Tensor> const& addend) {
  auto qs=quantize_packed16(input,group);
  auto y=sdacc_dualscale_sm80::run(std::get<0>(qs),packed,metadata,std::get<1>(qs),
      scales,bias,input.numel()/input.size(-1),n,config,addend);
  auto shape=input.sizes().vec();shape.back()=n;return y.reshape(shape);
}
