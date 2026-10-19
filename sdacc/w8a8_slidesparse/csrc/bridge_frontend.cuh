// Single-patch graph frontends; original frontend bodies remain unchanged.
template<int K,int L,bool Vectorized=true>
__global__ void quant_lift_bridge_native(const at::BFloat16* x,int64_t sx,int8_t* q,float* scales,int m,int pk,at::BFloat16* output,int64_t* destination) {
  if(blockIdx.x==0&&threadIdx.x==0)destination[0]=reinterpret_cast<int64_t>(output);

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
template<int K,int L>
__global__ void quant_lift_bridge_packed16(const at::BFloat16* x,int64_t sx,int8_t* q,float* scales,int m,int pk,at::BFloat16* output,int64_t* destination) {
  if(blockIdx.x==0&&threadIdx.x==0)destination[0]=reinterpret_cast<int64_t>(output);

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
