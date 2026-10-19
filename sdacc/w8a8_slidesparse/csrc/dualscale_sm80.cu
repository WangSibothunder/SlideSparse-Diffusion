// SD-acc SM80 sparse W8A8: register-only dual-scale epilogue.
// The Tensor Core mainloop/metadata reordering use pinned NVIDIA CUTLASS.
// See reference/cutlass_pin.json and its retained BSD license.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_bf16.h>
#include <cutlass/cutlass.h>
#include <cutlass/kernel_launch.h>
#include <cutlass/gemm/threadblock/default_sparse_mma.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/gemm/kernel/sparse_gemm.h>
#include <cutlass/util/host_reorder.h>
#include <cstdint>
#include <vector>
#include <tuple>

namespace sdacc_dualscale_sm80 {
using BF = cutlass::bfloat16_t;
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;

// CUTLASS's generic kernel constructs these iterator objects; actual stores
// below use the physical [token,channel] address rather than its [channel,token]
// logical row-major address. No exchanged/sparsified activation operand.
struct Destination {
  using Element = BF;
  using Layout = Row;
  using TensorRef = cutlass::TensorRef<BF, Row>;
  static constexpr int kElementsPerAccess = 1; // tail path supports unaligned N
  struct Params {
    CUTLASS_HOST_DEVICE Params() {}
    CUTLASS_HOST_DEVICE explicit Params(Row const&) {}
  };
  BF* pointer;
  cutlass::MatrixCoord tile;
  CUTLASS_DEVICE Destination(Params const&, BF* p, cutlass::MatrixCoord const&,
                             int, cutlass::MatrixCoord offset):pointer(p),tile(offset) {}
};
struct OutputOp {
  struct Params {
    float const* sa=nullptr;
    float const* sw=nullptr;
    float const* bias=nullptr;
    at::BFloat16 const* addend=nullptr;
    int64_t addend_stride=0;
    int rows=0; // actual tokens, excluding quant padding
    int columns=0; // actual channels, excluding packed-weight padding
  };
  Params p;
  CUTLASS_DEVICE explicit OutputOp(Params const& params):p(params) {}
  CUTLASS_DEVICE bool is_source_needed() const {return false;}
  CUTLASS_DEVICE void set_k_partition(int,int) {}
};
CUTLASS_DEVICE uint32_t bf_pair(float low,float high) {
  uint32_t out;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(low),"f"(high));
  return out;
}
CUTLASS_DEVICE uint32_t bf_multiply(uint32_t a,uint32_t b) {
  uint32_t out;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(out) : "r"(a),"r"(b),"r"(uint32_t(0)));
  return out;
}
CUTLASS_DEVICE uint32_t bf_fma(uint32_t a,uint32_t b,uint32_t c) {
  uint32_t out;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(out) : "r"(a),"r"(b),"r"(c));
  return out;
}
template<bool Fast>
CUTLASS_DEVICE float scaled(int32_t value,float sa,float sw,float bias) {
  // Match the strict I32-storage oracle: float(acc) * (SA*SW) + bias.
  // Explicit rounded operations avoid a hidden FMA or early16-bit rounding.
  if constexpr(Fast) return __fmaf_rn(__int2float_rn(value),__fmul_rn(sa,sw),bias);
  return __fadd_rn(__fmul_rn(__int2float_rn(value),__fmul_rn(sa,sw)),bias);
}

template<class Shape,class WarpShape,class Fragment,bool Fast,bool Full,bool Butterfly=false,bool BFMath=false,bool Streaming=false,bool HasAddend=false>
struct DirectEpilogue {
  using OutputOp = sdacc_dualscale_sm80::OutputOp;
  using OutputTileIterator = Destination;
  struct SharedStorage {}; // no additional epilogue shared tile or barriers
  int warp,lane;
  CUTLASS_DEVICE DirectEpilogue(SharedStorage&,int,int w,int l):warp(w),lane(l) {}
  CUTLASS_DEVICE void operator()(OutputOp const& op,Destination const& dest,
      Fragment const& accum,Destination const&) {
    constexpr int WM=WarpShape::kM/16, WN=WarpShape::kN/8;
    static_assert(WarpShape::kK==Shape::kK,"split-K is unsupported");
    static_assert(Fragment::kElements==WM*WN*4,"unexpected accumulator format");
    int warp_m=warp%(Shape::kM/WarpShape::kM);
    int warp_n=warp/(Shape::kM/WarpShape::kM);
    int origin_ch=dest.tile.row()+warp_m*WarpShape::kM;
    int origin_token=dest.tile.column()+warp_n*WarpShape::kN;
    CUTLASS_PRAGMA_UNROLL
    for(int im=0;im<WM;++im) {
      int channel0=origin_ch+im*16+(lane>>2);
      int channel8=channel0+8;
      float sw0=(Full||channel0<op.p.columns)?__ldg(op.p.sw+channel0):1.f;
      float sw8=(Full||channel8<op.p.columns)?__ldg(op.p.sw+channel8):1.f;
      float b0=(op.p.bias&&(Full||channel0<op.p.columns))?__ldg(op.p.bias+channel0):0.f;
      float b8=(op.p.bias&&(Full||channel8<op.p.columns))?__ldg(op.p.bias+channel8):0.f;
      uint32_t sw_pair=0,bias_pair=0;
      if constexpr(BFMath) {sw_pair=bf_pair(sw0,sw8);bias_pair=bf_pair(b0,b8);}
      CUTLASS_PRAGMA_UNROLL
      for(int in=0;in<WN;++in) {
        // DefaultSparseMma(..., AccumulatorsInRowMajor=false) fixes this order.
        constexpr int components=4;
        int index=(im+in*WM)*components;
        int token0=origin_token+in*8+(lane&3)*2;
        float sa0=(Full||token0<op.p.rows)?__ldg(op.p.sa+token0):1.f;
        float sa1=(Full||token0+1<op.p.rows)?__ldg(op.p.sa+token0+1):1.f;
        uint32_t even,odd;
        if constexpr(BFMath) {
          // Experimental relaxed epilogue: accumulate still uses native I32.
          // Round only after GEMM; BF16 range avoids FP16 integer overflow.
          uint32_t acc0=bf_pair(__int2float_rn(accum[index]),__int2float_rn(accum[index+2]));
          uint32_t acc1=bf_pair(__int2float_rn(accum[index+1]),__int2float_rn(accum[index+3]));
          even=bf_fma(bf_multiply(acc0,sw_pair),bf_pair(sa0,sa0),bias_pair);
          odd=bf_fma(bf_multiply(acc1,sw_pair),bf_pair(sa1,sa1),bias_pair);
        } else {
          float v00=scaled<Fast>(accum[index],sa0,sw0,b0);
          float v08=scaled<Fast>(accum[index+2],sa0,sw8,b8);
          float v10=scaled<Fast>(accum[index+1],sa1,sw0,b0);
          float v18=scaled<Fast>(accum[index+3],sa1,sw8,b8);
          if constexpr(HasAddend) {
            const auto* add=reinterpret_cast<const __nv_bfloat16*>(op.p.addend);
            if(Full||(token0<op.p.rows&&channel0<op.p.columns))v00+=__bfloat162float(add[int64_t(token0)*op.p.addend_stride+channel0]);
            if(Full||(token0<op.p.rows&&channel8<op.p.columns))v08+=__bfloat162float(add[int64_t(token0)*op.p.addend_stride+channel8]);
            if(Full||(token0+1<op.p.rows&&channel0<op.p.columns))v10+=__bfloat162float(add[int64_t(token0+1)*op.p.addend_stride+channel0]);
            if(Full||(token0+1<op.p.rows&&channel8<op.p.columns))v18+=__bfloat162float(add[int64_t(token0+1)*op.p.addend_stride+channel8]);
          }
          even=bf_pair(v00,v08);odd=bf_pair(v10,v18);
        }
        int output_token=origin_token+in*8+(lane>>2);
        int channel_base=origin_ch+im*16+(lane&3)*4;
        uint16_t values[4];
        if constexpr(Butterfly) {
          // Swap fragment register bit(token parity) with lane bit3(channel bit1).
          uint32_t e8=__shfl_xor_sync(0xffffffffu,even,8);
          uint32_t o8=__shfl_xor_sync(0xffffffffu,odd,8);
          uint32_t r0=(lane&8)?o8:even;
          uint32_t r1=(lane&8)?odd:e8;
          // Swap packed16 bit(channel bit3) with lane bit2(channel bit0).
          uint32_t a4=__shfl_xor_sync(0xffffffffu,r0,4);
          uint32_t b4=__shfl_xor_sync(0xffffffffu,r1,4);
          uint32_t s0=(lane&4)?__byte_perm(a4,r0,0x7632):__byte_perm(r0,a4,0x5410);
          uint32_t s1=(lane&4)?__byte_perm(b4,r1,0x7632):__byte_perm(r1,b4,0x5410);
          // Final lane-only permutation: [ch2,tok0,ch3,tok2,tok1]
          // becomes [tok2,tok1,tok0,ch3,ch2]. Two contiguous channel pairs.
          int source=((lane&1)<<4)|((lane&4)<<1)|((lane&2)<<1)|((lane&16)>>3)|((lane&8)>>3);
          uint32_t lo=__shfl_sync(0xffffffffu,s0,source);
          uint32_t hi=__shfl_sync(0xffffffffu,s1,source);
          values[0]=uint16_t(lo);values[1]=uint16_t(lo>>16);
          values[2]=uint16_t(hi);values[3]=uint16_t(hi>>16);
        } else {
        CUTLASS_PRAGMA_UNROLL
        for(int j=0;j<4;++j) {
          int local_ch=(lane&3)*4+j;
          int source=(local_ch&7)*4+((lane>>2)>>1);
          // ALL32 lanes execute both shuffles; only the final store is masked.
          uint32_t a=__shfl_sync(0xffffffffu,even,source);
          uint32_t b=__shfl_sync(0xffffffffu,odd,source);
          uint32_t chosen=((lane>>2)&1)?b:a;
          values[j]=uint16_t(chosen>>((local_ch>=8)?16:0));
        }
        }
        if(Full||(output_token<op.p.rows && channel_base<op.p.columns)) {
          auto* out=reinterpret_cast<uint16_t*>(dest.pointer)+
                    int64_t(output_token)*op.p.columns+channel_base;
          if(Full||(channel_base+3<op.p.columns && (reinterpret_cast<uintptr_t>(out)&7)==0)) {
            // Four channels per lane: 8 token rows x32 useful bytes per warp.
            uint2 packed=make_uint2(uint32_t(values[0])|(uint32_t(values[1])<<16),
                                   uint32_t(values[2])|(uint32_t(values[3])<<16));
            if constexpr(Streaming) {
              asm volatile("st.global.cs.v2.u32 [%0], {%1,%2};" :: "l"(out),"r"(packed.x),"r"(packed.y):"memory");
            } else *reinterpret_cast<uint2*>(out)=packed;
          } else {
            CUTLASS_PRAGMA_UNROLL
            for(int j=0;j<4;++j) if(channel_base+j<op.p.columns)out[j]=values[j];
          }
        }
      }
    }
  }
};

template<int Channels,bool Fast,int Tokens=128,int WarpM=64,int WarpN=64,bool Full=false,bool Butterfly=false,int Stages=3,bool BFMath=false,bool Streaming=false,bool HasAddend=false>
struct Configuration {
  static constexpr int kStages=Stages;
  using Shape=cutlass::gemm::GemmShape<Channels,Tokens,128>;
  using WarpShape=cutlass::gemm::GemmShape<WarpM,WarpN,128>;
  using Mma=typename cutlass::gemm::threadblock::DefaultSparseMma<
      int8_t,Row,16,int8_t,Col,16,int32_t,Row,
      cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
      Shape,WarpShape,cutlass::gemm::GemmShape<16,8,64>,Stages,
      cutlass::arch::OpMultiplyAddSaturate,false>::ThreadblockMma;
  using Epilogue=DirectEpilogue<Shape,WarpShape,typename Mma::FragmentC,Fast,Full,Butterfly,BFMath,Streaming,HasAddend>;
  using Swizzle=cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using Kernel=cutlass::gemm::kernel::SparseGemm<Mma,Epilogue,Swizzle,false>;
  static_assert(Kernel::kSparse==2&&Kernel::kElementsPerElementE==16,"metadata assumption");
  static_assert(sizeof(typename Kernel::ElementE)==4,"metadata must be uint32");
  static_assert(Kernel::LayoutE::kInterleave==2,"metadata layout assumption");
};
using C0=Configuration<128,false>;
using C1=Configuration<128,true>;
using C2=Configuration<256,true>;
using C3=Configuration<128,true,64,64,32>;
using C3F=Configuration<128,true,64,64,32,true>;
using C4=Configuration<128,true,128,32,64>;
using C4F=Configuration<128,true,128,32,64,true>;
using C5=Configuration<128,true,128,64,64,false,true>;
using C5F=Configuration<128,true,128,64,64,true,true>;
using C6=Configuration<256,true,128,64,64,false,true>;
using C6F=Configuration<256,true,128,64,64,true,true>;
using C11=Configuration<128,true,128,64,64,false,true,3,false,true>;
using C11F=Configuration<128,true,128,64,64,true,true,3,false,true>;

using C12=Configuration<128,true,64,64,32,false,true>;
using C12F=Configuration<128,true,64,64,32,true,true>;
using C13=Configuration<128,true,128,64,64,false,true,2>;
using C13F=Configuration<128,true,128,64,64,true,true,2>;
using A5=Configuration<128,true,128,64,64,false,true,3,false,false,true>;
using A5F=Configuration<128,true,128,64,64,true,true,3,false,false,true>;
using A6=Configuration<256,true,128,64,64,false,true,3,false,false,true>;
using A6F=Configuration<256,true,128,64,64,true,true,3,false,false,true>;
using A11=Configuration<128,true,128,64,64,false,true,3,false,true,true>;
using A11F=Configuration<128,true,128,64,64,true,true,3,false,true,true>;
using A12=Configuration<128,true,64,64,32,false,true,3,false,false,true>;
using A12F=Configuration<128,true,64,64,32,true,true,3,false,false,true>;
using A13=Configuration<128,true,128,64,64,false,true,2,false,false,true>;
using A13F=Configuration<128,true,128,64,64,true,true,2,false,false,true>;

using C14=Configuration<64,true,64,32,32,false,true,2>;
using C14F=Configuration<64,true,64,32,32,true,true,2>;
using A14=Configuration<64,true,64,32,32,false,true,2,false,false,true>;
using A14F=Configuration<64,true,64,32,32,true,true,2,false,false,true>;

using C15=Configuration<64,true,64,32,32,false,true,3>;
using C15F=Configuration<64,true,64,32,32,true,true,3>;
using A15=Configuration<64,true,64,32,32,false,true,3,false,false,true>;
using A15F=Configuration<64,true,64,32,32,true,true,3,false,false,true>;

using C16=Configuration<128,true,64,64,32,false,true,2,false,false,false>;
using C16F=Configuration<128,true,64,64,32,true,true,2,false,false,false>;
using A16=Configuration<128,true,64,64,32,false,true,2,false,false,true>;
using A16F=Configuration<128,true,64,64,32,true,true,2,false,false,true>;
using C17=Configuration<64,true,128,32,64,false,true,3,false,false,false>;
using C17F=Configuration<64,true,128,32,64,true,true,3,false,false,false>;
using A17=Configuration<64,true,128,32,64,false,true,3,false,false,true>;
using A17F=Configuration<64,true,128,32,64,true,true,3,false,false,true>;

std::tuple<at::Tensor,at::Tensor> pack_cpu(at::Tensor const& weight,int64_t padded_k) {
  TORCH_CHECK(weight.device().is_cpu()&&weight.scalar_type()==at::kChar&&weight.dim()==2&&weight.is_contiguous(),"pack needs contiguous CPU int8 weight");
  int64_t n=weight.size(0),k=weight.size(1),pn=((n+31)/32)*32,pk=padded_k;
  TORCH_CHECK(n>0&&k>0&&pk>=k&&pk%128==0,"invalid128-K padding");
  auto packed=at::empty({pn,pk/2},weight.options());
  auto raw=at::zeros({pn,pk/32},weight.options().dtype(at::kInt));
  auto metadata=at::empty({pn*pk/32},weight.options().dtype(at::kInt));
  auto* w=weight.data_ptr<int8_t>();auto* a=packed.data_ptr<int8_t>();
  auto* e=reinterpret_cast<uint32_t*>(raw.data_ptr<int32_t>());
  for(int64_t row=0;row<pn;++row)for(int64_t g=0;g<pk/4;++g) {
    int8_t vals[4]={};int selected[2]={};int count=0;
    for(int j=0;j<4;++j) {
      if(row<n&&g*4+j<k)vals[j]=w[row*k+g*4+j];
      if(vals[j]) {TORCH_CHECK(count<2,"more than2 nonzeros in a frozen2:4 group");selected[count++]=j;}
    }
    for(int j=0;count<2&&j<4;++j)if(!vals[j])selected[count++]=j;
    if(selected[0]>selected[1])std::swap(selected[0],selected[1]);
    a[row*(pk/2)+2*g]=vals[selected[0]];a[row*(pk/2)+2*g+1]=vals[selected[1]];
    uint32_t nibble=uint32_t(selected[0])|(uint32_t(selected[1])<<2);
    e[row*(pk/32)+g/8]|=nibble<<(4*(g%8));
  }
  using LE=typename C0::Kernel::LayoutE;
  auto src=cutlass::TensorRef<uint32_t,Row>(e,Row(pk/32));
  auto dst=cutlass::TensorRef<uint32_t,LE>(reinterpret_cast<uint32_t*>(metadata.data_ptr<int32_t>()),LE::packed({int(pn),int(pk/32)}));
  cutlass::reorder_meta(dst,src,{int(pn),1,int(pk/32)});
  return {packed,metadata};
}

template<class Config>
std::vector<int64_t> configure_t() {
  using K=typename Config::Kernel;
  constexpr int bytes=sizeof(typename K::SharedStorage);
  if constexpr(Config::kStages==2)C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<K>,cudaFuncAttributePreferredSharedMemoryCarveout,100));
  if(bytes>48*1024)C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<K>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,cutlass::Kernel<K>));
  int blocks=0;C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,cutlass::Kernel<K>,K::kThreadCount,bytes));
  return {K::kThreadCount,bytes,a.numRegs,int64_t(a.localSizeBytes),blocks};
}
std::vector<int64_t> configure(at::Tensor const& anchor,int64_t config,bool has_addend) {
  TORCH_CHECK(anchor.is_cuda(),"CUDA anchor required");c10::cuda::CUDAGuard guard(anchor.device());
  cudaDeviceProp p;C10_CUDA_CHECK(cudaGetDeviceProperties(&p,anchor.get_device()));TORCH_CHECK(p.major==8&&p.minor==0,"this prototype is qualified only onSM80");
  TORCH_CHECK(config>=0&&(config<=6||config==11||config==12||config==13||config==14||config==15||config==16||config==17),"invalid config");
  if(has_addend&&config==5){auto b=configure_t<A5>();auto f=configure_t<A5F>();b.insert(b.end(),f.begin(),f.end());return b;}
  if(has_addend&&config==6){auto b=configure_t<A6>();auto f=configure_t<A6F>();b.insert(b.end(),f.begin(),f.end());return b;}
  if(has_addend&&config==11){auto b=configure_t<A11>();auto f=configure_t<A11F>();b.insert(b.end(),f.begin(),f.end());return b;}
  if(has_addend&&config==12){auto b=configure_t<A12>();auto f=configure_t<A12F>();b.insert(b.end(),f.begin(),f.end());return b;}
  if(has_addend&&config==13){auto b=configure_t<A13>();auto f=configure_t<A13F>();b.insert(b.end(),f.begin(),f.end());return b;}
  if(has_addend&&config==14){auto b=configure_t<A14>();auto f=configure_t<A14F>();b.insert(b.end(),f.begin(),f.end());return b;}
  if(has_addend&&config==15){auto b=configure_t<A15>();auto f=configure_t<A15F>();b.insert(b.end(),f.begin(),f.end());return b;}
  if(has_addend&&config==16){auto b=configure_t<A16>();auto f=configure_t<A16F>();b.insert(b.end(),f.begin(),f.end());return b;}
  if(has_addend&&config==17){auto b=configure_t<A17>();auto f=configure_t<A17F>();b.insert(b.end(),f.begin(),f.end());return b;}
  TORCH_CHECK(!has_addend,"addend supports configs5/6/11/12/13/14/15");
  if(config==12){auto b=configure_t<C12>();auto f=configure_t<C12F>();b.insert(b.end(),f.begin(),f.end());return b;}
  if(config==13){auto b=configure_t<C13>();auto f=configure_t<C13F>();b.insert(b.end(),f.begin(),f.end());return b;}
  if(config==14){auto b=configure_t<C14>();auto f=configure_t<C14F>();b.insert(b.end(),f.begin(),f.end());return b;}
  if(config==15){auto b=configure_t<C15>();auto f=configure_t<C15F>();b.insert(b.end(),f.begin(),f.end());return b;}
  if(config==16){auto b=configure_t<C16>();auto f=configure_t<C16F>();b.insert(b.end(),f.begin(),f.end());return b;}
  if(config==17){auto b=configure_t<C17>();auto f=configure_t<C17F>();b.insert(b.end(),f.begin(),f.end());return b;}
  if(config==0)return configure_t<C0>();
  if(config==1)return configure_t<C1>();
  if(config==2)return configure_t<C2>();
  auto base=config==3?configure_t<C3>():config==4?configure_t<C4>():config==5?configure_t<C5>():config==6?configure_t<C6>():configure_t<C11>();
  auto full=config==3?configure_t<C3F>():config==4?configure_t<C4F>():config==5?configure_t<C5F>():config==6?configure_t<C6F>():configure_t<C11F>();
  base.insert(base.end(),full.begin(),full.end());return base;
}

template<class Config>
void launch(at::Tensor const& q,at::Tensor const& packed,at::Tensor const& meta,
    at::Tensor const& sa,at::Tensor const& sw,at::Tensor const& bias,
    at::Tensor const& output,int m,int n,c10::optional<at::Tensor> const& addend) {
  using K=typename Config::Kernel;using LE=typename K::LayoutE;
  int pn=packed.size(0),pk=q.size(1),pm=q.size(0);
  cutlass::gemm::GemmCoord problem(pn,pm,pk);
  typename Config::Swizzle swizzle;auto tiled=swizzle.get_tiled_shape(problem,{Config::Shape::kM,Config::Shape::kN,Config::Shape::kK},1);
  typename K::TensorRefA refa(packed.data_ptr<int8_t>(),Row(pk/2));
  typename K::TensorRefB refb(q.data_ptr<int8_t>(),Col(q.stride(0)));
  typename K::TensorRefE refe(reinterpret_cast<typename K::ElementE*>(meta.data_ptr<int32_t>()),LE::packed({pn,pk/32}));
  // Logical row-major iterator stride is a descriptor placeholder. Direct
  // epilogue stores use actual physical output stride n from OutputOp::Params.
  Destination::TensorRef refc(nullptr,Row(pm));
  Destination::TensorRef refd(reinterpret_cast<BF*>(output.data_ptr<at::BFloat16>()),Row(pm));
  TORCH_CHECK(K::can_implement(problem,refa,refb,refc,refd,refe)==cutlass::Status::kSuccess,"unsupported alignment/problem");
  OutputOp::Params op;op.sa=sa.data_ptr<float>();op.sw=sw.data_ptr<float>();op.bias=bias.numel()?bias.data_ptr<float>():nullptr;op.rows=m;op.columns=n;
  if(addend.has_value()){op.addend=addend->data_ptr<at::BFloat16>();op.addend_stride=addend->stride(0);}
  typename K::Params params(problem,tiled,refa,refb,refc,refd,refe,op,nullptr);
  auto grid=swizzle.get_grid_shape(tiled);auto stream=at::cuda::getCurrentCUDAStream();
  cutlass::Kernel<K><<<grid,K::kThreadCount,sizeof(typename K::SharedStorage),stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
at::Tensor run(at::Tensor const& q,at::Tensor const& packed,at::Tensor const& meta,
    at::Tensor const& sa,at::Tensor const& sw,at::Tensor const& bias,
    int64_t m,int64_t n,int64_t config,c10::optional<at::Tensor> const& addend) {
  TORCH_CHECK(q.is_cuda()&&q.scalar_type()==at::kChar&&q.dim()==2&&q.is_contiguous(),"q must be row-major CUDA int8");
  c10::cuda::CUDAGuard guard(q.device());
  for(auto const& t:{packed,meta,sa,sw,bias})TORCH_CHECK(t.device()==q.device()&&t.is_contiguous(),"buffer device/layout mismatch");
  TORCH_CHECK(packed.scalar_type()==at::kChar&&packed.dim()==2&&meta.scalar_type()==at::kInt,"packed types");
  TORCH_CHECK(sa.scalar_type()==at::kFloat&&sw.scalar_type()==at::kFloat&&bias.scalar_type()==at::kFloat,"scale/bias must be FP32");
  TORCH_CHECK(q.size(1)%128==0&&q.size(0)%32==0&&packed.size(0)%32==0&&packed.size(1)*2==q.size(1),"padding mismatch");
  TORCH_CHECK(q.size(1)<=131008,"INT32 worst-case bound exceeded");
  TORCH_CHECK(meta.numel()==packed.size(0)*q.size(1)/32&&sa.numel()==q.size(0),"metadata/token scales size");
  TORCH_CHECK(m>0&&m<=q.size(0)&&n>0&&n<=packed.size(0)&&sw.numel()>=n&&(bias.numel()==0||bias.numel()>=n),"original dimensions/channels");
  TORCH_CHECK(config>=0&&(config<=6||config==11||config==12||config==13||config==14||config==15||config==16||config==17),"invalid config");
  auto out=at::empty({m,n},q.options().dtype(at::kBFloat16));
  if(addend.has_value()) {
    auto const& t=*addend;
    TORCH_CHECK(t.is_cuda()&&t.device()==q.device()&&t.scalar_type()==at::kBFloat16&&t.dim()==2&&t.size(0)==m&&t.size(1)==n&&t.stride(1)==1,"addend must be BF16 [M,N] with unit column stride");
    if(config==5){if(n%128==0&&m%128==0)launch<A5F>(q,packed,meta,sa,sw,bias,out,m,n,addend);else launch<A5>(q,packed,meta,sa,sw,bias,out,m,n,addend);}
    else if(config==6){if(n%256==0&&m%128==0)launch<A6F>(q,packed,meta,sa,sw,bias,out,m,n,addend);else launch<A6>(q,packed,meta,sa,sw,bias,out,m,n,addend);}
    else if(config==11){if(n%128==0&&m%128==0)launch<A11F>(q,packed,meta,sa,sw,bias,out,m,n,addend);else launch<A11>(q,packed,meta,sa,sw,bias,out,m,n,addend);}
    else if(config==12){if(n%128==0&&m%64==0)launch<A12F>(q,packed,meta,sa,sw,bias,out,m,n,addend);else launch<A12>(q,packed,meta,sa,sw,bias,out,m,n,addend);}
    else if(config==13){if(n%128==0&&m%128==0)launch<A13F>(q,packed,meta,sa,sw,bias,out,m,n,addend);else launch<A13>(q,packed,meta,sa,sw,bias,out,m,n,addend);}
    else if(config==14){if(n%64==0&&m%64==0)launch<A14F>(q,packed,meta,sa,sw,bias,out,m,n,addend);else launch<A14>(q,packed,meta,sa,sw,bias,out,m,n,addend);}
    else if(config==15){if(n%64==0&&m%64==0)launch<A15F>(q,packed,meta,sa,sw,bias,out,m,n,addend);else launch<A15>(q,packed,meta,sa,sw,bias,out,m,n,addend);}
    else if(config==16){if(n%128==0&&m%64==0)launch<A16F>(q,packed,meta,sa,sw,bias,out,m,n,addend);else launch<A16>(q,packed,meta,sa,sw,bias,out,m,n,addend);}
    else if(config==17){if(n%64==0&&m%128==0)launch<A17F>(q,packed,meta,sa,sw,bias,out,m,n,addend);else launch<A17>(q,packed,meta,sa,sw,bias,out,m,n,addend);}
    else TORCH_CHECK(false,"unsupported addend config");
    return out;
  }
  if(config==12){if(n%128==0&&m%64==0)launch<C12F>(q,packed,meta,sa,sw,bias,out,m,n,addend);else launch<C12>(q,packed,meta,sa,sw,bias,out,m,n,addend);return out;}
  if(config==13){if(n%128==0&&m%128==0)launch<C13F>(q,packed,meta,sa,sw,bias,out,m,n,addend);else launch<C13>(q,packed,meta,sa,sw,bias,out,m,n,addend);return out;}
  if(config==14){if(n%64==0&&m%64==0)launch<C14F>(q,packed,meta,sa,sw,bias,out,m,n,addend);else launch<C14>(q,packed,meta,sa,sw,bias,out,m,n,addend);return out;}
  if(config==15){if(n%64==0&&m%64==0)launch<C15F>(q,packed,meta,sa,sw,bias,out,m,n,addend);else launch<C15>(q,packed,meta,sa,sw,bias,out,m,n,addend);return out;}
  if(config==16){if(n%128==0&&m%64==0)launch<C16F>(q,packed,meta,sa,sw,bias,out,m,n,addend);else launch<C16>(q,packed,meta,sa,sw,bias,out,m,n,addend);return out;}
  if(config==17){if(n%64==0&&m%128==0)launch<C17F>(q,packed,meta,sa,sw,bias,out,m,n,addend);else launch<C17>(q,packed,meta,sa,sw,bias,out,m,n,addend);return out;}
  if(config==0)launch<C0>(q,packed,meta,sa,sw,bias,out,m,n,addend);
  else if(config==1)launch<C1>(q,packed,meta,sa,sw,bias,out,m,n,addend);
  else if(config==2)launch<C2>(q,packed,meta,sa,sw,bias,out,m,n,addend);
  else if(config==3) {
    if(n%128==0&&m%64==0)launch<C3F>(q,packed,meta,sa,sw,bias,out,m,n,addend);
    else launch<C3>(q,packed,meta,sa,sw,bias,out,m,n,addend);
  } else if(config==4) {
    if(n%128==0&&m%128==0)launch<C4F>(q,packed,meta,sa,sw,bias,out,m,n,addend);
    else launch<C4>(q,packed,meta,sa,sw,bias,out,m,n,addend);
  } else if(config==5) {
    if(n%128==0&&m%128==0)launch<C5F>(q,packed,meta,sa,sw,bias,out,m,n,addend);
    else launch<C5>(q,packed,meta,sa,sw,bias,out,m,n,addend);
  } else if(config==6) {
    if(n%256==0&&m%128==0)launch<C6F>(q,packed,meta,sa,sw,bias,out,m,n,addend);
    else launch<C6>(q,packed,meta,sa,sw,bias,out,m,n,addend);
  } else {
    if(n%128==0&&m%128==0)launch<C11F>(q,packed,meta,sa,sw,bias,out,m,n,addend);
    else launch<C11>(q,packed,meta,sa,sw,bias,out,m,n,addend);
  }
  return out;
}
} // namespace sdacc_dualscale_sm80
TORCH_LIBRARY(sdacc_dualscale_sm80,m) {
  m.def("pack_cpu(Tensor weight,int padded_k)->(Tensor,Tensor)");
  m.def("configure(Tensor anchor,int config,bool has_addend=False)->int[]");
  m.def("run(Tensor q,Tensor packed,Tensor metadata,Tensor sa,Tensor sw,Tensor bias,int m,int n,int config,Tensor? output_addend=None)->Tensor");
}
TORCH_LIBRARY_IMPL(sdacc_dualscale_sm80,CPU,m) {m.impl("pack_cpu",sdacc_dualscale_sm80::pack_cpu);}
TORCH_LIBRARY_IMPL(sdacc_dualscale_sm80,CUDA,m) {m.impl("configure",sdacc_dualscale_sm80::configure);m.impl("run",sdacc_dualscale_sm80::run);}
