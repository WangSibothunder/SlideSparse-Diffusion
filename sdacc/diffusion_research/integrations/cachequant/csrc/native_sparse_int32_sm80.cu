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

// Private integer epilogue. Reuses frozen public C0 MMA and metadata packing.
namespace sigma_orthogonal_sparse_int32_sm80 {
using Row=cutlass::layout::RowMajor;
using Col=cutlass::layout::ColumnMajor;
struct Destination {
 using Element=int32_t; using Layout=Row;
 using TensorRef=cutlass::TensorRef<int32_t,Row>;
 static constexpr int kElementsPerAccess=1;
 struct Params {CUTLASS_HOST_DEVICE Params(){} CUTLASS_HOST_DEVICE explicit Params(Row const&) {}};
 int32_t* pointer;cutlass::MatrixCoord tile;
 CUTLASS_DEVICE Destination(Params const&,int32_t* p,cutlass::MatrixCoord const&,int,cutlass::MatrixCoord o):pointer(p),tile(o){}
};
struct OutputOp {
 struct Params {int rows=0,columns=0;}; Params p;
 CUTLASS_DEVICE explicit OutputOp(Params const& x):p(x){}
 CUTLASS_DEVICE bool is_source_needed()const{return false;}
 CUTLASS_DEVICE void set_k_partition(int,int){}
};
template<class Shape,class WarpShape,class Fragment>
struct IntegerEpilogue {
 using OutputOp=sigma_orthogonal_sparse_int32_sm80::OutputOp;
 using OutputTileIterator=Destination;struct SharedStorage{};
 int warp,lane;
 CUTLASS_DEVICE IntegerEpilogue(SharedStorage&,int,int w,int l):warp(w),lane(l){}
 CUTLASS_DEVICE void operator()(OutputOp const& op,Destination const& dst,Fragment const& acc,Destination const&){
  constexpr int WM=WarpShape::kM/16,WN=WarpShape::kN/8;
  static_assert(Fragment::kElements==WM*WN*4,"accumulator register order");
  int wm=warp%(Shape::kM/WarpShape::kM),wn=warp/(Shape::kM/WarpShape::kM);
  int ch0=dst.tile.row()+wm*WarpShape::kM,t0=dst.tile.column()+wn*WarpShape::kN;
  CUTLASS_PRAGMA_UNROLL
  for(int im=0;im<WM;++im){
   int ch=ch0+im*16+(lane>>2);
   CUTLASS_PRAGMA_UNROLL
   for(int in=0;in<WN;++in){
    int idx=(im+in*WM)*4,t=t0+in*8+(lane&3)*2;
    if(t<op.p.rows&&ch<op.p.columns)dst.pointer[int64_t(t)*op.p.columns+ch]=acc[idx];
    if(t+1<op.p.rows&&ch<op.p.columns)dst.pointer[int64_t(t+1)*op.p.columns+ch]=acc[idx+1];
    if(t<op.p.rows&&ch+8<op.p.columns)dst.pointer[int64_t(t)*op.p.columns+ch+8]=acc[idx+2];
    if(t+1<op.p.rows&&ch+8<op.p.columns)dst.pointer[int64_t(t+1)*op.p.columns+ch+8]=acc[idx+3];
   }
  }
 }
};
struct C0 {
 using Shape=cutlass::gemm::GemmShape<128,128,128>;
 using WarpShape=cutlass::gemm::GemmShape<64,64,128>;
 using Mma=typename cutlass::gemm::threadblock::DefaultSparseMma<int8_t,Row,16,int8_t,Col,16,int32_t,Row,
  cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,Shape,WarpShape,cutlass::gemm::GemmShape<16,8,64>,3,
  cutlass::arch::OpMultiplyAddSaturate,false>::ThreadblockMma;
 using Epilogue=IntegerEpilogue<Shape,WarpShape,typename Mma::FragmentC>;
 using Swizzle=cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
 using Kernel=cutlass::gemm::kernel::SparseGemm<Mma,Epilogue,Swizzle,false>;
 static_assert(Kernel::kSparse==2&&Kernel::kElementsPerElementE==16,"metadata assumptions");
 static_assert(sizeof(typename Kernel::ElementE)==4&&Kernel::LayoutE::kInterleave==2,"metadata layout");
};
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


std::vector<int64_t> configure(at::Tensor const& anchor){
 TORCH_CHECK(anchor.is_cuda(),"CUDA anchor required");c10::cuda::CUDAGuard guard(anchor.device());
 cudaDeviceProp p;C10_CUDA_CHECK(cudaGetDeviceProperties(&p,anchor.get_device()));TORCH_CHECK(p.major==8&&p.minor==0,"SM80 only");
 using K=C0::Kernel;constexpr int bytes=sizeof(K::SharedStorage);
 if(bytes>48*1024)C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<K>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
 cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,cutlass::Kernel<K>));
 return {K::kThreadCount,bytes,a.numRegs,int64_t(a.localSizeBytes)};
}
at::Tensor run(at::Tensor const& q,at::Tensor const& packed,at::Tensor const& meta,int64_t m,int64_t n){
 TORCH_CHECK(q.is_cuda()&&q.scalar_type()==at::kChar&&q.dim()==2&&q.is_contiguous(),"row-major CUDA INT8 input");
 c10::cuda::CUDAGuard guard(q.device());
 TORCH_CHECK(packed.device()==q.device()&&meta.device()==q.device()&&packed.is_contiguous()&&meta.is_contiguous(),"packed buffer device/layout");
 TORCH_CHECK(packed.scalar_type()==at::kChar&&packed.dim()==2&&meta.scalar_type()==at::kInt,"packed types");
 TORCH_CHECK(q.size(1)%128==0&&q.size(0)%32==0&&packed.size(0)%32==0&&packed.size(1)*2==q.size(1),"128K/32M padding");
 TORCH_CHECK(q.size(1)<=131008&&meta.numel()==packed.size(0)*q.size(1)/32,"integer bound/metadata size");
 TORCH_CHECK(m>0&&m<=q.size(0)&&n>0&&n<=packed.size(0),"logical dimensions");
 auto output=at::empty({m,n},q.options().dtype(at::kInt));
 using K=C0::Kernel;using LE=K::LayoutE;int pn=packed.size(0),pk=q.size(1),pm=q.size(0);
 cutlass::gemm::GemmCoord problem(pn,pm,pk);C0::Swizzle swizzle;
 auto tiled=swizzle.get_tiled_shape(problem,{128,128,128},1);
 K::TensorRefA refa(packed.data_ptr<int8_t>(),Row(pk/2));
 K::TensorRefB refb(q.data_ptr<int8_t>(),Col(q.stride(0)));
 K::TensorRefE refe(reinterpret_cast<K::ElementE*>(meta.data_ptr<int32_t>()),LE::packed({pn,pk/32}));
 Destination::TensorRef refc(nullptr,Row(pm)),refd(output.data_ptr<int32_t>(),Row(pm));
 TORCH_CHECK(K::can_implement(problem,refa,refb,refc,refd,refe)==cutlass::Status::kSuccess,"CUTLASS alignment");
 OutputOp::Params op;op.rows=m;op.columns=n;
 K::Params params(problem,tiled,refa,refb,refc,refd,refe,op,nullptr);
 auto grid=swizzle.get_grid_shape(tiled);auto stream=at::cuda::getCurrentCUDAStream();
 cutlass::Kernel<K><<<grid,K::kThreadCount,sizeof(K::SharedStorage),stream>>>(params);C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
}
TORCH_LIBRARY(sigma_orthogonal_sparse_int32_sm80,m){
 m.def("pack_cpu(Tensor weight,int padded_k)->(Tensor,Tensor)");
 m.def("configure(Tensor anchor)->int[]");
 m.def("run(Tensor q,Tensor packed,Tensor metadata,int m,int n)->Tensor");
}
TORCH_LIBRARY_IMPL(sigma_orthogonal_sparse_int32_sm80,CPU,m){m.impl("pack_cpu",sigma_orthogonal_sparse_int32_sm80::pack_cpu);}
TORCH_LIBRARY_IMPL(sigma_orthogonal_sparse_int32_sm80,CUDA,m){m.impl("configure",sigma_orthogonal_sparse_int32_sm80::configure);m.impl("run",sigma_orthogonal_sparse_int32_sm80::run);}
