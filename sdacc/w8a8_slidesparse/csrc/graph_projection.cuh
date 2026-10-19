// Prepared two-node CUDA graph. No capture/search/configure in forward.
// Input and fresh output pointers are updated for every launch. Earlier outputs
// remain independent, including calls made before a GPU synchronization.
namespace sdacc_prepared_projection_sm80 {
using namespace sdacc_dualscale_sm80;
template<class Kernel>
__global__
void SparseGemmIndirect(typename Kernel::Params params,int64_t const* destination) {
  extern __shared__ int storage[];
  params.ref_D=Destination::TensorRef(reinterpret_cast<BF*>(destination[0]),Row(params.problem_size.n()));
  Kernel op;op(params,*reinterpret_cast<typename Kernel::SharedStorage*>(storage));
}
struct Plan {
  at::Tensor packed,metadata,sw,bias,q,sa,destination;
  int k,n,m,pm,pk,group,device;bool packed16,singlepatch;bool direct2=false;
  cudaStream_t owner;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
  cudaGraphNode_t quant_node=nullptr,gemm_node=nullptr;
  cudaKernelNodeParams quant{};std::mutex mutex;
  virtual at::Tensor forward(at::Tensor const&)=0;
  virtual std::vector<int64_t> resources()=0;
  virtual ~Plan() {
    if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);
  }
};
template<int Width,int L>
void quant_function(cudaKernelNodeParams& p,bool packed16,bool singlepatch) {
  if(singlepatch){p.func=packed16?reinterpret_cast<void*>(sdacc_mild_projection_sm80::quant_lift_bridge_packed16<Width,L>):reinterpret_cast<void*>(sdacc_mild_projection_sm80::quant_lift_bridge_native<Width,L>);return;}
  p.func=packed16?reinterpret_cast<void*>(sdacc_mild_projection_sm80::quant_lift_packed16<Width,L>):
      reinterpret_cast<void*>(sdacc_mild_projection_sm80::quant_lift<Width,L>);
}
template<int Width>
void quant_function_width(cudaKernelNodeParams& p,int l,bool packed16,bool singlepatch) {
  if(l==4)quant_function<Width,4>(p,packed16,singlepatch);
  else if(l==6)quant_function<Width,6>(p,packed16,singlepatch);
  else quant_function<Width,8>(p,packed16,singlepatch);
}
template<class Config>
struct TypedPlan:Plan {
  using Kernel=typename Config::Kernel;
  typename Kernel::Params params;
  cudaKernelNodeParams gemm{};
  TypedPlan(at::Tensor const& p,at::Tensor const& e,at::Tensor const& s,
      at::Tensor const& b,int width,int rows,int cols,int l,bool front16,bool bridge) {
    packed=p;metadata=e;sw=s;bias=b;k=width;m=rows;n=cols;group=l;
    pm=((m+31)/32)*32;pk=packed.size(1)*2;device=p.get_device();packed16=front16;singlepatch=bridge;destination=at::empty({1},p.options().dtype(at::kLong));
    owner=at::cuda::getCurrentCUDAStream();
    q=at::empty({pm,pk},packed.options());sa=at::empty({pm},sw.options());
    using LE=typename Kernel::LayoutE;
    cutlass::gemm::GemmCoord problem(packed.size(0),pm,pk);
    typename Config::Swizzle swizzle;
    auto tiled=swizzle.get_tiled_shape(problem,{Config::Shape::kM,Config::Shape::kN,Config::Shape::kK},1);
    typename Kernel::TensorRefA a(packed.data_ptr<int8_t>(),Row(pk/2));
    typename Kernel::TensorRefB bb(q.data_ptr<int8_t>(),Col(pk));
    typename Kernel::TensorRefE ee(reinterpret_cast<typename Kernel::ElementE*>(metadata.data_ptr<int32_t>()),LE::packed({int(packed.size(0)),pk/32}));
    Destination::TensorRef c(nullptr,Row(pm));
    // Placeholder is never dereferenced: nodes only execute after pointer patch.
    Destination::TensorRef d(reinterpret_cast<BF*>(q.data_ptr<int8_t>()),Row(pm));
    TORCH_CHECK(Kernel::can_implement(problem,a,bb,c,d,ee)==cutlass::Status::kSuccess,"prepared graph alignment");
    OutputOp::Params op;op.sa=sa.data_ptr<float>();op.sw=sw.data_ptr<float>();
    op.bias=bias.numel()?bias.data_ptr<float>():nullptr;op.rows=m;op.columns=n;
    params=typename Kernel::Params(problem,tiled,a,bb,c,d,ee,op,nullptr);
    quant.gridDim=dim3(pm);quant.blockDim=dim3(128);quant.sharedMemBytes=0;
    if(k==1152)quant_function_width<1152>(quant,l,packed16,singlepatch);
    else if(k==1536)quant_function_width<1536>(quant,l,packed16,singlepatch);
    else if(k==2432)quant_function_width<2432>(quant,l,packed16,singlepatch);
    else if(k==4608)quant_function_width<4608>(quant,l,packed16,singlepatch);
    else if(k==6144)quant_function_width<6144>(quant,l,packed16,singlepatch);
    else quant_function_width<9728>(quant,l,packed16,singlepatch);
    const at::BFloat16* ptr=reinterpret_cast<const at::BFloat16*>(q.data_ptr<int8_t>());
    int64_t stride=k;int8_t* qp=q.data_ptr<int8_t>();float* sp=sa.data_ptr<float>();
    at::BFloat16* target=reinterpret_cast<at::BFloat16*>(q.data_ptr<int8_t>());int64_t* dp=destination.data_ptr<int64_t>();
    void* qa[]={&ptr,&stride,&qp,&sp,&m,&pk,&target,&dp};quant.kernelParams=qa;
    gemm.func=singlepatch?reinterpret_cast<void*>(SparseGemmIndirect<Kernel>):reinterpret_cast<void*>(cutlass::Kernel<Kernel>);
    if(singlepatch){if(Config::kStages==2)C10_CUDA_CHECK(cudaFuncSetAttribute(gemm.func,cudaFuncAttributePreferredSharedMemoryCarveout,100));
      if(sizeof(typename Kernel::SharedStorage)>48*1024)C10_CUDA_CHECK(cudaFuncSetAttribute(gemm.func,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename Kernel::SharedStorage)));}
    gemm.gridDim=swizzle.get_grid_shape(tiled);gemm.blockDim=dim3(Kernel::kThreadCount);
    gemm.sharedMemBytes=sizeof(typename Kernel::SharedStorage);
    void* ga[]={&params,&dp};gemm.kernelParams=ga;
    C10_CUDA_CHECK(cudaGraphCreate(&graph,0));
    C10_CUDA_CHECK(cudaGraphAddKernelNode(&quant_node,graph,nullptr,0,&quant));
    C10_CUDA_CHECK(cudaGraphAddKernelNode(&gemm_node,graph,&quant_node,1,&gemm));
    C10_CUDA_CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    // Each API copies parameter values. Stored pointers below must never retain
    // the addresses of these constructor locals.
    quant.kernelParams=nullptr;gemm.kernelParams=nullptr;
  }
  std::vector<int64_t> resources() override {
    cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,gemm.func));int blocks=0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,gemm.func,Kernel::kThreadCount,gemm.sharedMemBytes));
    return {Kernel::kThreadCount,int64_t(gemm.sharedMemBytes),a.numRegs,int64_t(a.localSizeBytes),blocks};
  }
  at::Tensor forward(at::Tensor const& input) override {
    TORCH_CHECK(input.is_cuda()&&input.get_device()==device&&input.scalar_type()==at::kBFloat16&&input.size(-1)==k,"prepared input identity");
    auto flat=input.reshape({m,k});if(flat.stride(1)!=1)flat=flat.contiguous();
    auto shape=input.sizes().vec();shape.back()=n;
    auto out=at::empty(shape,input.options());
    const at::BFloat16* ptr=flat.data_ptr<at::BFloat16>();int64_t stride=flat.stride(0);
    auto stream=at::cuda::getCurrentCUDAStream(device);
    cudaStreamCaptureStatus capture;C10_CUDA_CHECK(cudaStreamIsCapturing(stream,&capture));
    if(direct2||stream.stream()!=owner||capture!=cudaStreamCaptureStatusNone) {
      // A different stream or an outer capture gets independent scratch. It
      // executes the identical kernels and never waits on shared graph scratch.
      bool resident=direct2&&stream.stream()==owner&&capture==cudaStreamCaptureStatusNone;
      std::unique_lock<std::mutex> lock(mutex,std::defer_lock);if(resident)lock.lock();
      auto qq=resident?q:at::empty({pm,pk},q.options());auto ss=resident?sa:at::empty({pm},sa.options());auto dest=resident?destination:at::empty({1},destination.options());
      int8_t* qp=qq.data_ptr<int8_t>();float* sp=ss.data_ptr<float>();
      auto* target=out.data_ptr<at::BFloat16>();int64_t* dp=dest.data_ptr<int64_t>();
      void* qa[]={&ptr,&stride,&qp,&sp,&m,&pk,&target,&dp};
      C10_CUDA_CHECK(cudaLaunchKernel(quant.func,quant.gridDim,quant.blockDim,qa,0,stream));
      auto local=params;local.ref_B=typename Kernel::TensorRefB(qp,Col(pk));
      local.output_op.sa=sp;local.ref_D=Destination::TensorRef(reinterpret_cast<BF*>(out.data_ptr<at::BFloat16>()),Row(pm));
      void* ga[]={&local,&dp};C10_CUDA_CHECK(cudaLaunchKernel(gemm.func,gemm.gridDim,gemm.blockDim,ga,gemm.sharedMemBytes,stream));
    } else {
      std::lock_guard<std::mutex> lock(mutex);
      int8_t* qp=q.data_ptr<int8_t>();float* sp=sa.data_ptr<float>();
      auto* target=out.data_ptr<at::BFloat16>();int64_t* dp=destination.data_ptr<int64_t>();
      void* qa[]={&ptr,&stride,&qp,&sp,&m,&pk,&target,&dp};auto updated_quant=quant;updated_quant.kernelParams=qa;
      params.ref_D=Destination::TensorRef(reinterpret_cast<BF*>(out.data_ptr<at::BFloat16>()),Row(pm));
      void* ga[]={&params};auto updated_gemm=gemm;updated_gemm.kernelParams=ga;
      C10_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(exec,quant_node,&updated_quant));
      if(!singlepatch)C10_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(exec,gemm_node,&updated_gemm));
      C10_CUDA_CHECK(cudaGraphLaunch(exec,stream));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
  }
};
struct Registry {std::mutex mutex;int64_t next=1;std::unordered_map<int64_t,std::shared_ptr<Plan>> plans;};
Registry& registry(){static Registry* r=new Registry();return *r;}
template<class Config>
std::shared_ptr<Plan> create(at::Tensor const& p,at::Tensor const& e,at::Tensor const& s,
    at::Tensor const& b,int k,int m,int n,int l,bool packed16,bool singlepatch) {
  return std::make_shared<TypedPlan<Config>>(p,e,s,b,k,m,n,l,packed16,singlepatch);
}
int64_t prepare_impl(at::Tensor const& p,at::Tensor const& e,at::Tensor const& s,
    at::Tensor const& b,int64_t k,int64_t m,int64_t n,int64_t l,int64_t config,bool packed16,bool singlepatch) {
  TORCH_CHECK(p.is_cuda()&&m>0&&n>0&&(k==1152||k==1536||k==2432||k==4608||k==6144||k==9728)&&(l==4||l==6||l==8),"prepared graph geometry");
  c10::cuda::CUDAGuard guard(p.device());
  for(auto const& t:{e,s,b})TORCH_CHECK(t.device()==p.device()&&t.is_contiguous(),"prepared buffer identity");
  TORCH_CHECK(p.is_contiguous()&&p.scalar_type()==at::kChar&&e.scalar_type()==at::kInt&&s.scalar_type()==at::kFloat&&b.scalar_type()==at::kFloat,"prepared buffer types");
  int lifted=((k+l-1)/l)*(l/2-1)*4,pk=((lifted+127)/128)*128;
  TORCH_CHECK(p.size(1)*2==pk&&p.size(0)%32==0&&p.size(0)>=n&&s.numel()>=n&&(b.numel()==0||b.numel()>=n)&&e.numel()==p.size(0)*pk/32,"prepared packing");
  // Resource configuration must have been performed explicitly before prepare.
  std::shared_ptr<Plan> plan;
#define SDACC_GRAPH_CONFIG(ID,CHANNELS,TOKENS) \
  if(config==ID){if(n%CHANNELS==0&&m%TOKENS==0)plan=create<C##ID##F>(p,e,s,b,k,m,n,l,packed16,singlepatch);else plan=create<C##ID>(p,e,s,b,k,m,n,l,packed16,singlepatch);}
  SDACC_GRAPH_CONFIG(5,128,128)
  else SDACC_GRAPH_CONFIG(11,128,128)
  else SDACC_GRAPH_CONFIG(12,128,64)
  else SDACC_GRAPH_CONFIG(14,64,64)
  else SDACC_GRAPH_CONFIG(15,64,64)
  else SDACC_GRAPH_CONFIG(16,128,64)
  else SDACC_GRAPH_CONFIG(17,64,128)
  else TORCH_CHECK(false,"unsupported prepared config");
#undef SDACC_GRAPH_CONFIG
  auto& r=registry();std::lock_guard<std::mutex> lock(r.mutex);int64_t id=r.next++;r.plans[id]=plan;return id;
}
at::Tensor run(at::Tensor const& input,int64_t id,int64_t n) {
  std::shared_ptr<Plan> p;
  {auto& r=registry();std::lock_guard<std::mutex> lock(r.mutex);auto it=r.plans.find(id);TORCH_CHECK(it!=r.plans.end(),"prepared graph handle absent");p=it->second;}
  TORCH_CHECK(input.numel()==int64_t(p->m)*p->k&&n==p->n,"prepared shape changed");
  c10::cuda::CUDAGuard guard(input.device());return p->forward(input);
}
void release(int64_t id) {
  std::shared_ptr<Plan> p;
  {auto& r=registry();std::lock_guard<std::mutex> lock(r.mutex);auto it=r.plans.find(id);if(it==r.plans.end())return;p=it->second;r.plans.erase(it);}
  c10::cuda::CUDAGuard guard(p->device);C10_CUDA_CHECK(cudaStreamSynchronize(p->owner));
}
int64_t prepare(at::Tensor const& p,at::Tensor const& e,at::Tensor const& s,at::Tensor const& b,int64_t k,int64_t m,int64_t n,int64_t l,int64_t config,bool packed16){return prepare_impl(p,e,s,b,k,m,n,l,config,packed16,false);}
int64_t prepare_singlepatch(at::Tensor const& p,at::Tensor const& e,at::Tensor const& s,at::Tensor const& b,int64_t k,int64_t m,int64_t n,int64_t l,int64_t config,bool packed16){return prepare_impl(p,e,s,b,k,m,n,l,config,packed16,true);}
int64_t prepare_direct2(at::Tensor const& p,at::Tensor const& e,at::Tensor const& s,at::Tensor const& b,int64_t k,int64_t m,int64_t n,int64_t l,int64_t config,bool packed16){
  auto id=prepare_impl(p,e,s,b,k,m,n,l,config,packed16,false);
  auto& reg=registry();std::lock_guard<std::mutex> lock(reg.mutex);auto plan=reg.plans.at(id);
  C10_CUDA_CHECK(cudaGraphExecDestroy(plan->exec));plan->exec=nullptr;
  C10_CUDA_CHECK(cudaGraphDestroy(plan->graph));plan->graph=nullptr;plan->direct2=true;return id;
}
std::vector<int64_t> resources(int64_t handle){auto& r=registry();std::shared_ptr<Plan> p;{std::lock_guard<std::mutex> lock(r.mutex);p=r.plans.at(handle);}c10::cuda::CUDAGuard guard(p->packed.device());return p->resources();}
}
TORCH_LIBRARY(sdacc_prepared_projection_sm80,m) {
  m.def("prepare_direct2(Tensor packed,Tensor metadata,Tensor scales,Tensor bias,int k,int m,int n,int group,int config,bool packed16)->int");
  m.def("prepare_singlepatch(Tensor packed,Tensor metadata,Tensor scales,Tensor bias,int k,int m,int n,int group,int config,bool packed16)->int");
  m.def("resources(int handle)->int[]");
  m.def("prepare(Tensor packed,Tensor metadata,Tensor scales,Tensor bias,int k,int m,int n,int group,int config,bool packed16)->int");
  m.def("run(Tensor input,int handle,int n)->Tensor");
  m.def("release(int handle)->()");
}
TORCH_LIBRARY_IMPL(sdacc_prepared_projection_sm80,CUDA,m) {
  m.impl("prepare",sdacc_prepared_projection_sm80::prepare);
  m.impl("prepare_direct2",sdacc_prepared_projection_sm80::prepare_direct2);
  m.impl("prepare_singlepatch",sdacc_prepared_projection_sm80::prepare_singlepatch);
  m.impl("run",sdacc_prepared_projection_sm80::run);
}
TORCH_LIBRARY_IMPL(sdacc_prepared_projection_sm80,CatchAll,m) {
  m.impl("release",sdacc_prepared_projection_sm80::release);
  m.impl("resources",sdacc_prepared_projection_sm80::resources);
}
