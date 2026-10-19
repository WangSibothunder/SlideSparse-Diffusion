"""One generation, conditioning, policy selector and timing path for every model."""
import os,copy,sys
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from .io import read,write,sha
from .library import bind
from .pipeline import Adapter,SPECS
from .resources import PhysicalGPU

class Timer(torch.nn.Module):
    def __init__(self,model,call,steps,select=None):
        super().__init__();self.model=model;self.call=call;self.select=select;self.index=0;self.enabled=False
        self.events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(steps)]
    def __getattr__(self,name):
        try:return super().__getattr__(name)
        except AttributeError:return getattr(super().__getattr__('model'),name)
    def forward(self,*args,**kwargs):
        if self.index>=len(self.events):raise ValueError('extra denoiser call')
        if self.enabled:self.events[self.index][0].record()
        if self.select is not None:self.select(self.index)
        result=self.call(*args,**kwargs)
        if self.enabled:self.events[self.index][1].record()
        self.index+=1;return result


class CapturedCall(torch.nn.Module):
    """Replay a warmed complete denoiser; parameter/input copies stay timed.

    Prepared projections explicitly support an outer capture. This removes
    thousands of per-projection graph pointer patches and host dispatches.
    Inputs/weights are allocated outside capture and outputs stay referenced.
    A serial runtime may share transient graph scratch to fit larger models.
    """
    def __init__(self,call,key,timing,pool=None):
        super().__init__()
        self.call=call;self.key=key;self.timing=timing;self.graphs={};self.max_nrmse=0.;self.pool=pool;self.bypass=False
    @staticmethod
    def clone(value):
        if isinstance(value,torch.Tensor):return value.clone(memory_format=torch.preserve_format)
        if isinstance(value,dict):return {k:CapturedCall.clone(v) for k,v in value.items()}
        if isinstance(value,tuple):return tuple(CapturedCall.clone(v) for v in value)
        if isinstance(value,list):return [CapturedCall.clone(v) for v in value]
        return value
    @staticmethod
    def copy(target,source):
        if isinstance(target,torch.Tensor):target.copy_(source,non_blocking=True)
        elif isinstance(target,dict):
            for k,v in target.items():CapturedCall.copy(v,source[k])
        elif isinstance(target,(list,tuple)):
            for a,b in zip(target,source):CapturedCall.copy(a,b)
        elif target!=source:raise ValueError('captured non-tensor input changed')
    @staticmethod
    def tensors(value):
        if isinstance(value,torch.Tensor):return [value]
        if isinstance(value,dict):return [t for v in value.values() for t in CapturedCall.tensors(v)]
        if isinstance(value,(list,tuple)):return [t for v in value for t in CapturedCall.tensors(v)]
        return []
    def forward(self,*args,**kwargs):
        # Calibration's Python collector must see every actual invocation.
        if self.bypass or sys.getprofile() is not None:return self.call(*args,**kwargs)
        key=self.key()
        if key not in self.graphs:
            if self.timing():raise RuntimeError('warm the denoiser route before timing')
            inputs=self.clone((args,kwargs))
            reference=self.clone(self.call(*inputs[0],**inputs[1]))
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool):output=self.call(*inputs[0],**inputs[1])
            graph.replay();torch.cuda.synchronize()
            for actual,expected in zip(self.tensors(output),self.tensors(reference)):
                error=float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12))
                self.max_nrmse=max(self.max_nrmse,error)
                if not torch.isfinite(actual).all() or error>.01:raise RuntimeError('captured denoiser differs from its compiled caller')
            self.graphs[key]=(inputs,graph,output)
        else:
            inputs,graph,output=self.graphs[key];self.copy(inputs,(args,kwargs));graph.replay()
        return output
    def close(self,owner=None):
        if owner is None:self.graphs.clear()
        else:
            for key in list(self.graphs):
                if key[0] is owner:del self.graphs[key]

class Runtime:
    spec_name=None
    def start(self,config,output):
        self.config=config;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.library=bind(config['library']);self.physical=PhysicalGPU(os.environ['SDACC_PHYSICAL_GPU'])
        if self.physical.uuid!=os.environ['CUDA_VISIBLE_DEVICES']:raise ValueError('physical GPU differs from selected UUID')
        self.physical.wait_clean();torch.set_num_threads(8);torch._dynamo.config.cache_size_limit=512
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        import torch._inductor.config as compiler
        compiler.triton.cudagraphs=False;compiler.freezing=False
        spec=replace(SPECS[config.get('spec_name',self.spec_name)],**config.get('spec',{}));self.steps=spec.steps
        self.adapter=Adapter(spec,Path(config['checkpoint']),cached_only=True)
        self.adapter.model.requires_grad_(False);self.physical.wait_clean();self.adapter.denoiser_gpu();self.native=self.adapter.model
        self.bank_slots={};self.data={};self.variant=None;self.selector=None;self.transfer_records=[]
        contract=read(config['conditions']);splits=contract.get('conditions',contract)
        self.cal=splits['calibration'];self.pilot=splits.get('pilot',splits.get('expansion',[]))[:4];self.evaluation=splits['evaluation'][:64]
        self.condition_sources={v['id']:v for v in contract.get('conditioning',[])}
        if [len(self.cal),len(self.pilot),len(self.evaluation)]!=[100,4,64]:raise ValueError('fixed100/4/64 condition split required')
        groups=[{v['id'] for v in x} for x in [self.cal,self.pilot,self.evaluation]]
        if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):raise ValueError('overlapping experiment splits')
    def finish(self):
        options=self.config.get('compile_options',{})
        self.nt=Timer(self.native,torch.compile(self.native,fullgraph=True,dynamic=False,options=options),self.steps)
        self.pt=Timer(self.patched,torch.compile(self.patched,fullgraph=True,dynamic=False,options=options),self.steps,self.select)
        if self.config.get('sparse_cuda_graph',False):
            pool=torch.cuda.graph_pool_handle() if self.config.get('share_graph_scratch',False) else None
            self.pt.call=CapturedCall(self.pt.call,lambda:(None if hasattr(self,'coefficient_arena') else self.selector,tuple(self.variant['policy'][self.pt.index])),lambda:self.pt.enabled,pool=pool)
        if self.config.get('benchmark_dense_graph_control',False):
            self.gt=Timer(self.native,self.nt.call,self.steps)
            self.gt.call=CapturedCall(self.gt.call,lambda:None,lambda:self.gt.enabled)
        self.et=Timer(self.native,self.native,self.steps)
        write(self.output/'runtime.json',dict(model=self.config['model'],PID=os.getpid(),GPU=int(os.environ['SDACC_PHYSICAL_GPU']),GPU_UUID=self.physical.uuid,steps=self.steps,units=len(self.states),spec=self.adapter.receipt,library_binding_SHA=sha(Path(self.config['library'])/'SOURCE_BINDING.json')))
    def prepare_variant(self,policy):
        policy=np.asarray(policy)
        if policy.shape!=(self.steps,len(self.states)) or not np.issubdtype(policy.dtype,np.integer) or not np.isin(policy,[0,4,6,8]).all():raise ValueError('policy dimensions or integer modes differ from model')
        updates=[];previous=[-1]*len(self.states)
        for row in policy:
            updates.append([(uid,int(mode)) for uid,mode in enumerate(row) if previous[uid]!=mode]);previous=row.tolist()
        variant=dict(policy=policy.tolist(),updates=updates)
        if hasattr(self,'coefficient_arena'):
            parameters=[]
            for step,row in enumerate(variant['policy']):
                if not updates[step]:parameters.append(None);continue
                key=tuple(row)
                if key not in self.raw_parameters_cache:
                    values=torch.zeros(self.coefficient_arena.numel(),dtype=torch.float32,pin_memory=True)
                    for uid,mode in enumerate(row):
                        if not mode:continue
                        offset,n=self.coefficient_ranges[uid];source=self.bank_slots[uid].host[mode]
                        values[offset:offset+n].copy_(source['scales']);values[offset+n:offset+2*n].copy_(source['bias'])
                    self.raw_parameters_cache[key]=values
                parameters.append(self.raw_parameters_cache[key])
            variant['raw_parameters']=parameters
        return variant
    def select(self,step):
        if self.selector is not None:return self.selector.select(step)
        self.select_raw(step)
    def select_raw(self,step):
        bulk=hasattr(self,'coefficient_arena');moved=0;changed=0
        for uid,mode in self.variant['updates'][step]:
            if mode and uid in self.bank_slots:
                count=self.bank_slots[uid].activate(mode,include_parameters=not bulk);moved+=count;changed+=bool(count)
            self.states[uid].select(mode)
        coefficient_bytes=0
        if bulk and self.selector is None and self.variant['raw_parameters'][step] is not None:
            self.coefficient_arena.copy_(self.variant['raw_parameters'][step],non_blocking=True)
            coefficient_bytes=self.coefficient_arena.numel()*self.coefficient_arena.element_size()
        if bulk and (moved or coefficient_bytes):self.transfer_records.append(dict(step=step,weight_metadata_bytes=moved,weight_mode_changes=changed,raw_parameter_bytes=coefficient_bytes,direction='H2D',scope='Timer.select'))
    def conditioning(self,condition):
        if self.adapter.family=='dit':return {},None
        key=condition['id']
        if key not in self.data:
            source=self.condition_sources.get(key)
            path=source['path'] if source else str(Path(self.config['conditioning_directory'])/(key+'.pt'))
            if source and sha(path)!=source['sha']:raise ValueError('conditioning bytes changed')
            packet=torch.load(path,map_location='cpu',weights_only=True)
            if packet['condition']!=condition:raise ValueError('conditioning differs')
            expected=self.adapter.receipt['configs_sha256'];actual=packet['checkpoint_configs']
            if self.config.get('reuse_text_conditioning'):
                keys=[k for k in expected if k.startswith(('text_encoder/','tokenizer/'))]
                if not keys or any(actual.get(k)!=expected[k] for k in keys):raise ValueError('text conditioning encoder/tokenizer differs')
            elif actual!=expected:raise ValueError('conditioning/checkpoint identity differs')
            self.data[key]=self.adapter.combine([packet.get('payload',packet.get('data'))])
        return self.data[key]
    @torch.inference_mode()
    def generate(self,condition,variant=None,timing=False,eager=False,dense_graph=False):
        self.variant=variant;self.transfer_records=[];timer=self.gt if dense_graph else self.et if eager else self.nt if variant is None else self.pt
        if eager and variant is not None:raise ValueError('eager teacher must remain Dense')
        timer.index=0;timer.enabled=timing;data,self.text_mask=self.conditioning(condition);self.mask=self.text_mask
        self.adapter.pipe.transformer=timer;latent,_=self.adapter.generate([condition],data,return_cpu=True);torch.cuda.synchronize()
        if timer.index!=self.steps or not torch.isfinite(latent).all():raise ValueError('incomplete or nonfinite trajectory')
        return latent,dict(pid=os.getpid(),condition=condition['id'],denoiser_ms=sum(s.elapsed_time(e) for s,e in timer.events) if timing else None)
    def close(self):
        for timer in (getattr(self,name) for name in ('nt','pt','gt') if hasattr(self,name)):
            if isinstance(timer.call,CapturedCall):timer.call.close()
        for state in self.states:
            for op in state.ops.values():
                if hasattr(op,'close_graphs'):op.close_graphs()
        self.physical.close()
