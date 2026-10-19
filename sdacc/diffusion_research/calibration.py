"""Shared own-trajectory sampling and FP64 sufficient statistics; no launch on import."""
import sys,ast,textwrap,os,gc,json
from pathlib import Path
import numpy as np
import torch
from torch._subclasses.fake_tensor import is_fake
from torch.nn import functional
from .io import read,write,sha,save_tensor
from .moments import Moments
class Collector:
 def __init__(self,model,policy,memory):
  self.model=model;self.policy=policy;self.memory=memory;self.active=False;self.entries={};self.delta={};self.opcode=torch._ops.OpOverload.__call__.__code__
  self.functions={id(f):f for f in [torch.ops.sdacc_mild_projection_sm80.projection.default,torch.ops.sdacc_dualscale_sm80.run.default,torch.ops.sdacc_mild_projection_sm80.projection_scalar.default]}
  self.prepared=torch.ops.sdacc_prepared_projection_sm80.run.default;self.functions[id(self.prepared)]=self.prepared
  self.by_handle={handle:u for u,state in enumerate(model.states) for op in state.ops.values() for handle in op._graphs.values()}
  self.schemas={k:list(v._schema.arguments) for k,v in self.functions.items()}
  self.by_packed={op.packed.data_ptr():u for u,state in enumerate(model.states) for op in state.ops.values()}
  self.expected={(t,u) for t,row in enumerate(policy) for u,m in enumerate(row) if m}
  self.protected_auxiliary=0;self.q_inputs={};self.quant_frames={};self.quant_roles={}
  self.bulk=model.config.get('bulk_moments',False);self.offsets={};offset=0
  if self.bulk:
   for key in sorted(self.expected):
    size=5*model.units[key[1]]['n'];self.offsets[key]=(offset,offset+size);offset+=size
   self.moment_buffer=torch.empty(offset,dtype=torch.float64,device='cuda')
  from torch._inductor.runtime.triton_heuristics import CachingAutotuner
  self.quant_code=CachingAutotuner.run.__code__
 def values(self,frame,func):
  args=frame.f_locals.get('args',());kwargs=frame.f_locals.get('kwargs',{}) or {};declared=self.schemas[id(func)];assert len(args)<=len(declared)
  values={a.name:v for a,v in zip(declared,args)}
  for a in declared[len(args):]:
   if a.name in kwargs:values[a.name]=kwargs[a.name]
   else:assert a.has_default_value();values[a.name]=a.default_value
  assert set(kwargs).issubset({a.name for a in declared});return values
 def begin(self):assert not self.active and sys.getprofile() is None;self.delta={};self.entries={};self.q_inputs={};self.quant_frames={};self.protected_auxiliary=0;self.active=True;sys.setprofile(self.profile)
 def end(self):
  sys.setprofile(None);self.active=False;assert not self.entries and not self.quant_frames and set(self.delta)==self.expected,f'actualSparse coverage {len(self.delta)}/{len(self.expected)}; missing {sorted(self.expected-set(self.delta))[:8]}'
  if self.bulk:
   host=self.moment_buffer.cpu().numpy()
   self.delta={key:(value[0],host[self.offsets[key][0]:self.offsets[key][1]].reshape(5,self.model.units[key[1]]['n'])) for key,value in self.delta.items()}
  return self.delta
 def cancel(self):sys.setprofile(None);self.active=False;self.entries={};self.delta={};self.q_inputs={};self.quant_frames={}
 def quant(self,frame,event):
  from .quant_inputs import bind_actual_launch
  if event=='call':
   receiver=frame.f_locals['self'];name=receiver.fn.__name__
   if '_quant_register' not in name and '_quant_lift' not in name:return
   vals,_=bind_actual_launch(receiver,frame,lambda v: None)
   if id(receiver.fn) not in self.quant_roles:
    tree=ast.parse(textwrap.dedent(receiver.fn.src));bindings={}
    for node in ast.walk(tree):
     if isinstance(node,ast.Assign):
      for target in node.targets:
       if isinstance(target,ast.Name):bindings[target.id]=node.value
    def roots(node,seen=None):
     seen=set() if seen is None else seen
     if isinstance(node,ast.Name):
      if node.id in vals and isinstance(vals[node.id],torch.Tensor):return {node.id}
      if node.id in bindings and node.id not in seen:return roots(bindings[node.id],seen|{node.id})
     out=set()
     for child in ast.iter_child_nodes(node):out|=roots(child,seen)
     return out
    reads=set();writes=set()
    for node in ast.walk(tree):
     if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.args:
      if node.func.attr=='load':reads|=roots(node.args[0])
      if node.func.attr=='store':writes|=roots(node.args[0])
    self.quant_roles[id(receiver.fn)]=(reads,writes)
   reads,writes=self.quant_roles[id(receiver.fn)];inputs=[vals[n] for n in reads if isinstance(vals[n],torch.Tensor)];outputs=[vals[n] for n in writes if isinstance(vals[n],torch.Tensor) and vals[n].dtype==torch.int8]
   if not outputs:return
   raw=[v for v in inputs if v.dtype==torch.bfloat16]
   parents=[v for v in inputs if v.dtype==torch.int8 and v.data_ptr() in self.q_inputs]
   if raw:
    assert len(raw)==1,'ambiguous actual Quant BF16 input';x=raw[0]
   else:
    assert parents,'actual lift is missing its actual Quant input';x=self.q_inputs[parents[0].data_ptr()]
    assert all(self.q_inputs[v.data_ptr()] is x for v in parents)
   self.quant_frames[id(frame)]=(x,outputs,parents)
  elif event=='return' and id(frame) in self.quant_frames:
   raw,outputs,parents=self.quant_frames.pop(id(frame))
   for v in parents:self.q_inputs.pop(v.data_ptr(),None)
   for v in outputs:self.q_inputs[v.data_ptr()]=raw
 @torch._dynamo.disable
 def profile(self,frame,event,arg):
  if frame.f_code is self.quant_code:
   self.quant(frame,event);return
  if frame.f_code is not self.opcode:return
  func=frame.f_locals.get('self')
  if id(func) not in self.functions:return
  if event=='call':
   values=self.values(frame,func)
   # Compiler shape tracing does not execute a real Sparse projection.
   if any(is_fake(v) for v in values.values() if isinstance(v,torch.Tensor)):return
   if func is self.prepared:
    actual=list(values.values());uid=self.by_handle[actual[1]];raw=actual[0]
   else:uid=self.by_packed[values['packed'].data_ptr()]
   t=self.model.pt.index;mode=self.policy[t][uid]
   if func is self.prepared:pass
   elif 'input' in values:raw=values['input']
   else:
    assert 'q' in values and values['q'].data_ptr() in self.q_inputs,'actual Quant source missing for compiled Sparse input'
    raw=self.q_inputs.pop(values['q'].data_ptr())
   assert raw.dtype==torch.bfloat16
   assert mode and mode==getattr(self.model.states[uid].active, 'mode', self.model.states[uid].active)
   if uid in self.model.bank_slots:assert mode==self.model.bank_slots[uid].current_mode
   assert (t,uid) not in self.delta and id(frame) not in self.entries
   self.entries[id(frame)]=(t,uid,mode,raw)
  elif event=='return':
   if id(frame) not in self.entries:return
   t,u,mode,raw=self.entries.pop(id(frame));assert type(arg) is torch.Tensor
   state=self.model.states[u];unit=self.model.units[u];n=unit['n'];x=raw
   # Full Dense teacher BEFORE selecting valid uniform rows. Original bias once.
   teacher=self.model.calibration_teacher(u,x)
   if getattr(state,'protected',0):
    sparse=state.ops[mode](x);self.protected_auxiliary+=1
   else:sparse=arg
   s=sparse.reshape(-1,n);d=teacher.reshape(-1,n);assert s.shape==d.shape and torch.isfinite(d).all() and torch.isfinite(s).all()
   valid=self.model.calibration_rows(u,s)
   sf=s.index_select(0,valid).double();df=d.index_select(0,valid).double()
   values=torch.stack((sf.sum(0),df.sum(0),(sf*sf).sum(0),(sf*df).sum(0),(df*df).sum(0)))
   if self.bulk:
    first,last=self.offsets[t,u];self.moment_buffer[first:last].copy_(values.flatten());self.delta[t,u]=(len(valid),None)
   else:
    values=values.cpu().numpy();assert np.isfinite(values).all();self.delta[t,u]=(len(valid),values)
   if self.model.config.get('sample_memory_per_projection',True):self.memory.sample()


def collect_conditions(model,policy,output,Memory):
    """Explicit future calibration; never reads heldout images or fits online."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True);variant=model.prepare_variant(policy);widths=[u['n'] for u in model.units]
    identity=dict(model=model.config['model'],checkpoint=model.config['checkpoint'],resolution=model.adapter.spec.resolution,steps=model.steps,calibration_ids=[c['id'] for c in model.cal],policy=policy,own_uncorrected_trajectory=True,full_same_input_Dense_teacher_before_sampling=True,rows=128,FP64=True,evaluation_used=False)
    db=Moments(output/'moments.sqlite',identity,policy,widths);memory=Memory(model);collector=Collector(model,policy,memory)
    first=db.prefix
    call=model.pt.call;captured=hasattr(call,'bypass');previous_bypass=getattr(call,'bypass',False)
    # Reference and collected calls use identical input layouts. Calibration
    # already executes the compiled caller directly so every invocation is seen.
    if captured:call.bypass=True
    try:
        for i in range(first,len(model.cal)):
            condition=model.cal[i]
            for slot in model.bank_slots.values():slot.current_mode=None
            # Warm the first current trajectory before the sampling callback.
            if i==first:
                memory.begin();reference,_=model.generate(condition,variant);warm_peak=memory.finish()
            memory.begin()
            try:collector.begin();latent,_=model.generate(condition,variant);delta=collector.end()
            except BaseException:collector.cancel();raise
            peak=memory.finish()
            if i==first and not torch.equal(latent,reference):raise RuntimeError('collector altered trajectory')
            receipt=db.commit(i,condition['id'],delta,dict(pid=os.getpid(),steps=model.steps,positions=len(delta),own_peak=peak,warm_peak=warm_peak if i==first else None));write(output/'progress.json',receipt);collector.delta={};model.data.pop(condition['id'],None);gc.collect()
        coefficients=db.coefficients();payload={k:dict(a=torch.from_numpy(v['a']),b=torch.from_numpy(v['b']),diagnostic=v['diagnostic']) for k,v in coefficients.items()}
        path=output/'coefficients.pt';save_tensor(path,dict(complete=True,pid=os.getpid(),identity=identity,positions=payload,conditions=100,recovery_coverage=1.0));write(output/'complete.json',dict(complete=True,pid=os.getpid(),conditions=100,positions=len(payload),coefficients_SHA=sha(path)))
    finally:
        if captured:call.bypass=previous_bypass
        db.close()
