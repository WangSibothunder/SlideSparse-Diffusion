"""Sigma adaptation of native CacheQuant DPS/DEC residual-cache boundary.

Block0/27 always execute, middle1..26 cache residual. A frozen native
quantization model is supplied explicitly; this host never trains/fits online.
"""
import torch
from ..integrations.cachequant.contracts import FORCED_REFRESH

class ResidualContext:
 def __init__(self):
  self.step=0;self.refresh=tuple(range(20));self.scope=None;self.residual=None;self.entry=None;self.producer_step=None;self.coefficient_source=None;self.a=None;self.b=None
 def begin(self,sample,cfg_roles,stage,policy_sha,coefficient_source):
  scope=(sample,tuple(cfg_roles),stage,policy_sha,coefficient_source)
  if self.scope!=scope:self.residual=None;self.entry=None;self.producer_step=None
  self.scope=scope;self.coefficient_source=coefficient_source
 def select(self,step,a=None,b=None):
  self.step=step;self.a=a;self.b=b
 def full(self):return self.step in self.refresh or self.step in FORCED_REFRESH or self.residual is None

class SigmaResidualBlock(torch.nn.Module):
 def __init__(self,original,context,index):super().__init__();self.original=original;self.context=context;self.index=index
 def forward(self,hidden_states,*args,**kwargs):
  c=self.context
  if len(c.refresh)==20 or self.index in (0,27):return self.original(hidden_states,*args,**kwargs)
  if self.index==1:c.entry=hidden_states
  if c.full():
   output=self.original(hidden_states,*args,**kwargs)
   if self.index==26:c.residual=output-c.entry;c.producer_step=c.step
   return output
  if self.index==26:
   if c.a is None or c.b is None:raise ValueError('frozen DEC coefficients missing at cached boundary')
   corrected=(c.residual.float()*c.a+c.b).to(hidden_states.dtype)
   return hidden_states+corrected
  return hidden_states

def install(model,context):
 if len(model.transformer_blocks)!=28:raise ValueError('Sigma28 blocks')
 for i,b in enumerate(list(model.transformer_blocks)):model.transformer_blocks[i]=SigmaResidualBlock(b,context,i)

def sparse_DEC_composition(sparse_a,sparse_b,DEC_a,DEC_b):
 # Cache-input correction has another boundary; never merge it with Sparse.
 return DEC_a*sparse_a,DEC_a*sparse_b+DEC_b
