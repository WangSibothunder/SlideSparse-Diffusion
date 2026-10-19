"""Same-input member views and BF16 protected attention contribution."""
import torch
from torch.nn import functional as F
class SharedProjection(torch.nn.Module):
 def __init__(self,packed,widths):
  super().__init__();self.packed=packed;self.widths=tuple(widths)
  if sum(self.widths)!=packed.n:raise ValueError('member widths/order mismatch')
 def configure(self):return self.packed.configure()
 def forward(self,x):return self.packed(x).split(self.widths,dim=-1)
class ProtectedFFNOutput(torch.nn.Module):
 def __init__(self,ffn,attention_weight):
  super().__init__();self.ffn=ffn;self.protected=attention_weight.shape[1]
  if attention_weight.dtype!=torch.bfloat16 or attention_weight.shape[0]!=ffn.n:raise ValueError('protected BF16 weight shape mismatch')
  self.register_buffer('attention_weight',attention_weight)
 def configure(self):return dict(ffn=self.ffn.configure(),mixed=self.ffn.configure(has_addend=True))
 def forward(self,x):
  if x.shape[-1]!=self.protected+self.ffn.k:raise ValueError('physical mixed input width mismatch')
  protected=F.linear(x[...,:self.protected],self.attention_weight,None)
  return self.ffn(x[...,self.protected:],output_addend=protected)
