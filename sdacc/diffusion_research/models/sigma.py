"""PixArt family: fused QKV/KV routing and three resident packed modes."""
import torch
from torch.nn import functional as F
from ..runtime import Runtime
from ..io import read
from ..banks import load_resident_banks
from .projections import shared_module_graph
class State:
    def __init__(self,descriptor,ops,weight,bias):
        self.desc=descriptor;self.ops=ops;self.weight=weight;self.bias=bias
        self.sparse_on=False;self.active=None
    def select(self,mode):
        self.sparse_on=bool(mode);self.active=self.ops[mode] if mode else None
    def project(self,x):
        # Metadata views keep the public kernel's actual M/K/N invocation.
        y=self.active(x.reshape(-1,self.desc['k']))
        return y.reshape(*x.shape[:-1],self.desc['n'])
    def teacher(self,x):
        widths=[self.weight.shape[0]//len(self.desc['targets'])]*len(self.desc['targets'])
        return torch.cat([F.linear(x,w,b) for w,b in zip(self.weight.split(widths),self.bias.split(widths))],-1)

class Proxy(torch.nn.Module):
    def __init__(self,state,original):super().__init__();self.state=state;self.original=original
    @property
    def weight(self):return self.original.weight
    @property
    def bias(self):return self.original.bias
    def forward(self,x):return self.state.project(x) if self.state.sparse_on else self.original(x)

class Joined(torch.nn.Module):
    def __init__(self,state):super().__init__();self.state=state
    def forward(self,x):return self.state.project(x)

class Processor:
    def __init__(self,state,original,fused):self.state=state;self.original=original;self.fused=fused
    def __call__(self,attn,hidden_states,encoder_hidden_states=None,attention_mask=None,**kwargs):
        processor=self.fused if self.state.sparse_on else self.original
        return processor(attn,hidden_states,encoder_hidden_states=encoder_hidden_states,attention_mask=attention_mask,**kwargs)

def install_projections(native,units,operators):
    """Shared PixArt/DiT routing; keep native attention and original Dense weights."""
    from diffusers.models.attention_processor import FusedAttnProcessor2_0
    patched=shared_module_graph(native);states=[]
    for descriptor in units:
        modules=[patched.get_submodule(name) for name in descriptor['targets']]
        weight=torch.cat([m.weight for m in modules]).detach();bias=torch.cat([m.bias for m in modules]).detach()
        state=State(descriptor,operators[descriptor['unit']],weight,bias);states.append(state)
        for name,original in zip(descriptor['targets'],modules):
            parent,leaf=name.rsplit('.',1);setattr(patched.get_submodule(parent),leaf,Proxy(state,original))
        if len(modules)>1:
            parent=patched.get_submodule(descriptor['targets'][0].rsplit('.',1)[0])
            parent.set_processor(Processor(state,parent.processor,FusedAttnProcessor2_0()))
            setattr(parent,'to_qkv' if len(modules)==3 else 'to_kv',Joined(state))
    return patched,states


class SigmaModel(Runtime):
    spec_name='pixart-sigma'
    parameter_transport='D2D'
    def __init__(self,config,output):
        self.start(config,output);self.units=read(config['units'])['execution_units']
        self.patched,self.states=install_projections(self.native,self.units,load_resident_banks(self))
        self.finish()

    def calibration_teacher(self,uid,x):return self.states[uid].teacher(x)
    def calibration_rows(self,uid,s):
        valid=torch.arange(s.shape[0],device=s.device)
        if self.units[uid]['module']=='cross-KV':valid=valid[self.text_mask.bool().flatten()]
        return valid[torch.linspace(0,len(valid)-1,min(128,len(valid)),device=s.device).long()]


def projection_inventory(model,resolution,text_length):
    """Shared PixArt projection groups; attention core/out remain Dense."""
    units=[]
    for block in range(len(model.transformer_blocks)):
        prefix=f'transformer_blocks.{block}'
        groups=[('QKV',[prefix+'.attn1.to_'+x for x in ['q','k','v']]),
                ('cross-Q',[prefix+'.attn2.to_q']),('cross-KV',[prefix+'.attn2.to_'+x for x in ['k','v']]),
                ('FFN-in',[prefix+'.ff.net.0.proj']),('FFN-out',[prefix+'.ff.net.2'])]
        for role,names in groups:
            modules=[model.get_submodule(n) for n in names]
            if len({m.in_features for m in modules})!=1:raise ValueError('projection inputs differ')
            units.append(dict(unit=len(units),block=block,module=role,targets=names,
                m=2*(text_length if role=='cross-KV' else (resolution//16)**2),
                k=modules[0].in_features,n=sum(m.out_features for m in modules)))
    return units

SigmaModel.inventory=staticmethod(projection_inventory)
