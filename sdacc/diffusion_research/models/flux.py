"""FLUX-specific dual/single blocks, shared BF16 storage, protected attention."""
import copy
import torch
from ..runtime import Runtime
from ..io import read
from ..banks import load_flux_banks,coalesce_parameters
from .flux_layers import ProjectionState,OrdinaryProxy,SingleBlockProxy,processor_from_original
from .projections import shared_module_graph,verify_shared_tensors

class DoubleQKV:
    def __init__(self,image,text):self.image=image;self.text=text
    def __call__(self,attn,hidden_states,encoder_hidden_states):
        q,k,v=self.image.members_output(hidden_states)
        eq,ek,ev=(None,None,None) if encoder_hidden_states is None else self.text.members_output(encoder_hidden_states)
        return q,k,v,eq,ek,ev

def patch_shared_graph(original,descriptors,ops_by_unit):
    clone=shared_module_graph(original);proof=verify_shared_tensors(original,clone);states=[]
    for descriptor in descriptors:
        if descriptor['unit']!=len(states):raise ValueError('ordered physical units required')
        modules=[clone.get_submodule(name) for name in descriptor['targets']]
        states.append(ProjectionState(modules,ops_by_unit.get(descriptor['unit'],{}),descriptor['layout']['slice_start']))
    for block in range(19):
        image,q_in,q_out,text,t_in,t_out=states[block*6:block*6+6]
        layer=clone.transformer_blocks[block]
        layer.attn.set_processor(processor_from_original(layer.attn.processor,DoubleQKV(image,text)))
        for descriptor,state in zip(descriptors[block*6:block*6+6],states[block*6:block*6+6]):
            if descriptor['module'].endswith('QKV'):continue
            parent,leaf=descriptor['targets'][0].rsplit('.',1);setattr(clone.get_submodule(parent),leaf,OrdinaryProxy(state))
    for block in range(38):
        first=114+block*2;layer=clone.single_transformer_blocks[block]
        clone.single_transformer_blocks[block]=SingleBlockProxy(layer,states[first],states[first+1],processor_from_original(layer.attn.processor))
    return clone,states,proof

class FluxModel(Runtime):
    spec_name='flux-dev'
    parameter_transport='H2D'
    def __init__(self,config,output):
        self.start(config,output);self.units=read(config['units'])['units']
        ops=load_flux_banks(self)
        if config.get('batch_parameters',False):coalesce_parameters(self,ops)
        before=torch.cuda.memory_allocated()
        self.patched,self.states,self.shared=patch_shared_graph(self.native,self.units,ops)
        if torch.cuda.memory_allocated()!=before:raise RuntimeError('module skeleton copied model tensors')
        for uid,slot in self.bank_slots.items():self.states[uid].bank_slot=slot
        self.finish()


    def calibration_teacher(self,uid,x):
        state=self.states[uid]
        if state.protected:
            m=state.members[0];return torch.nn.functional.linear(x,m.weight[:,state.protected:],m.bias)
        return torch.cat([m(x) for m in state.members],dim=-1)
    def calibration_rows(self,uid,s):
        branch=self.units[uid]['module'].split('.')[0]
        if branch=='image':valid=torch.arange(s.shape[0],device=s.device)
        else:
            mask=self.text_mask.bool().to(s.device).reshape(-1)
            if branch=='single':mask=torch.cat((mask,torch.ones((self.adapter.spec.resolution//16)**2,device=s.device,dtype=torch.bool)))
            if mask.numel()!=s.shape[0]:raise ValueError('actual valid-row mask differs from native input')
            valid=torch.nonzero(mask,as_tuple=False).flatten()
        if len(valid)>128:valid=valid[torch.linspace(0,len(valid)-1,128,device=s.device).round().long()]
        return valid
