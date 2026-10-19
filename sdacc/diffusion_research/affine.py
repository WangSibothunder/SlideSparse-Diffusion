"""Shared FP32 scale/bias fold and per-step parameter selection."""
from pathlib import Path
import torch
from .library import prepare_selected
from .affine_math import positions

class AffinePolicy:
    def __init__(self,model,policy,coefficient_file):
        self.model=model;self.variant=model.prepare_variant(policy);packet=torch.load(coefficient_file,map_location='cpu',weights_only=True)
        coefficients=packet.get('positions',packet)
        if set(coefficients)!=set(positions(policy)):raise ValueError('coefficient coverage differs from sparse policy')
        self.parameters={};self.operators={};configured=set()
        self.bulk=(model.parameter_transport=='D2D' and not model.bank_slots) or hasattr(model,'coefficient_arena')
        if self.bulk:
            # All prepared operators refer to stable views of one resident slab.
            # Select copies a complete step in one transfer, rather than issuing
            # two tiny transfers for every sparse projection.
            self.offsets={};size=0
            if hasattr(model,'coefficient_arena'):
                self.target=model.coefficient_arena;self.offsets=model.coefficient_ranges;size=self.target.numel()
            else:
                for uid in sorted({int(key.split(':')[1]) for key in coefficients}):
                    n=model.units[uid]['n'];self.offsets[uid]=(size,n);size+=2*n
                self.target=torch.empty(size,device='cuda',dtype=torch.float32)
            host=torch.zeros((len(policy),size),dtype=torch.float32)
        sources={}
        for key,value in coefficients.items():
            step,uid=map(int,key.split(':'));mode=policy[step][uid];state=model.states[uid];raw=state.ops[mode]
            if (uid,mode) not in sources:
                if uid in model.bank_slots:
                    source=model.bank_slots[uid].host[mode];sources[uid,mode]=(source['scales'],source['bias'])
                else:sources[uid,mode]=(raw.scales.cpu(),raw.bias.cpu())
            scales,bias=sources[uid,mode]
            a=torch.as_tensor(value['a'],dtype=torch.float32);b=torch.as_tensor(value['b'],dtype=torch.float32)
            sw=scales.float()*a;bv=bias.float()*a+b
            if self.bulk:
                offset,n=self.offsets[uid];host[step,offset:offset+n].copy_(sw);host[step,offset+n:offset+2*n].copy_(bv)
            if model.parameter_transport=='D2D':
                if self.bulk:
                    target_sw=self.target[offset:offset+n];target_bv=self.target[offset+n:offset+2*n]
                else:
                    sw,bv=sw.cuda(),bv.cuda();target_sw,target_bv=raw.scales.clone(),raw.bias.clone()
                if (uid,mode) not in self.operators:
                    op=model.library.PackedLinear(raw.packed,raw.metadata,target_sw,target_bv,raw.k,raw.n,mode,config=raw.config,strategy=raw.strategy,frontend=raw.frontend,identity=raw.identity)
                    prepare_selected(model.library,op,model.units[uid]['m'],configured);self.operators[uid,mode]=op
            elif not self.bulk:sw,bv=sw.pin_memory(),bv.pin_memory()
            if not self.bulk:self.parameters[step,uid]=(sw,bv)
        if self.bulk:
            self.steps=host.cuda() if model.parameter_transport=='D2D' else host.pin_memory()
            self.step_parameters=list(self.steps.unbind(0))
        self.reset()
    def reset(self):
        self.cached={};self.records=[]
        if not hasattr(self.model,'coefficient_arena'):
            for slot in self.model.bank_slots.values():slot.current_mode=None
    def select(self,step):
        model=self.model;model.select_raw(step);moved=0;copies=0
        if self.bulk:
            self.target.copy_(self.step_parameters[step],non_blocking=True)
            if not model.bank_slots:
                for uid,mode in self.variant['updates'][step]:
                    if mode:model.states[uid].active=self.operators[uid,mode]
            self.records.append(dict(step=step,projections=sum(bool(m) for m in self.variant['policy'][step]),copies=1,bytes=self.target.numel()*self.target.element_size(),direction=model.parameter_transport,scope='Timer.select'))
            return
        for uid,mode in enumerate(self.variant['policy'][step]):
            if not mode:continue
            sw,bv=self.parameters[step,uid]
            if uid in model.bank_slots:
                slot=model.bank_slots[uid];target=slot.views[mode];generation=slot.transfers
            else:
                op=self.operators[uid,mode];target=dict(scales=op.scales,bias=op.bias);model.states[uid].active=op;generation=0
            token=(mode,step,generation)
            if self.cached.get(uid)!=token:
                target['scales'].copy_(sw,non_blocking=True);target['bias'].copy_(bv,non_blocking=True);self.cached[uid]=token
                moved+=sw.numel()*sw.element_size()+bv.numel()*bv.element_size();copies+=1
        if copies:self.records.append(dict(step=step,projections=copies,bytes=moved,direction=model.parameter_transport,scope='Timer.select'))
    def close(self):
        from .runtime import CapturedCall
        if isinstance(self.model.pt.call,CapturedCall):self.model.pt.call.close(self)
        for op in self.operators.values():op.close_graphs()
