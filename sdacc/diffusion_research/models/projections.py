"""Shared original-weight projection states and tensor-sharing module graphs."""
import copy
import torch
from torch.nn import functional as F

def shared_module_graph(module,memo=None):
    """Clone Python module nodes only, preserving Parameter and buffer objects."""
    if memo is None:memo={}
    if id(module) in memo:return memo[id(module)]
    clone=copy.copy(module);memo[id(module)]=clone
    clone._parameters=dict(module._parameters);clone._buffers=dict(module._buffers)
    clone._modules={name:None if child is None else shared_module_graph(child,memo) for name,child in module._modules.items()}
    clone._non_persistent_buffers_set=set(module._non_persistent_buffers_set)
    for name in ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_state_dict_hooks','_load_state_dict_pre_hooks'):
        if hasattr(module,name):setattr(clone,name,copy.copy(getattr(module,name)))
    return clone

def verify_shared_tensors(original,clone):
    left={**dict(original.named_parameters()),**dict(original.named_buffers())}
    right={**dict(clone.named_parameters()),**dict(clone.named_buffers())}
    if left.keys()!=right.keys() or any(left[k] is not right[k] for k in left):raise ValueError('BF16 parameters or buffers were copied')
    return dict(complete=True,tensor_objects=len(left),all_objects_shared=True,
                unique_retained_bytes=sum(t.numel()*t.element_size() for t in {id(v):v for v in left.values()}.values()),
                extra_full_model_tensor_bytes=0)

class ProjectionState:
    def __init__(self, members, packed_by_mode, protected_prefix=0):
        self.members=tuple(members);self.ops=dict(packed_by_mode);self.active=0
        self.protected=int(protected_prefix)
        if self.protected and len(self.members)!=1:raise ValueError('one original mixed output member required')
        self.widths=tuple(m.out_features for m in members)

    def select(self,mode):
        if type(mode)is not int or mode not in (0,4,6,8) or (mode and mode not in self.ops):
            raise ValueError('missing frozen packed execution mode')
        self.active=mode

    def members_output(self,x):
        if self.active==0:return tuple(module(x) for module in self.members)
        if self.protected:raise ValueError('protected mixed output is not a shared member projection')
        return self.ops[self.active](x).split(self.widths,dim=-1)

    def output(self,x):
        if self.active==0:return self.members[0](x)
        if not self.protected:return self.ops[self.active](x)
        original=self.members[0]
        # Dense remains the original full F.linear. Sparse/recovery FFN epilogue
        # includes FFN bias once and combines the protected BF16 contribution.
        attention=F.linear(x[...,:self.protected],original.weight[:,:self.protected],None)
        return self.ops[self.active](x[...,self.protected:],output_addend=attention)

    def teacher(self,x):
        if not self.protected:return torch.cat([module(x) for module in self.members],dim=-1)
        original=self.members[0]
        return F.linear(x[...,self.protected:],original.weight[:,self.protected:],original.bias)


class OrdinaryProxy(torch.nn.Module):
    def __init__(self,state):
        super().__init__();self.state=state;self.original=state.members[0]
        self.in_features=self.original.in_features;self.out_features=self.original.out_features
    def forward(self,x):return self.state.output(x)


