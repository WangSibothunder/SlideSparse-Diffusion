"""SD3 MMDiT: original joint attention, shared resident sparse projections."""
import ast
import copy
import inspect
import textwrap
from pathlib import Path
import torch
from ..runtime import Runtime
from ..io import read
from ..banks import load_resident_banks
from .projections import ProjectionState,OrdinaryProxy,shared_module_graph


class ResidentProjection(ProjectionState):
    def members_output(self,x):
        if isinstance(self.active,int) and self.active==0:return tuple(module(x) for module in self.members)
        op=self.ops[self.active] if isinstance(self.active,int) else self.active
        value=op(x.reshape(-1,self.members[0].in_features))
        return value.reshape(*x.shape[:-1],sum(self.widths)).split(self.widths,dim=-1)
    def output(self,x):return self.members_output(x)[0]


def joint_processor(original,image,text):
    """Replace the two QKV projection groups in the installed native method."""
    function=type(original).__call__;tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    groups={'to_q':('image',['query','key','value'],'hidden_states'),
            'add_q_proj':('text',['encoder_hidden_states_query_proj','encoder_hidden_states_key_proj','encoder_hidden_states_value_proj'],'encoder_hidden_states')}
    removed={'to_k','to_v','add_k_proj','add_v_proj'};changed=[]
    class Route(ast.NodeTransformer):
        def visit_Assign(self,node):
            call=node.value
            if not (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=='attn'):return self.generic_visit(node)
            name=call.func.attr
            if name in removed:changed.append(name);return None
            if name in groups:
                branch,targets,arg=groups[name];changed.append(name)
                return ast.Assign(targets=[ast.Tuple(elts=[ast.Name(id=v,ctx=ast.Store()) for v in targets],ctx=ast.Store())],
                    value=ast.Call(func=ast.Attribute(value=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr=branch,ctx=ast.Load()),attr='members_output',ctx=ast.Load()),args=[ast.Name(id=arg,ctx=ast.Load())],keywords=[]))
            return self.generic_visit(node)
    tree=Route().visit(tree);ast.fix_missing_locations(tree)
    if set(changed)!=set(groups)|removed or len(changed)!=6:raise ValueError('unsupported installed SD3 joint projection routing')
    namespace=dict(function.__globals__);exec(compile(tree,'<sd3_joint_projection_routing>','exec'),namespace)
    cls=type('SD3ProjectionProcessor',(type(original),),{'__call__':namespace['__call__']})
    result=cls();result.__dict__.update(copy.copy(original.__dict__));result.image=image;result.text=text
    return result


def projection_inventory(model,resolution,text_length):
    units=[]
    for block,layer in enumerate(model.transformer_blocks):
        prefix=f'transformer_blocks.{block}'
        for branch,ff,qkv in [('image','ff',['to_q','to_k','to_v']),('text','ff_context',['add_q_proj','add_k_proj','add_v_proj'])]:
            groups=[('QKV',[prefix+'.attn.'+name for name in qkv])]
            if branch=='image' or not layer.context_pre_only:
                groups += [('FFN-in',[prefix+'.'+ff+'.net.0.proj']),('FFN-out',[prefix+'.'+ff+'.net.2'])]
            for role,names in groups:
                modules=[model.get_submodule(name) for name in names]
                units.append(dict(unit=len(units),block=block,module=branch+'.'+role,branch=branch,targets=names,
                    m=2*((text_length+77) if branch=='text' else (resolution//16)**2),
                    k=modules[0].in_features,n=sum(m.out_features for m in modules)))
    return units


def install_projections(native,units,operators):
    """SD3 and SD3.5 share routing; native Q/K normalization stays in the processor."""
    patched=shared_module_graph(native);states=[];qkv={}
    for unit in units:
        modules=[patched.get_submodule(n) for n in unit['targets']]
        state=ResidentProjection(modules,operators[unit['unit']]);states.append(state)
        if unit['module'].endswith('QKV'):qkv[unit['block'],unit['branch']]=state
        else:
            parent,leaf=unit['targets'][0].rsplit('.',1)
            setattr(patched.get_submodule(parent),leaf,OrdinaryProxy(state))
    for block,layer in enumerate(patched.transformer_blocks):
        layer.attn.set_processor(joint_processor(layer.attn.processor,qkv[block,'image'],qkv[block,'text']))
    return patched,states


class SD3Model(Runtime):
    spec_name='sd3-medium'
    parameter_transport='D2D'
    inventory=staticmethod(projection_inventory)
    def __init__(self,config,output):
        self.start(config,output);self.units=read(config['units'])['execution_units']
        self.patched,self.states=install_projections(self.native,self.units,load_resident_banks(self))
        self.finish()
    def calibration_teacher(self,uid,x):return self.states[uid].teacher(x)
    def calibration_rows(self,uid,s):
        valid=torch.arange(s.shape[0],device=s.device)
        if self.units[uid]['branch']=='text':valid=valid[self.text_mask.bool().flatten()]
        return valid[torch.linspace(0,len(valid)-1,min(128,len(valid)),device=s.device).long()]


class SD35LargeModel(SD3Model):
    """38-layer MMDiT with checkpoint-native RMS Q/K normalization."""
    spec_name='sd3.5-large'
