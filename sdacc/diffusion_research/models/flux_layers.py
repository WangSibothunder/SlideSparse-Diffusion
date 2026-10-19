"""One-model projection wrappers; protected attention is never corrected."""
import ast
import inspect
import textwrap
import types
import torch
from torch.nn import functional as F


from .projections import ProjectionState,OrdinaryProxy


def processor_from_original(original,project_qkv=None):
    """Clone the accepted processor method, altering only QKV source routing.

Rotary embeddings, Q/K normalization, attention dispatch and output projections
retain the installed Diffusers method bytecode source. No global monkeypatch.
The optional precomputed tuple shares single-block QKV+FFN Quant exactly once.
"""
    function=type(original).__call__;source=textwrap.dedent(inspect.getsource(function));tree=ast.parse(source)
    definition=tree.body[0];calls=[n for n in ast.walk(definition) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_get_qkv_projections']
    if len(calls)!=1:raise ValueError('unsupported installed FLUX attention implementation')
    definition.args.kwonlyargs.append(ast.arg(arg='_precomputed_qkv'));definition.args.kw_defaults.append(ast.Constant(None))
    class Rewrite(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='_get_qkv_projections':
                call=ast.Call(func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='_project_qkv',ctx=ast.Load()),args=node.args,keywords=node.keywords)
                return ast.IfExp(test=ast.Compare(left=ast.Name(id='_precomputed_qkv',ctx=ast.Load()),ops=[ast.IsNot()],comparators=[ast.Constant(None)]),body=ast.Name(id='_precomputed_qkv',ctx=ast.Load()),orelse=call)
            return self.generic_visit(node)
    tree=Rewrite().visit(tree);ast.fix_missing_locations(tree);namespace=dict(function.__globals__)
    exec(compile(tree,'<frozen_flux_attention_projection_bridge>','exec'),namespace)
    cls=type('FrozenFluxProjectionProcessor',(type(original),),{'__call__':namespace['__call__']})
    result=cls();result._attention_backend=original._attention_backend;result._parallel_config=original._parallel_config
    result._project_qkv=project_qkv or function.__globals__['_get_qkv_projections']
    result.source_identity=dict(original_source=source,transformed_source=ast.unparse(tree),only_QKV_routing_changed=True)
    return result


class SingleBlockProxy(torch.nn.Module):
    """Same original parameter objects; fused sparse input, protected output."""
    def __init__(self,original,input_state,output_state,processor):
        super().__init__();self.original=original;self.input_state=input_state;self.output_state=output_state;self.processor=processor
    def forward(self,hidden_states,encoder_hidden_states,temb,image_rotary_emb=None,joint_attention_kwargs=None):
        original=self.original;text_len=encoder_hidden_states.shape[1]
        combined=torch.cat([encoder_hidden_states,hidden_states],dim=1);residual=combined
        normalized,gate=original.norm(combined,emb=temb)
        if self.input_state.active==0:
            # Match original single-block member order and full Dense projection.
            mlp=original.act_mlp(original.proj_mlp(normalized));attention=original.attn(hidden_states=normalized,image_rotary_emb=image_rotary_emb,**(joint_attention_kwargs or {}))
        else:
            q,k,v,mlp=self.input_state.members_output(normalized);mlp=original.act_mlp(mlp)
            if joint_attention_kwargs:raise ValueError('unsupported extra conditioning in frozen joint-only FLUX')
            attention=self.processor(original.attn,normalized,image_rotary_emb=image_rotary_emb,_precomputed_qkv=(q,k,v,None,None,None))
        mixed=torch.cat([attention,mlp],dim=2);output=self.output_state.output(mixed)
        result=residual+gate.unsqueeze(1)*output
        if result.dtype==torch.float16:result=result.clip(-65504,65504)
        return result[:,:text_len],result[:,text_len:]

