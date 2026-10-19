"""Pinned PixArt DuCa math, resident feature buffers and complete CUDA replay.

Port of Shenyi-Z/DuCa 0a7151b7: full / ToCa / aggressive; attention
entropy, paired CFG selection, 4x4 local bonus and original ToCa ratios.
CEM chooses full steps. This is NOT whole-prediction hold.
"""
import copy
import ctypes
import math
import time
import torch
from torch.nn import functional as F
from ..runtime import CapturedCall


def fresh_count(step, layer, module, tokens, steps=20):
    ratio = .75*(1.9-1.8*step/steps)*(1.6-1.2*layer/27)
    ratio *= .6 if module == 'cross' else 1.6
    # Match upstream torch.tensor(float)->FP32->multiply->int without a GPU sync.
    ratio = max(0.,min(1.,ctypes.c_float(ratio).value))
    return int(ctypes.c_float(ratio*tokens).value)


def local_bonus(score):
    b, n = score.shape
    side = math.isqrt(n)
    blocks = score.reshape(b, side//4, 4, side//4, 4).permute(0,1,3,2,4).contiguous().reshape(b,-1,16)
    maximum, index = blocks.max(-1, keepdim=True)
    blocks = blocks + torch.zeros_like(blocks).scatter(-1,index,1)*maximum*.4
    return blocks.reshape(b,side//4,side//4,4,4).permute(0,1,3,2,4).contiguous().reshape(b,n)


def cross_attention(attn, x, encoder, mask):
    """One Q/K/V projection and one probability matrix for output AND score."""
    if attn.norm_cross:
        encoder = attn.norm_encoder_hidden_states(encoder)
    q = attn.to_q(x)
    joined = getattr(attn, 'to_kv', None)
    if joined is not None and joined.state.sparse_on:
        k,v = joined(encoder).chunk(2,-1)
    else:
        k,v = attn.to_k(encoder),attn.to_v(encoder)
    b,n,width = q.shape
    h,d = attn.heads,width//attn.heads
    q = q.reshape(b,n,h,d).transpose(1,2)
    k = k.reshape(b,-1,h,d).transpose(1,2)
    v = v.reshape(b,-1,h,d).transpose(1,2)
    if attn.norm_q is not None: q = attn.norm_q(q)
    if attn.norm_k is not None: k = attn.norm_k(k)
    logits = (q*(d**-.5))@k.transpose(-1,-2)
    if mask is not None:
        mask = attn.prepare_attention_mask(mask,k.shape[-2],b).reshape(b,h,-1,k.shape[-2])
        logits = logits.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else logits+mask.to(logits.dtype)
    probability = logits.softmax(-1)
    output = (probability@v).transpose(1,2).reshape(b,n,width)
    output = attn.to_out[1](attn.to_out[0](output))
    if attn.residual_connection: output = output+x
    return output/attn.rescale_output_factor,probability.mean(1)


class Features:
    def __init__(self, model, refresh):
        self.refresh = frozenset(refresh)
        self.steps = 20
        self.step = 0
        self.producer = 0
        self.enabled = True
        self.kind = 'full'
        self.layers = []
        dtype = next(model.parameters()).dtype
        width = model.config.num_attention_heads*model.config.attention_head_dim
        n = (model.config.sample_size//model.config.patch_size)**2
        length = model.config.caption_channels  # dimension checked from actual encoder below
        del length
        for _ in model.transformer_blocks:
            self.layers.append(dict(
                attn=torch.zeros((2,n,width),device='cuda',dtype=dtype),
                cross=torch.zeros((2,n,width),device='cuda',dtype=dtype),
                ff=torch.zeros((2,n,width),device='cuda',dtype=dtype),
                cmap=torch.zeros((2,n,120),device='cuda',dtype=dtype),
                cross_age=torch.zeros((2,n),device='cuda',dtype=torch.int32),
                ff_age=torch.zeros((2,n),device='cuda',dtype=torch.int32)))
        self.hidden = torch.zeros((2,n,width),device='cuda',dtype=dtype)

    def select(self,step):
        self.step = step
        if step in self.refresh: self.producer = step
        age = step-self.producer
        self.kind = 'full' if age==0 else 'partial' if age%2 else 'aggressive'

    def tensors(self):
        return [v for layer in self.layers for v in layer.values()]+[self.hidden]


class Block(torch.nn.Module):
    def __init__(self,original,features,layer):
        super().__init__()
        self.original,self.features,self.layer = original,features,layer
        if original.norm_type!='ada_norm_single' or original.only_cross_attention or original.pos_embed is not None or original._chunk_size is not None:
            raise ValueError('unsupported PixArt DuCa block')

    def forward(self,hidden_states,attention_mask=None,encoder_hidden_states=None,
                encoder_attention_mask=None,timestep=None,cross_attention_kwargs=None,
                class_labels=None,added_cond_kwargs=None):
        o,c,layer = self.original,self.features,self.layer
        if c.kind=='aggressive':
            return c.hidden if layer==27 else hidden_states
        if attention_mask is not None or cross_attention_kwargs:
            raise ValueError('unexpected self mask or attention customization')
        cache = c.layers[layer]
        b = hidden_states.shape[0]
        shift,scale,gate,shift_ff,scale_ff,gate_ff = (o.scale_shift_table[None]+timestep.reshape(b,6,-1)).chunk(6,1)
        if c.kind=='full':
            sa = o.attn1(o.norm1(hidden_states)*(1+scale)+shift)
            cache['attn'].copy_(sa)
            hidden_states = gate*sa+hidden_states
            cross,cmap = cross_attention(o.attn2,hidden_states,encoder_hidden_states,encoder_attention_mask)
            cache['cross'].copy_(cross);cache['cmap'].copy_(cmap)
            hidden_states = cross+hidden_states
            ff = o.ff(o.norm2(hidden_states)*(1+scale_ff)+shift_ff)
            cache['ff'].copy_(ff)
            cache['cross_age'].zero_();cache['ff_age'].zero_()
            hidden_states = gate_ff*ff+hidden_states
        else:
            hidden_states = gate*cache['attn']+hidden_states
            for name in ('cross','ff'):
                n = fresh_count(c.step,layer,name,hidden_states.shape[1])
                # Mean conditional/unconditional map, exactly as DuCa scores.py.
                cmap = (cache['cmap'][:1]+cache['cmap'][1:])*.5
                score = F.normalize(1-(cmap*(cmap+1e-7).log()).sum(-1),p=2,dim=1).repeat(2,1)
                score = local_bonus(score+.25*cache[name+'_age'].float()/3)
                indices = score.argsort(-1,descending=True)[:,:n]
                cache[name+'_age'].add_(1).scatter_(1,indices,torch.zeros_like(indices,dtype=torch.int32))
                if n:
                    expanded = indices.unsqueeze(-1).expand(-1,-1,hidden_states.shape[-1])
                    x = hidden_states.gather(1,expanded)
                    if name=='cross':
                        new,cmap = cross_attention(o.attn2,x,encoder_hidden_states,encoder_attention_mask)
                        cache['cmap'].scatter_(1,indices.unsqueeze(-1).expand(-1,-1,cmap.shape[-1]),cmap)
                    else:
                        new = o.ff(o.norm2(x)*(1+scale_ff)+shift_ff)
                    cache[name].scatter_(1,expanded,new)
                hidden_states = cache[name]+hidden_states if name=='cross' else gate_ff*cache[name]+hidden_states
        if layer==27: c.hidden.copy_(hidden_states)
        return hidden_states


def install(model,features):
    """New routing shell shares existing weights; never duplicate a model bank."""
    shell = copy.copy(model)
    shell._modules = model._modules.copy()
    shell.transformer_blocks = torch.nn.ModuleList([Block(b,features,i) for i,b in enumerate(model.transformer_blocks)])
    return shell


class RowProjection(torch.nn.Module):
    """Keep accepted full-M caller; partial rows use generic public cfg5/native.

    No weight repacking, autotuning or full-row padding. Same scale/bias views.
    """
    def __init__(self,op,full_rows,library):
        super().__init__()
        self.full,self.rows = op,full_rows
        self.partial = library.PackedLinear(op.packed,op.metadata,op.scales,op.bias,
            op.k,op.n,op.mode,config=5,frontend='native',identity=op.identity)
        torch.ops.sdacc_dualscale_sm80.configure(op.packed,5,False)

    def forward(self,x):
        return (self.full if x.numel()//x.shape[-1]==self.rows else self.partial)(x)

    def close_graphs(self):
        self.full.close_graphs();self.partial.close_graphs()


class Replay(CapturedCall):
    """Static conditioning copied once per condition/route, dynamic inputs always.

    Stateful DuCa captures restore feature state before replay: capture itself
    must not age or update the cache an extra time. No timestep recompilation.
    """
    STATIC = frozenset(('encoder_hidden_states','encoder_attention_mask','attention_mask','added_cond_kwargs'))

    def __init__(self,call,key,timing,features=None,pool=None):
        super().__init__(call,key,timing,pool=pool)
        self.features = features
        self.condition = None
        self.condition_by_key = {}
        self.copy_bytes = self.copy_calls = 0
        self.select_cpu_ms = 0.
        self.capture_seconds = 0.

    def _copy(self,target,source):
        if isinstance(target,torch.Tensor):
            self.copy_bytes += target.numel()*target.element_size();self.copy_calls += 1
        elif isinstance(target,dict):
            for k,v in target.items():self._copy(v,source[k])
            return
        elif isinstance(target,(tuple,list)):
            for a,b in zip(target,source):self._copy(a,b)
            return
        super().copy(target,source)

    def forward(self,*args,**kwargs):
        key = self.key()
        if key not in self.graphs:
            if self.timing(): raise RuntimeError('new route during formal timing')
            start = time.perf_counter()
            inputs = self.clone((args,kwargs))
            state = self.features.tensors() if self.features is not None else []
            saved = [x.clone() for x in state]
            def restore():
                for x,y in zip(state,saved):x.copy_(y)
            reference = self.clone(self.call(*inputs[0],**inputs[1]))
            restore();torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool): output = self.call(*inputs[0],**inputs[1])
            restore();graph.replay();torch.cuda.synchronize()
            for actual,expected in zip(self.tensors(output),self.tensors(reference)):
                error = float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12))
                self.max_nrmse = max(error,self.max_nrmse)
                if not torch.isfinite(actual).all() or error>.01:raise RuntimeError('capture differs from its actual caller')
            self.graphs[key] = inputs,graph,output
            self.condition_by_key[key] = self.condition
            self.capture_seconds += time.perf_counter()-start
        else:
            inputs,graph,output = self.graphs[key]
            self._copy(inputs[0],args)
            for name,target in inputs[1].items():
                if name not in self.STATIC or self.condition_by_key[key]!=self.condition:
                    self._copy(target,kwargs[name])
            self.condition_by_key[key] = self.condition
            graph.replay()
        return output
