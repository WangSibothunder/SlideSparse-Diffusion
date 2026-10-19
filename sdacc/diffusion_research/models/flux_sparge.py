"""FLUX bridge: retain the installed processor, replace only joint attention.

SpargeAttn ae5b629e native SM80 kernel and original preparation operations.
Preparation uses explicit strides and a reduced pairwise-similarity identity.
"""
import ast
import copy
import inspect
import textwrap
import torch
import os
from sdacc_sparge_sm80_current_stream import attention as _native_attention
from .flux_sparge_prepare import prepare as raw_prepare
prepare = torch.compile(raw_prepare, fullgraph=True, dynamic=False)

RECEIPT = {}
_BACKEND = None


def _complete(q, prepared):
    qi, ki, vh, lut, valid, pv, qs, ks = prepared
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _native_attention(qi, ki, vh, out.transpose(1, 2), lut, valid, pv, qs, ks,
                      1, False, 1, q.shape[-1]**-.5, 0)
    return out


def official(q, k, v, topk):
    # Pinned preparation and attention math; native launch honors current stream.
    return _complete(q, prepare(q, k, v, topk))


def _cost(fn, args):
    for _ in range(4): fn(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): out = fn(*args)
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(8)]
    for a, b in pairs:
        a.record()
        for _ in range(16): graph.replay()
        b.record()
    torch.cuda.synchronize()
    values = [a.elapsed_time(b)/16 for a, b in pairs]
    del graph, out
    return values


def _choose(q, k, v, topk):
    global _BACKEND
    if torch.cuda.get_device_capability(q.device) != (8, 0):
        raise ValueError('pinned native SM80 backend required')
    _BACKEND = official
    RECEIPT.update(shape=list(q.shape), strides=list(q.stride()), dtype=str(q.dtype),
        actual_first_model_QKV=True, native_kernel='qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold',
        topk=topk, input_padding_bytes=0, head_dimension=128,
        source_commit='ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a',
        native_stream='PyTorch current CUDA stream',
        native_binary_SHA=os.environ['SDACC_SPARGE_NATIVE_SHA'],
        native_launch_repair='explicit current stream; same upstream SM80 kernel math')
    RECEIPT.update(selected_backend=_BACKEND.__name__, preparation='stride-explicit reduced similarity',
        output_layout='NHD direct native write', no_runtime_backend_search=True,
        no_extra_model_forward=True, full_model_speed_not_inferred=True)
    print('SPARGE_BACKEND', RECEIPT, flush=True)


@torch.library.custom_op('sdacc_flux_sparge::attention', mutates_args=())
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, topk: float) -> torch.Tensor:
    if q.shape[-1] != 128 or q.dtype != torch.bfloat16 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError('FLUX BF16 joint attention with head128 required')
    if _BACKEND is None: _choose(q, k, v, topk)
    return _BACKEND(q, k, v, topk)


@attention.register_fake
def fake_attention(q, k, v, topk):
    return torch.empty_like(q, memory_format=torch.contiguous_format)


def dispatch(q, k, v, topk, attn_mask=None, backend=None, parallel_config=None):
    if attn_mask is not None or backend is not None or parallel_config is not None:
        raise ValueError('unmasked, nonparallel FLUX attention only')
    return attention(q, k, v, topk)


def processor(original):
    function = type(original).__call__
    identity = getattr(original, 'source_identity', None)
    source = identity['transformed_source'] if identity else textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'dispatch_attention_fn']
    if len(calls) != 1: raise ValueError('expected one installed FLUX attention dispatch')
    node = calls[0]
    node.func = ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_sparge_dispatch', ctx=ast.Load())
    node.args.append(ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='sparge_topk', ctx=ast.Load()))
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    exec(compile(tree, '<flux_sparge_joint_attention_bridge>', 'exec'), namespace)
    cls = type('FluxSpargeProcessor', (type(original),), {'__call__': namespace['__call__']})
    result = object.__new__(cls)
    result.__dict__.update(copy.copy(original.__dict__))
    result._sparge_dispatch = dispatch
    result.sparge_topk = .5
    return result


def install(model):
    """Clone Python module nodes only. All weights/banks remain shared."""
    from .flux import shared_module_graph, verify_shared_tensors
    from .flux_layers import SingleBlockProxy
    clone = shared_module_graph(model)
    verify_shared_tensors(model, clone)
    entries = []
    for block in clone.transformer_blocks:
        new = processor(block.attn.processor); block.attn.set_processor(new); entries.append(new)
    for block in clone.single_transformer_blocks:
        attn = block.original.attn if isinstance(block, SingleBlockProxy) else block.attn
        new = processor(attn.processor); attn.set_processor(new); entries.append(new)
        if isinstance(block, SingleBlockProxy):
            block.processor = processor(block.processor); entries.append(block.processor)
    if len(clone.transformer_blocks) != 19 or len(clone.single_transformer_blocks) != 38:
        raise ValueError('original FLUX block coverage required')
    return clone, entries


def select(entries, topk):
    if topk not in (.5, 1.): raise ValueError('fixed default or full-block control only')
    for obj in entries: obj.sparge_topk = topk
