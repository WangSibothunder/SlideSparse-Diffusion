"""Optional admitted compiler recipe; no mutation of an existing projection."""
import json,math
from pathlib import Path
import torch
from .runtime import PackedLinear
from .composite import SharedProjection
_DATA=json.loads((Path(__file__).parent/'compiler_profiles.json').read_text())
_RECIPES={(v['mode'],v['m'],v['k'],v['n']):v for v in _DATA['entries']}

def compile_projection(projection,example_input):
    """Prepare and compile offline; explicit C++ wrapper recipes only when gated.

    Default forward profiles remain independently admitted for ordinary
    torch.compile callers. A C++ recipe clones only the execution module and
    shares immutable packed weights; graph scratch/output remain independent.
    """
    packed=projection.packed if isinstance(projection,SharedProjection) else projection
    recipe=None
    if isinstance(packed,PackedLinear) and example_input.is_contiguous():
        m=math.prod(example_input.shape[:-1]);recipe=_RECIPES.get((packed.mode,m,packed.k,packed.n))
    if recipe:
        op=PackedLinear(packed.packed,packed.metadata,packed.scales,packed.bias,
                        packed.k,packed.n,packed.mode,recipe['config'],
                        packed.strategy,packed.identity,recipe['frontend'])
        op.configure_for(m)
        result=SharedProjection(op,projection.widths) if isinstance(projection,SharedProjection) else op
    else:
        result=projection;result.configure()
    return torch.compile(result,fullgraph=True,dynamic=False,
                         options={'cpp_wrapper':bool(recipe),'triton.cudagraphs':False})
