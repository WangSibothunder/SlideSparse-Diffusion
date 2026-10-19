from .io import sha
"""Bind a frozen public library and prepare only the deployed row shape."""
import importlib.util
import json
from pathlib import Path
import sys




def bind(snapshot):
    snapshot=Path(snapshot)
    receipt=json.loads((snapshot/'SOURCE_BINDING.json').read_text())
    assert receipt['complete']
    for name,digest in receipt['files'].items():assert sha(snapshot/name)==digest,name
    name='sdacc_research_library'
    if name in sys.modules:
        module=sys.modules[name]
        assert Path(module.__file__).parent==snapshot.resolve()
        return module
    spec=importlib.util.spec_from_file_location(name,snapshot/'__init__.py',submodule_search_locations=[str(snapshot)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    import torch
    assert not torch.cuda.is_initialized(),'bind the ABI before allocating CUDA tensors'
    module.load_native(snapshot/'native_sm80')
    assert not torch.cuda.is_initialized()
    return module


def prepare_selected(library, projection, rows, configured):
    """Same library forward; allocate one selected shape's graph, offline.

PackedLinear.configure() also prepares other catalogued M values. A model
owns many actual banks, so duplicating irrelevant graph scratch is costly.
All operations here precede compilation/timing. No packing is performed.
"""
    import torch
    config,frontend=projection.configuration_for(rows)
    key=(projection.mode,projection.k,projection.n,config)
    if key not in configured:
        torch.ops.sdacc_dualscale_sm80.configure(projection.packed,config,False)
        configured.add(key)
    if frontend.startswith(('graph','direct2_')):
        if rows not in projection._graphs:
            namespace=torch.ops.sdacc_prepared_projection_sm80
            function=(namespace.prepare_direct2 if frontend.startswith('direct2_') else
                      namespace.prepare_singlepatch if frontend.startswith('graph1_') else namespace.prepare)
            projection._graphs[rows]=function(projection.packed,projection.metadata,projection.scales,
              projection.bias,projection.k,rows,projection.n,projection.mode,config,frontend.endswith('packed16'))
    return config,frontend


def profile_for(library, mode, descriptor, recommended=False):
    import importlib
    if recommended:
        rows=read_compiler_profiles(library)
        match=next((r for r in rows if (r['mode'],r['m'],r['k'],r['n'])==(mode,descriptor['m'],descriptor['k'],descriptor['n'])),None)
        if match:return match['config'],match['frontend']
    profile=importlib.import_module(library.__name__+'.profile')
    return profile.execution_profile(mode,descriptor['m'],descriptor['k'],descriptor['n'])


def read_compiler_profiles(library):
    return json.loads((Path(library.__file__).parent/'compiler_profiles.json').read_text())['entries']

