"""One FLUX geometry, no model loading, calibration, quality or timing claim."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
output = Path(os.environ['SPARGE_STREAM_PROBE'])
output.mkdir(parents=True, exist_ok=True)

if len(sys.argv) == 1:
    runs = {}
    for mode in ('stock', 'fixed'):
        p = subprocess.run([sys.executable, __file__, mode], capture_output=True, text=True)
        (output/f'{mode}.log').write_text(p.stdout+'\n'+p.stderr)
        runs[mode] = dict(returncode=p.returncode, result=str(output/f'{mode}.json'))
        print(mode, p.returncode, p.stdout[-1800:], p.stderr[-1800:], flush=True)
    fixed = json.loads((output/'fixed.json').read_text()) if (output/'fixed.json').exists() else {}
    result = dict(runs=runs, passed=runs['fixed']['returncode']==0 and fixed.get('passed') is True,
        scope='synthetic QKV at first actual FLUX geometry [1,1536,24,128]; stream compatibility only',
        full_model_speed_not_measured=True, pilot_quality_not_accepted=True)
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    assert result['passed'], result
    sys.exit(0)

import torch
from spas_sage_attn import core
mode = sys.argv[1]
spec = importlib.util.spec_from_file_location('frozen_reference', root/'reference_flux_sparge.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
stock = core.qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold
if mode == 'fixed':
    build = Path(os.environ['SPARGE_STREAM_BUILD'])
    receipt = json.loads((build/'complete.json').read_text())
    binary = Path(receipt['binary'])
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == receipt['binary_sha256']
    assert receipt['source_manifest_sha256'] == hashlib.sha256((root/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()
    sys.path.insert(0, str(build))
    import sdacc_sparge_sm80_current_stream
    native = sdacc_sparge_sm80_current_stream.attention
else:
    native = stock

torch.manual_seed(241018)
q,k,v = [torch.randn(1,1536,24,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
rows = []
result = dict(mode=mode, shape=list(q.shape), synthetic_values=True, rows=rows, passed=False)

def candidate(topk):
    qi,ki,vh,lut,valid,pv,qs,ks = reference.prepare(q,k,v,topk)
    out = torch.empty_like(qi, dtype=q.dtype)
    native(qi,ki,vh,out,lut,valid,pv,qs,ks,1,False,1,128**-.5,0)
    return out.transpose(1,2).contiguous()

def error(x,y):
    return float((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-12))

try:
    for topk in (.5,1.):
        # Default-stream synchronized oracle retains the pinned upstream math.
        oracle = reference.official(q,k,v,topk)
        torch.cuda.synchronize()
        eager = candidate(topk)
        torch.cuda.synchronize()
        e = error(eager,oracle)
        assert bool(torch.isfinite(eager).all()) and e <= .01, ('eager',e)
        candidate(topk)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = candidate(topk)
        torch.cuda.synchronize()
        # A replay must overwrite poisoned output: a default-stream launch
        # accidentally executed during capture cannot satisfy this check.
        captured.fill_(float('nan'))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        g = error(captured,oracle)
        assert bool(torch.isfinite(captured).all()) and g <= .01, ('graph_replay',g)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            graph.replay()
            torch.cuda.synchronize()
        kernels = sorted({ev.name for ev in prof.events() if 'qk_int_sv_f16_block_sparse_attn_kernel' in ev.name})
        assert kernels, 'native Sparge kernel absent from graph replay'
        rows.append(dict(topk=topk, eager_nrmse=e, graph_nrmse=g,
            replay_overwrites_nan=True, native_graph_kernels=kernels))
        del graph,captured,eager,oracle
    result['passed'] = True
finally:
    (output/f'{mode}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result), flush=True)
