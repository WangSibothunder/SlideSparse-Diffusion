"""Single-idle-GPU 64-image quality backfill. No calibration or speed repetition."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get('SDACC_ORTHOGONAL_ROOT', 'data/orthogonal_plugins')).expanduser().resolve()
OUT = ROOT/'quality/eval64_20261019'
CODE = Path(__file__).resolve().parents[2]


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp'); tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n'); tmp.replace(path)


def admit(folder):
    sys.path[:0] = [str(ROOT/'code/v079/src'), str(ROOT/'baseline/source_snapshot/sdacc/vendor_nvml')]
    from sigma_orthogonal.gpu import admit as choose
    choose(folder/'resource', shared=False)
    os.environ['SDACC_PHYSICAL_GPU'] = os.environ['SIGMA_ORTHOGONAL_GPU']


def wait_for_exclusive_card(folder):
    from sigma_orthogonal.gpu import snapshot, healthy
    while True:
        state=snapshot(int(os.environ['SIGMA_ORTHOGONAL_GPU']))
        foreign=sorted(set(state['pids'])-{os.getpid()})
        if not foreign:
            if not healthy(state): raise RuntimeError('own runtime lacks the required4GiB/health margin')
            return
        # Preserve valid quality output; pause at the next condition boundary.
        write(folder/'waiting_foreign.json',dict(gpu=state['index'],foreign=foreign,
            timestamp=time.time(),own_PID=os.getpid(),action='wait_without_migration_or_signals'))
        time.sleep(15)


def generate(worker):
    folder = Path(worker['output']); plan_sha = sha(OUT/'PLAN.json')
    if (folder/'generation_complete.json').exists():
        assert read(folder/'generation_complete.json')['plan_SHA'] == plan_sha
        return
    admit(folder)
    sys.path[:0] = [str(ROOT/'plugins/sparge_v001'), str(ROOT/'runtime/einops_v005'),
                    str(ROOT/'sparge/flux_stream_repair/build')]
    os.environ['SDACC_SPARGE_NATIVE_SHA'] = '611fcf1a775b7558a9424f6d2e4a119274713b67880053e74d718bf79844de4b'
    import torch
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        from .factories import factory
    except ImportError:
        from factories import factory
    make = factory(worker, ROOT)
    conditions = worker['conditions']; methods = worker['methods']
    assert len(conditions) == 64 and len({c['id'] for c in conditions}) == 64
    sources = []
    def committed(method, c):
        path = folder/'latents'/method/(c['id']+'.pt'); meta = path.with_suffix('.json')
        if not meta.exists(): return None
        row = read(meta); assert row['plan_SHA'] == plan_sha and path.exists()
        return row
    missing = [(m,c) for c in conditions for m in methods if committed(m,c) is None]
    if missing:
        config = dict(worker['config'], config_path=worker['config_path'])
        args = (folder,) if worker['worker']=='sigma_plain' else (
            config,folder,worker['schedule_file']) if worker['worker']=='alpha_duca' else (config,folder)
        with make(*args) as (model, operation), torch.inference_mode():
            if worker['worker'] == 'flux_sparge':
                from sdacc.diffusion_research.models import flux_sparge
                old = read(ROOT/'sparge/flux_current_stream/attention_backend.json')
                assert old['selected_backend'] == 'official'
                # Reuse the accepted backend choice; never repeat first-QKV timing/search.
                flux_sparge._BACKEND = flux_sparge.official
            for ci,c in enumerate(conditions):
                todo = [m for m in methods if committed(m,c) is None]
                if not todo: continue
                wait_for_exclusive_card(folder)
                for method in todo:
                    write(folder/'status.json',dict(stage='generating64',condition=c['id'],condition_index=ci+1,
                        method=method,pid=os.getpid(),timestamp=time.time()))
                    latent,_ = operation(c,method) if worker['worker']=='sigma_plain' else operation(c,method,False)
                    assert torch.isfinite(latent).all()
                    path=folder/'latents'/method/(c['id']+'.pt'); path.parent.mkdir(parents=True,exist_ok=True)
                    tmp=path.with_suffix('.tmp');torch.save(dict(latent=latent.detach().cpu(),condition=c,
                        method=method,pid=os.getpid()),tmp);tmp.replace(path)
                    row=dict(method=method,condition=c['id'],path=str(path),sha=sha(path),generation_pid=os.getpid(),
                        plan_SHA=plan_sha,frozen_runtime_source=worker['source'],runtime_source_SHA=worker['source_manifest_SHA'])
                    write(path.with_suffix('.json'),row)
                write(folder/'progress.json',dict(conditions_committed=ci+1,total_conditions=64,
                    methods=methods,pid=os.getpid(),timestamp=time.time()))
    for c in conditions:
        for m in methods:sources.append(committed(m,c))
    assert all(sources)
    write(folder/'generation_complete.json',dict(complete=True,pid=os.getpid(),methods=methods,conditions=conditions,
        latent_sources=sources,plan_SHA=plan_sha,evaluation_used_for_fit_or_selection=False,
        repeated_speed=False,extra_full_trajectory_warmups=0,original_schedule_reused=True))


def decode(worker):
    folder=Path(worker['output'])
    if (folder/'decode_complete.json').exists(): return
    admit(folder)
    sys.path.insert(0,str(CODE.parents[1]))
    from sdacc.diffusion_research.image_metrics import decode as run
    run(worker['config'],folder,read(folder/'generation_complete.json'))


def score(worker):
    folder=Path(worker['output'])
    if (folder/'quality_complete.json').exists(): return
    admit(folder)
    sys.path.insert(0,str(CODE.parents[1]))
    from sdacc.diffusion_research.image_metrics import score as run
    config=dict(worker['config'],dense_root=str(OUT/'reuse'/worker['model']),
        dense_generation=str(OUT/'reuse'/worker['model']/'generation_complete.json'),
        dense_decode=str(OUT/'reuse'/worker['model']/'decode_complete.json'),verify_imagereward_scalar=False)
    run(CODE.parents[1],folder,read(folder/'generation_complete.json'),config)


def dense_alpha():
    folder=OUT/'reuse/pixart-alpha';g=read(folder/'generation_complete.json');d=read(folder/'decode_complete.json')
    d['generation_SHA']=sha(folder/'generation_complete.json');write(folder/'decode_complete.json',d)
    worker=dict(worker='alpha_dense_existing',model='pixart-alpha',output=str(folder),
        config=read(OUT/'PLAN.json')['workers']['alpha_plain']['config'])
    score(worker)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['generate','decode','score','dense-alpha']);p.add_argument('--worker');args=p.parse_args()
    manifest=read(Path(__file__).parent/'SOURCE_MANIFEST.json') if (Path(__file__).parent/'SOURCE_MANIFEST.json').exists() else None
    if manifest and 'bound_plan_SHA' in manifest:
        assert sha(OUT/'PLAN.json')==manifest['bound_plan_SHA']
        for source,digest in manifest.get('metric_sources',{}).items():assert sha(source)==digest
    if args.stage=='dense-alpha':dense_alpha()
    else:
        worker=read(OUT/'PLAN.json')['workers'][args.worker]
        {'generate':generate,'decode':decode,'score':score}[args.stage](worker)
