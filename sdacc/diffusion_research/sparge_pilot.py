"""Fixed FLUX Sparse + native SpargeAttn pilot, original banks and coefficients."""
import argparse
import os
import time
from pathlib import Path
from .io import read, write, sha, save_tensor


def execute(config, output):
    import numpy as np
    import torch
    from .registry import build
    from .runner import policy_book
    from .affine import AffinePolicy
    from .runtime import Timer, CapturedCall
    from .resources import Isolation, Memory
    from .models import flux_sparge as sparge

    os.environ['SDACC_PHYSICAL_GPU'] = os.environ['SIGMA_ORTHOGONAL_GPU']
    output = Path(output)
    if (output/'formal_complete.json').exists(): raise FileExistsError('completed pilot cannot rerun')
    output.mkdir(parents=True, exist_ok=True)
    def status(phase, **fields):
        write(output/'status.json', dict(pid=os.getpid(), phase=phase, timestamp=time.time(), **fields))
    status('loading_existing_FLUX')
    model = build('flux', config, output/'runtime')
    recoveries = {}
    rates = (15, 25, 35, 45)
    rows, windows, memory_rows, latent_rows = [], [], [], []
    extra_timers = []
    try:
        book = policy_book(config)
        definitions = {}
        for rate in rates:
            name = f'priority_s{rate}'
            path = Path(config['coefficients_directory'])/name/'coefficients.pt'
            recoveries[rate] = AffinePolicy(model, book[name], path)
            definitions[str(rate)] = dict(policy=book[name], coefficient_source=str(path), coefficient_SHA=sha(path))
        dense_sparge, dense_entries = sparge.install(model.native)
        sparse_sparge, sparse_entries = sparge.install(model.patched)
        # All four caller families receive equal fullgraph compile + whole-model graph replay.
        pool = torch.cuda.graph_pool_handle()
        sparse_call = model.pt.call.call if isinstance(model.pt.call, CapturedCall) else model.pt.call
        model.pt.call = CapturedCall(sparse_call, lambda:tuple(model.variant['policy'][model.pt.index]), lambda:model.pt.enabled, pool=pool)
        model.gt.call = CapturedCall(model.nt.call, lambda:None, lambda:model.gt.enabled, pool=pool)
        options = config.get('compile_options', {})
        dt = Timer(dense_sparge, None, model.steps)
        st = Timer(sparse_sparge, None, model.steps, model.select)
        dt.call = CapturedCall(torch.compile(dense_sparge, fullgraph=True, dynamic=False, options=options),
            lambda:dense_entries[0].sparge_topk, lambda:dt.enabled, pool=pool)
        st.call = CapturedCall(torch.compile(sparse_sparge, fullgraph=True, dynamic=False, options=options),
            lambda:(sparse_entries[0].sparge_topk, tuple(model.variant['policy'][st.index])), lambda:st.enabled, pool=pool)
        extra_timers = [dt, st]
        names = ['dense_default', 'dense_graph', 'sparge_half', 'sparge_fullmask']
        for rate in rates: names += [f'sparse_s{rate}', f'sparge_s{rate}']
        binding = dict(model='FLUX.1-dev', spec=model.adapter.receipt, PID=os.getpid(),
            config_SHA=sha(config['config_path']), definitions=definitions,
            root_manifest_SHA=sha(Path(__file__).resolve().parents[3]/'SOURCE_MANIFEST.json'),
            upstream_Sparge_commit='ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a',
            attention='original image+text joint topology; QKnorm/RoPE/order/output unchanged',
            topk=.5, fullmask_control=1., all_four_families_compiled_and_graph_replayed=True,
            calibration_reused=True, coefficients_not_refit_for_changed_attention=True,
            no_cache_method=True, scientific_quality_complete=False)
        write(output/'frozen_inputs.json', binding)

        @torch.inference_mode()
        def generate(condition, name, timing=False):
            rate = int(name.rsplit('_s', 1)[1]) if '_s' in name else None
            recovery = recoveries.get(rate)
            model.selector = recovery
            model.variant = recovery.variant if recovery else None
            if recovery: recovery.reset()
            is_sparge = name.startswith('sparge')
            topk = 1. if name == 'sparge_fullmask' else .5
            if is_sparge: sparge.select(sparse_entries if rate else dense_entries, topk)
            timer = st if is_sparge and rate else dt if is_sparge else model.pt if rate else model.gt if name=='dense_graph' else model.nt
            timer.index = 0; timer.enabled = timing
            data, model.text_mask = model.conditioning(condition); model.mask = model.text_mask
            model.transfer_records = []
            model.adapter.pipe.transformer = timer
            torch.cuda.synchronize(); start = time.perf_counter()
            z, _ = model.adapter.generate([condition], data, return_cpu=True)
            torch.cuda.synchronize(); wall = (time.perf_counter()-start)*1000
            if timer.index != model.steps or not torch.isfinite(z).all(): raise RuntimeError('incomplete/nonfinite trajectory')
            records = list(recovery.records) if recovery else []
            return z, dict(pid=os.getpid(), condition=condition['id'], method=name,
                denoiser_ms=sum(a.elapsed_time(b) for a,b in timer.events) if timing else None,
                trajectory_wall_ms=wall, affine_copy_bytes=sum(v['bytes'] for v in records),
                affine_copies=sum(v.get('copies',1) for v in records), bank_transfers=list(model.transfer_records))

        memory = Memory(model)
        for ci, condition in enumerate(model.pilot):
            attempt = 0
            while True:
                model.physical.wait_clean(); reference = {}; records = []
                with Isolation(model.physical) as isolation:
                    memory.begin()
                    for name in names:
                        status('warm4', condition=condition['id'], method=name, attempt=attempt)
                        for warm in range(4):
                            z, _ = generate(condition, name)
                            reference[name] = z
                            if sparge.RECEIPT: write(output/'attention_backend.json', sparge.RECEIPT)
                        print('WARM_COMPLETE', ci+1, name, flush=True)
                    counters = dict(torch._dynamo.utils.counters['stats'])
                    status('paired8', condition=condition['id'], attempt=attempt)
                    for repeat in range(8):
                        for name in names if repeat%2==0 else names[::-1]:
                            z, row = generate(condition, name, timing=True)
                            error = float((z.float()-reference[name].float()).norm()/reference[name].float().norm().clamp_min(1e-12))
                            if error>.01: raise RuntimeError('repeat/reload differs from warm output')
                            reference[name] = z; records.append(dict(row, repeat=repeat))
                        if dict(torch._dynamo.utils.counters['stats']) != counters: raise RuntimeError('timed compilation')
                        status('paired8', condition=condition['id'], repeat=repeat+1, attempt=attempt)
                    peak = memory.finish()
                receipt = dict(condition=condition['id'], attempt=attempt, foreign=sorted(isolation.foreign), error=isolation.error)
                if isolation.foreign or isolation.error:
                    write(output/'contaminated'/f'{condition["id"]}.{attempt}.json', dict(receipt, rows=records))
                    attempt += 1; continue
                rows += records; windows.append(receipt); memory_rows.append(peak)
                save_tensor(output/'pilot_latents'/(condition['id']+'.pt'), dict(condition=condition, pid=os.getpid(), latents=reference))
                dense = reference['dense_default'].float()
                latent_rows += [dict(condition=condition['id'], method=n,
                    latent_MSE=float((z.float()-dense).square().mean())) for n,z in reference.items()]
                write(output/'formal.partial.json', dict(pid=os.getpid(), rows=rows, windows=windows, latent_diagnostics=latent_rows))
                print('CONDITION_COMPLETE', ci+1, condition['id'], len(rows), flush=True)
                break
        rng = np.random.default_rng(20261017); samples = rng.integers(0,32,size=(4000,32)); summary = {}
        for field in ['denoiser_ms','trajectory_wall_ms']:
            values = {n:np.array([r[field] for c in model.pilot for rep in range(8) for r in rows
                if r['method']==n and r['condition']==c['id'] and r['repeat']==rep]) for n in names}
            methods = {n:dict(median_ms=float(np.median(v)), speedup_vs_default_Dense=float(np.median(values['dense_default'])/np.median(v))) for n,v in values.items()}
            gains = {}
            for rate in rates:
                n = f'sparge_s{rate}'
                gains[str(rate)] = {base:dict(speedup=float(np.median(values[base])/np.median(values[n])),
                    delta_CI95_ms=np.quantile(np.median((values[n]-values[base])[samples],axis=1),[.025,.975]).tolist())
                    for base in [f'sparse_s{rate}','sparge_half','dense_graph']}
            summary[field] = dict(methods=methods, additional_gain=gains)
        if len(rows)!=len(names)*32: raise RuntimeError('incomplete speed matrix')
        graphs = {name:dict(captures=len(timer.call.graphs),max_nrmse=timer.call.max_nrmse)
            for name,timer in [('dense',model.gt),('sparse',model.pt),('sparge',dt),('combined',st)]}
        write(output/'formal_complete.json', dict(complete=True, pid=os.getpid(), binding=binding, methods=names,
            rows=rows, windows=windows, summary=summary, graphs=graphs, memory=memory_rows,
            latent_diagnostics=latent_rows, attention_backend=sparge.RECEIPT, warmups=4, repeats=8,
            scientific_quality_complete=False, timed_compile=False,
            wall_scope='cached-text pipeline trajectory including scheduler/final CPU latent; excludes text encoder/VAE'))
        status('pilot_complete')
    finally:
        for timer in extra_timers: timer.call.close()
        for recovery in recoveries.values(): recovery.close()
        model.selector=None; model.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();from .config import load
    execute(load(a.config), a.output)


if __name__=='__main__': main()
