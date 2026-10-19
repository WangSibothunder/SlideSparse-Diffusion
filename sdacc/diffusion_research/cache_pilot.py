"""Minimal shared Sparse + plain CEM speed pilot; no new calibration or quality run."""
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
    from .cache import PredictionHold, cem_schedule, warmup_prior, execution_variant
    from .resources import Isolation, Memory

    os.environ['SDACC_PHYSICAL_GPU'] = os.environ['SIGMA_ORTHOGONAL_GPU']
    output = Path(output)
    if (output/'formal_complete.json').exists():
        raise FileExistsError('completed pilot must not be replayed')
    output.mkdir(parents=True, exist_ok=True)
    def status(phase, **fields):
        write(output/'status.json', dict(phase=phase, pid=os.getpid(), timestamp=time.time(), **fields))
    status('loading_reused_model')
    model = build(config['model'], config, output/'runtime')
    recoveries, gates = {}, {}
    memory = Memory(model)
    book = policy_book(config)
    rates = (15, 25, 35, 45)
    names = ['dense_default', 'plain_cem']
    graph_control = hasattr(model, 'gt')
    if graph_control:
        names += ['dense_graph_control', 'plain_cem_graph_control']
    for rate in rates:
        names += [f'sparse_s{rate}', f'plain_cem_s{rate}']
    boundaries = {0, model.steps-1}
    definitions = {}
    original_variants = {}
    schedule = None
    rows, windows, latent_errors, memory_records = [], [], [], []
    try:
        for rate in rates:
            name = f'priority_s{rate}'
            policy = book[name]
            path = Path(config['coefficients_directory'])/name/'coefficients.pt'
            recovery = AffinePolicy(model, policy, path)
            recoveries[rate] = recovery
            original_variants[rate] = recovery.variant
            boundaries.update(t for t in range(1, model.steps) if policy[t] != policy[t-1])
            definitions[str(rate)] = dict(policy=policy, coefficients=str(path), coefficient_SHA=sha(path))
        for key, timer in [('native', model.nt), ('sparse', model.pt), *([('graph', model.gt)] if graph_control else [])]:
            gates[key] = PredictionHold(timer.call, model.steps, timer.select)
            timer.call, timer.select = gates[key], None
        binding = dict(model=config['model'], spec=model.adapter.receipt, config_SHA=sha(config['config_path']),
            definitions=definitions, boundaries=sorted(boundaries), steps=model.steps,
            cache='owned raw transformer prediction, before pipeline CFG/variance/scheduler',
            prior_scope='one existing Dense first-condition warmup output trajectory; NOT100cal or heldout quality',
            source_manifest_SHA=sha(Path(__file__).parent/'SOURCE_MANIFEST.json'),
            sparse_coefficients='unchanged original fits; no cached-trajectory fit',
            no_DuCa=True, extra_model_forwards_for_prior=0, quality_evaluation=False)
        write(output/'frozen_inputs.json', binding)

        def generate(condition, name, timing=False, observe=False):
            rate = int(name.split('_s')[-1]) if '_s' in name else None
            cached = name.startswith('plain_cem')
            graph = name.endswith('graph_control')
            recovery = recoveries.get(rate)
            model.selector = recovery
            if recovery:
                recovery.variant = execution_variant(original_variants[rate], schedule['refresh']) if cached else original_variants[rate]
                recovery.reset()
            gate = gates['sparse' if rate else 'graph' if graph else 'native']
            gate.reset(cached, schedule['refresh'] if cached else (), boundaries, observe)
            torch.cuda.synchronize()
            start = time.perf_counter()
            latent, row = model.generate(condition, recovery.variant if recovery else None, timing, dense_graph=graph)
            row['trajectory_wall_ms'] = (time.perf_counter()-start)*1000
            expected = schedule['refresh'] if cached else list(range(model.steps))
            if gate.step != model.steps or gate.executed != expected:
                raise ValueError('scheduler or refresh coverage differs')
            transfers = list(recovery.records) if recovery else []
            row.update(method=name, transformer_calls=len(gate.executed), cache_hits=model.steps-len(gate.executed),
                       executed=gate.executed.copy(), affine_copy_bytes=sum(r['bytes'] for r in transfers),
                       affine_copy_records=len(transfers))
            return latent, row

        for ci, condition in enumerate(model.pilot):
            attempt = 0
            while True:
                model.physical.wait_clean()
                records, reference = [], {}
                with Isolation(model.physical) as isolation:
                    memory.begin()
                    for name in names:
                        status('warm4', condition=condition['id'], method=name, attempt=attempt)
                        for warm in range(4):
                            observe = schedule is None and name == 'dense_default' and warm == 0
                            z, _ = generate(condition, name, observe=observe)
                            if observe:
                                prior = warmup_prior(gates['native'].predictions)
                                gates['native'].predictions.clear()
                                schedule = cem_schedule(prior, model.steps//2, model.steps, boundaries)
                                write(output/'schedule.json', dict(schedule=schedule, priors=prior,
                                    source_condition=condition, source='first required native warmup',
                                    extra_model_forwards=0, frozen_before_any_cached_pilot=True))
                                print('CEM_SCHEDULE', schedule, flush=True)
                            reference[name] = z
                    counters = dict(torch._dynamo.utils.counters['stats'])
                    status('paired8', condition=condition['id'], attempt=attempt)
                    for repeat in range(8):
                        for name in (names if repeat%2 == 0 else names[::-1]):
                            z, row = generate(condition, name, timing=True)
                            if not torch.equal(z, reference[name]):
                                error=float((z.float()-reference[name].float()).norm()/reference[name].float().norm().clamp_min(1e-12))
                                if error > .01:
                                    raise RuntimeError('same-PID warmed/repeated output changed')
                            reference[name] = z
                            records.append(dict(row, repeat=repeat))
                        if dict(torch._dynamo.utils.counters['stats']) != counters:
                            raise RuntimeError('compilation during formal timing')
                    peak = memory.finish()
                receipt = dict(condition=condition['id'], attempt=attempt, foreign=sorted(isolation.foreign), error=isolation.error)
                if isolation.foreign or isolation.error:
                    write(output/'contaminated'/f'{condition["id"]}.{attempt}.json', dict(receipt, rows=records))
                    attempt += 1
                    continue
                rows.extend(records)
                windows.append(receipt)
                memory_records.append(peak)
                save_tensor(output/'pilot_latents'/(condition['id']+'.pt'), dict(condition=condition, pid=os.getpid(), latents=reference))
                dense = reference['dense_default'].float()
                errors = {n:float((v.float()-dense).square().sum()) for n,v in reference.items()}
                cache_base = 'plain_cem_graph_control' if graph_control else 'plain_cem'
                for rate in rates:
                    n=f'plain_cem_s{rate}'
                    latent_errors.append(dict(condition=condition['id'], rate=rate,
                        combined_MSE=float((reference[n].float()-dense).square().mean()),
                        squared_error_ratio_vs_Sparse=errors[n]/max(errors[f'sparse_s{rate}'],1e-30),
                        squared_error_ratio_vs_Cache=errors[n]/max(errors[cache_base],1e-30)))
                write(output/'formal.partial.json', dict(pid=os.getpid(), rows=rows, windows=windows))
                print('CONDITION_COMPLETE', ci+1, condition['id'], len(rows), flush=True)
                break
        bootstrap = np.random.default_rng(20261017).integers(32, size=(4000,32))
        summary = {}
        for field in ('denoiser_ms','trajectory_wall_ms'):
            values={n:np.array([r[field] for c in model.pilot for rep in range(8) for r in rows
                               if r['method']==n and r['condition']==c['id'] and r['repeat']==rep]) for n in names}
            axis={n:dict(median_ms=float(np.median(v)),speedup_vs_Dense=float(np.median(values['dense_default'])/np.median(v))) for n,v in values.items()}
            extra={}
            for rate in rates:
                n=f'plain_cem_s{rate}'
                bases=[f'sparse_s{rate}','plain_cem',*(['plain_cem_graph_control'] if graph_control else [])]
                result={b:dict(speedup=float(np.median(values[b])/np.median(values[n])),
                    delta_CI95_ms=np.quantile(np.median((values[n]-values[b])[bootstrap],axis=1),[.025,.975]).tolist()) for b in bases}
                b='dense_graph_control' if graph_control else 'dense_default'
                cb='plain_cem_graph_control' if graph_control else 'plain_cem'
                result['interaction_ratio_same_execution']=float(np.median(values[f'sparse_s{rate}'])*np.median(values[cb])/(np.median(values[n])*np.median(values[b])))
                extra[str(rate)]=result
            summary[field]=dict(methods=axis,additional_gain=extra)
        if len(rows)!=32*len(names):
            raise RuntimeError('incomplete matrix')
        write(output/'formal_complete.json', dict(complete=True,pid=os.getpid(),binding=binding,schedule=schedule,
            methods=names,rows=rows,summary=summary,latent_diagnostics=latent_errors,windows=windows,memory=memory_records,
            warmups=4,repeats=8,timed_compile=False,scientific_quality_complete=False,
            wall_scope='cached-text pipeline trajectory incl scheduler/CPU final latent; no text encode or VAE'))
        status('pilot_complete')
    finally:
        # The gate owns CapturedCall wrappers; release them without altering the shared runtime.
        from .runtime import CapturedCall
        for gate in gates.values():
            if isinstance(gate.core,CapturedCall):
                gate.core.close()
        for r in recoveries.values():
            r.close()
        model.selector = None
        model.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    from .config import load
    execute(load(a.config),a.output)


if __name__=='__main__':
    main()
