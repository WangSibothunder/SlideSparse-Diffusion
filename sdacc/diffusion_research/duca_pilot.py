"""One PixArt-alpha DuCa+CEM+Sparse four-rate pilot and bounded hotpath audit."""
import argparse
import os
import time
from pathlib import Path
from .io import read,write,sha,save_tensor


def execute(config,output,schedule_file):
    import numpy as np
    import torch
    from .registry import build
    from .runner import policy_book
    from .affine import AffinePolicy
    from .cache import execution_variant
    from .runtime import CapturedCall,Timer
    from .models.pixart_duca import Features,install,Replay,RowProjection
    from .resources import Isolation,Memory

    output = Path(output)
    if (output/'formal_complete.json').exists():raise FileExistsError('pilot already complete')
    output.mkdir(parents=True,exist_ok=True)
    def status(phase,**values):
        write(output/'status.json',dict(phase=phase,pid=os.getpid(),timestamp=time.time(),**values))
    status('loading')
    model = build(config['model'],config,output/'runtime')
    if config['model']!='pixart-alpha' or model.steps!=20:raise ValueError('native Alpha256/20 only')
    schedule = read(schedule_file)['schedule']
    refresh = schedule['refresh']
    c = Features(model.native,refresh)
    kinds=[]
    for step in range(20):c.select(step);kinds.append(c.kind)
    active_steps=[t for t,k in enumerate(kinds) if k!='aggressive']
    book=policy_book(config)
    recoveries={}
    definitions={}
    for rate in (15,25,35,45):
        name=f'priority_s{rate}';policy=book[name]
        boundaries={0,19}|{t for t in range(1,20) if policy[t]!=policy[t-1]}
        if not boundaries<=set(refresh):raise ValueError('CEM skips a Sparse policy boundary')
        path=Path(config['coefficients_directory'])/name/'coefficients.pt'
        recovery=AffinePolicy(model,policy,path)
        for (uid,mode),op in list(recovery.operators.items()):
            recovery.operators[uid,mode]=RowProjection(op,model.units[uid]['m'],model.library)
        recovery.original_variant=recovery.variant
        recovery.duca_variant=execution_variant(recovery.variant,active_steps)
        recovery.step_records=[dict(step=t,projections=sum(bool(m) for m in row),copies=1,
            bytes=recovery.target.numel()*recovery.target.element_size(),direction='D2D',scope='Timer.select')
            for t,row in enumerate(policy)]
        recoveries[rate]=recovery
        definitions[str(rate)]=dict(policy=policy,coefficients=str(path),coefficient_SHA=sha(path))

    # No weight movement on the hot path: original resident packed banks remain.
    # Precomputed transfer metadata replaces the old repeated per-step policy scan.
    def fast_select(recovery,step):
        model.select_raw(step)
        recovery.target.copy_(recovery.step_parameters[step],non_blocking=True)
        for uid,mode in recovery.variant['updates'][step]:
            if mode:model.states[uid].active=recovery.operators[uid,mode]
        recovery.records.append(recovery.step_records[step])

    pool=torch.cuda.graph_pool_handle()
    dense_call=model.nt.call
    sparse_call=model.pt.call.call if isinstance(model.pt.call,CapturedCall) else model.pt.call
    model.pt.call=Replay(sparse_call,lambda:(model.selector,tuple(model.variant['policy'][model.pt.index])),lambda:model.pt.enabled,pool=pool)
    model.gt.call=Replay(dense_call,lambda:None,lambda:model.gt.enabled,pool=pool)
    for rate,recovery in recoveries.items():
        recovery.select=lambda step,r=recovery:fast_select(r,step)
    dense_duca=install(model.native,c)
    sparse_duca=install(model.patched,c)
    dt=Timer(dense_duca,None,20)
    st=Timer(sparse_duca,None,20)
    def duca_key(sparse):
        return (model.selector if sparse else None,
            tuple(model.variant['policy'][c.step]) if sparse else None,
            c.kind,c.step if c.kind=='partial' else None)
    dt.call=Replay(dense_duca,lambda:duca_key(False),lambda:dt.enabled,features=c,pool=pool)
    st.call=Replay(sparse_duca,lambda:duca_key(True),lambda:st.enabled,features=c,pool=pool)
    def select_duca(step):
        c.select(step)
        if model.selector is not None and c.kind!='aggressive':model.selector.select(step)
    dt.select=select_duca;st.select=select_duca
    replays=[model.pt.call,model.gt.call,dt.call,st.call]
    names=['dense_default','dense_graph_control','duca_cem']
    for rate in recoveries:names.extend([f'sparse_s{rate}',f'duca_cem_s{rate}'])
    binding=dict(model=config['model'],spec=model.adapter.receipt,source_manifest_SHA=sha(Path(__file__).parent/'SOURCE_MANIFEST.json'),
        schedule_file=schedule_file,schedule_SHA=sha(schedule_file),schedule=schedule,kinds=kinds,
        prior_scope='unchanged previous single-input native warmup proxy; no100cal or additional prior forwards',
        definitions=definitions,DuCa_commit='0a7151b7bfea3822295e8d29f26cd2e10756670c',
        fresh_ratio=.75,ratio_scheduler='ToCa',soft_fresh_weight=.25,fresh_threshold=3,
        cache='per-block self/cross/MLP features; ToCa token refresh; aggressive hidden reuse BEFORE final head',
        sparse_partial_caller='public cfg5/native; same packed banks and affine scales/bias; exact selected M',
        full_caller='original accepted recommended complete-projection route',
        compiled_Dense_denominator=True,cache_graph_execution_matched=True,
        no_new_calibration=True,no64=True,quality_complete=False)
    write(output/'frozen_inputs.json',binding)
    rows=[];windows=[];memory_records=[];latent_errors=[]
    memory=Memory(model)

    def generate(condition,name,timing=False):
        rate=int(name.split('_s')[-1]) if '_s' in name else None
        duca=name.startswith('duca_cem')
        recovery=recoveries.get(rate)
        model.selector=recovery
        if recovery:
            recovery.variant=recovery.duca_variant if duca else recovery.original_variant
            recovery.reset()
        model.variant=recovery.variant if recovery else None
        c.producer=0;c.step=0
        timer=st if duca and rate else dt if duca else model.pt if rate else model.gt if name=='dense_graph_control' else model.nt
        timer.index=0;timer.enabled=timing
        data,model.text_mask=model.conditioning(condition);model.mask=model.text_mask
        for replay in replays:
            replay.condition=condition['id'];replay.copy_bytes=0;replay.copy_calls=0
        select_ms=0.
        original_select=timer.select
        def measured_select(step):
            nonlocal select_ms
            start=time.perf_counter();original_select(step);select_ms+=(time.perf_counter()-start)*1000
        if timing and original_select is not None:timer.select=measured_select
        model.adapter.pipe.transformer=timer
        torch.cuda.synchronize();start=time.perf_counter()
        try:
            latent,_=model.adapter.generate([condition],data,return_cpu=True)
            torch.cuda.synchronize()
        finally:
            timer.select=original_select
        wall=(time.perf_counter()-start)*1000
        if timer.index!=20 or not torch.isfinite(latent).all():raise ValueError('invalid trajectory')
        copies=recovery.records if recovery else []
        return latent,dict(pid=os.getpid(),condition=condition['id'],method=name,
            denoiser_ms=sum(s.elapsed_time(e) for s,e in timer.events) if timing else None,
            trajectory_wall_ms=wall,selector_cpu_ms=select_ms,
            affine_copy_bytes=sum(r['bytes'] for r in copies),affine_copy_records=len(copies),
            graph_input_copy_bytes=sum(r.copy_bytes for r in replays),graph_input_copy_calls=sum(r.copy_calls for r in replays),
            full_steps=len(refresh) if duca else 20,partial_steps=kinds.count('partial') if duca else 0,
            aggressive_steps=kinds.count('aggressive') if duca else 0)

    try:
        for ci,condition in enumerate(model.pilot):
            attempt=0
            while True:
                model.physical.wait_clean()
                records=[];reference={}
                with Isolation(model.physical) as isolation:
                    memory.begin()
                    for name in names:
                        status('warm4',condition=condition['id'],method=name,attempt=attempt)
                        for warm in range(4):
                            # No extra trajectory: profile one already required stable warmup.
                            if ci==0 and attempt==0 and name=='sparse_s25' and warm==3:
                                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
                                    z,_=generate(condition,name)
                                profiler.export_chrome_trace(str(output/'sparse25_warm_trace.json'))
                                (output/'sparse25_warm_profile.txt').write_text(profiler.key_averages().table(sort_by='self_cuda_time_total',row_limit=30))
                            else:z,_=generate(condition,name)
                            reference[name]=z
                        print('WARMED',ci,name,flush=True)
                    counters=dict(torch._dynamo.utils.counters['stats'])
                    graph_counts=[len(r.graphs) for r in replays]
                    status('paired8',condition=condition['id'],attempt=attempt)
                    for repeat in range(8):
                        for name in names if repeat%2==0 else names[::-1]:
                            z,row=generate(condition,name,True)
                            error=float((z.float()-reference[name].float()).norm()/reference[name].float().norm().clamp_min(1e-12))
                            if error>.01:raise RuntimeError('same-PID output changed')
                            reference[name]=z;records.append(dict(row,repeat=repeat))
                        if counters!=dict(torch._dynamo.utils.counters['stats']) or graph_counts!=[len(r.graphs) for r in replays]:
                            raise RuntimeError('compilation/capture during formal timing')
                    peak=memory.finish()
                receipt=dict(condition=condition['id'],attempt=attempt,foreign=sorted(isolation.foreign),error=isolation.error)
                if isolation.foreign or isolation.error:
                    write(output/'contaminated'/f'{condition["id"]}.{attempt}.json',dict(receipt,rows=records));attempt+=1;continue
                rows.extend(records);windows.append(receipt);memory_records.append(peak)
                save_tensor(output/'pilot_latents'/(condition['id']+'.pt'),dict(condition=condition,pid=os.getpid(),latents=reference))
                dense=reference['dense_default'].float()
                errors={n:float((z.float()-dense).square().sum()) for n,z in reference.items()}
                for rate in recoveries:
                    name=f'duca_cem_s{rate}'
                    latent_errors.append(dict(condition=condition['id'],rate=rate,
                        combined_MSE=float((reference[name].float()-dense).square().mean()),
                        squared_error_ratio_vs_Sparse=errors[name]/max(errors[f'sparse_s{rate}'],1e-30),
                        squared_error_ratio_vs_DuCaCEM=errors[name]/max(errors['duca_cem'],1e-30)))
                write(output/'formal.partial.json',dict(pid=os.getpid(),rows=rows,windows=windows))
                print('CONDITION_COMPLETE',ci+1,len(rows),flush=True)
                break
        random=np.random.default_rng(20261017).integers(32,size=(4000,32))
        summary={}
        for field in ('denoiser_ms','trajectory_wall_ms'):
            values={name:np.array([r[field] for condition in model.pilot for repeat in range(8) for r in rows if r['condition']==condition['id'] and r['repeat']==repeat and r['method']==name]) for name in names}
            axis={n:dict(median_ms=float(np.median(v)),speedup_vs_Dense=float(np.median(values['dense_default'])/np.median(v))) for n,v in values.items()}
            extra={}
            for rate in recoveries:
                n=f'duca_cem_s{rate}';base='duca_cem'
                extra[str(rate)]=dict(speedup_vs_DuCaCEM=float(np.median(values[base])/np.median(values[n])),
                    speedup_vs_Sparse=float(np.median(values[f'sparse_s{rate}'])/np.median(values[n])),
                    delta_CI95_ms=np.quantile(np.median((values[n]-values[base])[random],axis=1),[.025,.975]).tolist(),
                    interaction_ratio=float(np.median(values[f'sparse_s{rate}'])*np.median(values[base])/(np.median(values[n])*np.median(values['dense_graph_control']))))
            summary[field]=dict(methods=axis,additional_gain=extra)
        if len(rows)!=352:raise RuntimeError('incomplete matrix')
        write(output/'formal_complete.json',dict(complete=True,pid=os.getpid(),binding=binding,methods=names,rows=rows,
            summary=summary,latent_diagnostics=latent_errors,windows=windows,memory=memory_records,
            warmups=4,repeats=8,timed_compile=False,quality_complete=False,
            capture=dict(routes=[len(r.graphs) for r in replays],seconds=[r.capture_seconds for r in replays],max_nrmse=[r.max_nrmse for r in replays]),
            wall_scope='cached-text pipeline trajectory with scheduler and CPU final latent; excludes text/VAE'))
        status('pilot_complete')
    finally:
        for r in replays:r.close()
        for r in recoveries.values():r.close()
        model.selector=None;model.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--schedule',required=True)
    a=p.parse_args()
    from .config import load
    execute(load(a.config),a.output,a.schedule)


if __name__=='__main__':main()
