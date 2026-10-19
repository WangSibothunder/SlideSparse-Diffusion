"""Shared explicit execution. No experiments are launched by metadata commands."""
import os,json,time
from pathlib import Path
from .io import read,write,sha,save_tensor
from .registry import build

def policy_book(config):
    packet=read(config['policy_file'])
    if 'candidates' in packet:return {v['name']:v['policy'] for v in packet['candidates']}
    return {k:v['policy'] for k,v in packet['definitions'].items()}

def execute(config,args):
    if args.gpu is not None:
        import subprocess
        uuid=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        os.environ.update(SDACC_PHYSICAL_GPU=str(args.gpu),CUDA_VISIBLE_DEVICES=uuid)
    if not os.environ.get('SDACC_PHYSICAL_GPU') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('choose exactly one physical GPU with --gpu')
    output=Path(args.output).resolve()
    if args.stage=='study':
        from .study import launch
        return launch(config,args)
    if args.stage=='profile':
        from .performance import profile
        return profile(config,output)
    if args.stage in ['decode','score']:
        from .resources import PhysicalGPU
        physical=PhysicalGPU(os.environ['SDACC_PHYSICAL_GPU']);physical.wait_clean();physical.close()
        from . import image_metrics
        generated=read(output/'generation_complete.json')
        if not generated['complete']:raise ValueError('generation is incomplete')
        if args.stage=='decode':image_metrics.decode(config,output,generated)
        else:image_metrics.score(None,output,generated,dict(config,dense_root=config['dense_results']))
        return
    if output.exists() and any(output.iterdir()) and args.stage=='benchmark':raise FileExistsError('benchmark needs a new directory to retain one producer PID')
    if (output/'context.json').exists() and read(output/'context.json')['config_SHA']!=sha(config['config_path']):raise ValueError('resume config differs')
    output.mkdir(parents=True,exist_ok=True)
    import torch
    from .affine import AffinePolicy
    from .resources import Memory,Isolation,resource_snapshot
    owned=not hasattr(args,'runtime');model=build(config['model'],config,output) if owned else args.runtime;memory=Memory(model);policies=policy_book(config);recoveries={};methods={'dense_default':(None,None)}
    selected={args.policy:policies[args.policy]} if args.policy else policies
    if args.stage=='benchmark' and config.get('benchmark_dense_graph_control'):methods['dense_graph_control']=(None,None)
    if args.stage=='infer' and not args.policy:selected={}
    try:
        for name,policy in selected.items():
            methods[name+'_raw']=(model.prepare_variant(policy),None)
            if args.affine:
                recovery=AffinePolicy(model,policy,Path(config['coefficients_directory'])/name/'coefficients.pt');recoveries[name]=recovery;methods[name+'_affine']=(recovery.variant,recovery)
        def generate(condition,name,timing=False):
            variant,recovery=methods[name];model.selector=recovery
            if recovery:recovery.reset()
            elif not hasattr(model,'coefficient_arena'):
                for slot in model.bank_slots.values():slot.current_mode=None
            memory.begin();latent,row=model.generate(condition,variant,timing=timing,dense_graph=name=='dense_graph_control');row.update(method=name,own_peak=memory.finish())
            if recovery:row['transfers']=list(recovery.records)
            if hasattr(model,'coefficient_arena'):row['bank_transfers']=list(model.transfer_records)
            return latent,row
        write(output/'resources_initial.json',resource_snapshot(output))
        if not (output/'context.json').exists():write(output/'context.json',dict(pid=os.getpid(),model=config['model'],config_SHA=sha(config['config_path']),runner_SHA=sha(__file__),historical_results_reused_as_measurements=False))
        if args.stage=='calibrate':
            from .calibration import collect_conditions
            if not args.policy:raise ValueError('calibrate one explicitly named fixed policy')
            model.selector=None;collect_conditions(model,policies[args.policy],output/args.policy,Memory)
        elif args.stage=='benchmark':
            for condition in model.pilot:
                attempt=0
                while True:
                    model.physical.wait_clean();records=[];reference={}
                    with Isolation(model.physical) as isolation:
                        for warm in range(4):
                            for name in methods:
                                latent,row=generate(condition,name)
                                if name in reference and float((latent.float()-reference[name].float()).norm()/reference[name].float().norm().clamp_min(1e-12))>.01:raise RuntimeError('warm/reload output changed')
                                reference[name]=latent;records.append(dict(row,stage='warm',repeat=warm))
                        for repeat in range(8):
                            _,before=generate(condition,'dense_default',True);block=[]
                            names=list(methods)[1:];offset=repeat%len(names) if names else 0
                            for name in names[offset:]+names[:offset]:
                                latent,row=generate(condition,name,True)
                                if float((latent.float()-reference[name].float()).norm()/reference[name].float().norm().clamp_min(1e-12))>.01:raise RuntimeError('timed output changed')
                                block.append(row)
                            _,after=generate(condition,'dense_default',True);dense=(before['denoiser_ms']+after['denoiser_ms'])/2
                            records.extend(dict(row,stage='repeat',repeat=repeat,paired_Dense_before=before,paired_Dense_after=after,paired_Dense_ms=dense,speedup=dense/row['denoiser_ms']) for row in block)
                    packet=dict(pid=os.getpid(),condition=condition,records=records,steps=model.steps,warmups=4,repeats=8,foreign=sorted(isolation.foreign),monitor_error=isolation.error)
                    if isolation.foreign or isolation.error:
                        write(output/'contaminated'/(condition['id']+'.'+str(attempt)+'.json'),packet);attempt+=1;continue
                    write(output/'speed'/(condition['id']+'.json'),dict(packet,complete=True));break
            from .runtime import CapturedCall
            graphs={name:dict(captures=len(timer.call.graphs),max_nrmse=timer.call.max_nrmse) for name,timer in [('sparse',model.pt),*([('dense_control',model.gt)] if hasattr(model,'gt') else [])] if isinstance(timer.call,CapturedCall)}
            write(output/'speed_complete.json',dict(complete=True,pid=os.getpid(),conditions=[c['id'] for c in model.pilot],methods=list(methods),steps=model.steps,warmups=4,repeats=8,graphs=graphs))
        else:
            names=[args.policy+('_affine' if args.affine else '_raw')] if args.policy else ['dense_default'];sources=[]
            for method in names:
                for condition in model.evaluation:
                    path=output/'latents'/method/(condition['id']+'.pt');path.parent.mkdir(parents=True,exist_ok=True)
                    if path.exists():
                        packet=torch.load(path,map_location='cpu',weights_only=True)
                        if packet['condition']!=condition or packet['method']!=method:raise ValueError('committed latent differs')
                    else:
                        latent,row=generate(condition,method);packet=dict(latent=latent,condition=condition,generation_pid=os.getpid(),method=method,own_peak=row['own_peak']);save_tensor(path,packet)
                    sources.append(dict(method=method,condition=condition['id'],path=str(path),sha=sha(path),generation_pid=packet['generation_pid'],own_peak=packet.get('own_peak')));model.data.pop(condition['id'],None)
                    write(output/'generation_progress.json',dict(completed=len(sources),pid=os.getpid()))
            write(output/'generation_complete.json',dict(complete=True,pid=os.getpid(),methods=names,conditions=model.evaluation,latent_sources=sources,evaluation_used_for_fit_or_selection=False))
        write(output/'resources_final.json',resource_snapshot(output))
    finally:
        for recovery in recoveries.values():recovery.close()
        model.selector=None
        if owned:model.close()
