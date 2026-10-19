"""Common preparation: actual Dense100 curvature, packed banks and cal-only policies."""
import os,time,gc,hashlib
from pathlib import Path
import numpy as np
import torch
from .io import read,write,save_tensor,sha
from .runtime import Runtime,Timer
from .resources import Memory,resource_snapshot
from .registry import inventory

def status(output,stage,**values):
    write(Path(output)/'status.json',dict(stage=stage,pid=os.getpid(),timestamp=time.time(),**values))

def valid_rows(unit,x,mask,limit=None):
    flat=x.reshape(-1,x.shape[-1])
    if unit['module']=='cross-KV' or unit.get('branch')=='text':flat=flat[mask.bool().flatten()]
    if limit and len(flat)>limit:flat=flat[torch.linspace(0,len(flat)-1,limit,device=flat.device).long()]
    return flat

@torch.inference_mode()
def prepare_conditioning(config,output):
    """Encode a new model's fixed conditions once, before Dense statistics."""
    if not config.get('prepare_text_conditioning'):return
    from dataclasses import replace
    from .pipeline import Adapter,SPECS
    from .resources import PhysicalGPU
    from .library import bind
    bind(config['library']);physical=PhysicalGPU(os.environ['SDACC_PHYSICAL_GPU']);physical.wait_clean()
    spec=replace(SPECS[config['spec_name']],**config['spec'])
    adapter=Adapter(spec,Path(config['checkpoint']));adapter.conditioning_gpu()
    split=read(config['conditions']);conditions=split['calibration']+split['pilot']+split['evaluation']
    try:
        for i,condition in enumerate(conditions):
            path=Path(config['conditioning_directory'])/(condition['id']+'.pt')
            if not path.exists():save_tensor(path,dict(condition=condition,checkpoint_configs=adapter.receipt['configs_sha256'],payload=adapter.encode(condition),pid=os.getpid()))
            status(output,'text_conditioning',completed=i+1,total=len(conditions))
    finally:
        del adapter;gc.collect();torch.cuda.empty_cache();physical.close()


@torch.inference_mode()
def prepare(config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not Path(config['conditions']).exists():
        old=read(config['conditioning_source']);old=old.get('conditions',old);conditions=dict(calibration=old['calibration'][:100],pilot=old.get('pilot',old.get('expansion',[]))[:4],evaluation=old['evaluation'][:64])
        write(config['conditions'],conditions)
    if (output/'prepared_complete.json').exists():return
    prepare_conditioning(config,output)
    model=Runtime();model.spec_name=config['spec_name'];model.start(config,output/'preparation')
    model.states=[];model.et=Timer(model.native,model.native,model.steps);model.nt=model.et
    units=inventory(config['model'],model.native,config['spec']['resolution'],config['spec']['max_sequence_length']);model.units=units
    write(config['units'],dict(execution_units=units,excluded=['attention-core','attention-out'],resolution=config['spec']['resolution']))
    hessian_gpu_units=None
    if 'hessian_gpu_budget_bytes' in config:
        remaining=int(config['hessian_gpu_budget_bytes']);hessian_gpu_units=set()
        for u in sorted(units,key=lambda u:u['k']**2,reverse=True):
            size=4*u['k']**2
            if size<=remaining:hessian_gpu_units.add(u['unit']);remaining-=size
    memory=Memory(model);memory.begin();hessians={};counts={};hooks=[]
    complete=output/'hessians/complete.json'
    try:
        if not complete.exists():
            def hook(unit):
                def collect(module,args):
                    x=valid_rows(unit,args[0],model.text_mask,128).float();uid=unit['unit']
                    device='cuda' if hessian_gpu_units is None or uid in hessian_gpu_units else 'cpu'
                    x=x.to(device)
                    if uid not in hessians:hessians[uid]=torch.zeros((unit['k'],unit['k']),device=device);counts[uid]=0
                    hessians[uid].addmm_(x.T,x);counts[uid]+=len(x)
                return collect
            for u in units:hooks.append(model.native.get_submodule(u['targets'][0]).register_forward_pre_hook(hook(u)))
            for i,c in enumerate(model.cal):
                model.generate(c,eager=True);model.data.pop(c['id'],None)
                status(output,'Dense100_curvature',completed=i+1,total=100)
            for handle in hooks:handle.remove()
            hooks=[]
            for u in units:
                uid=u['unit'];save_tensor(output/'hessians'/f'{uid}.pt',dict(hessian=(hessians[uid]/counts[uid]).cpu(),rows=counts[uid],unit=uid,conditions=100))
            write(complete,dict(complete=True,conditions=100,units=len(units),own_peak=memory.finish(),pid=os.getpid()))
            hessians.clear();gc.collect();torch.cuda.empty_cache()
        from .reconstruction import factor_hessian,compress
        configured=0;counts_path=output/'bank_zero_counts.json';bank_zeros=read(counts_path) if counts_path.exists() else {}
        for u in units:
            uid=u['unit'];missing=[g for g in (8,6,4) if not (Path(config['bank_directory'])/str(g)/(str(uid)+'.pt')).exists()]
            if not missing:
                configured+=3
                for g in (4,6,8):
                    key=f'{uid}:{g}'
                    if key not in bank_zeros:bank_zeros[key]=int(torch.load(Path(config['bank_directory'])/str(g)/(str(uid)+'.pt'),map_location='cpu',weights_only=True)['structural_zeros'])
                continue
            weights=torch.cat([model.native.get_submodule(n).weight for n in u['targets']]);bias=torch.cat([model.native.get_submodule(n).bias for n in u['targets']]).float().cpu()
            h=torch.load(output/'hessians'/f'{uid}.pt',map_location='cpu',weights_only=True)['hessian'].cuda()
            memory.begin();factor=factor_hessian(h)
            for group in missing:
                start=time.monotonic();bank=compress(weights,factor,group,method='joint',blocksize=192,allow_tail=config.get('allow_tail_groups',False))
                q=bank['qweight'].cpu();op=model.library.PackedLinear.prepare(q,bank['scales'].cpu(),bias,group,config=5,frontend='triton')
                zero_count=int(bank['zeros'].sum());bank_zeros[f'{uid}:{group}']=zero_count
                save_tensor(Path(config['bank_directory'])/str(group)/(str(uid)+'.pt'),dict(unit=uid,group=group,compact=op.packed,metadata=op.metadata,scales=op.scales,q_sha=hashlib.sha256(q.numpy().tobytes()).hexdigest(),method='OBS-Diff fixed-order shared-curvature joint INT8',structural_zeros=zero_count,elapsed_s=time.monotonic()-start))
                write(counts_path,bank_zeros)
                configured+=1;status(output,'packed_banks',completed=configured,total=len(units)*3,unit=uid,group=group)
                del bank,q,op
            memory.finish();del weights,bias,h,factor;gc.collect();torch.cuda.empty_cache()
        for u in units:u['structural_zeros']={str(g):bank_zeros[f"{u['unit']}:{g}"] for g in (4,6,8)}
        write(config['units'],dict(execution_units=units,excluded=['attention-core','attention-out'],resolution=config['spec']['resolution']))
        write(output/'prepared_complete.json',dict(complete=True,units=len(units),banks=len(units)*3,conditions=100,method='OBS-Diff fixed-order shared-curvature joint INT8',not_official_ExactOBC=True,resources=resource_snapshot(output),pid=os.getpid()))
    finally:
        for handle in hooks:handle.remove()
        model.close()

@torch.inference_mode()
def sensitivity(model,config,output):
    output=Path(output)
    if (output/'maps_complete.json').exists():return np.load(output/'sensitivity.npz')['mean_nl2']
    partial=output/'maps_partial.npz';shape=(model.steps,len(model.units),3)
    if partial.exists():
        previous=np.load(partial);sums=torch.from_numpy(previous['sums']).cuda();start=int(previous['completed'])
    else:sums=torch.zeros(shape,dtype=torch.float64,device='cuda');start=0
    hooks=[];buffers={};condition_errors=torch.zeros_like(sums)
    def hook(unit,j):
        def collect(module,args,result):
            uid=unit['unit'];step=model.et.index
            if j==0:buffers[uid]=[args[0],[]]
            buffers[uid][1].append(result)
            if j!=len(unit['targets'])-1:return
            x,parts=buffers.pop(uid);teacher=torch.cat(parts,-1);dense=valid_rows(unit,teacher,model.text_mask).float();den=dense.square().sum().clamp_min(1e-12)
            for gi,g in enumerate((4,6,8)):
                actual=model.states[uid].ops[g](x.reshape(-1,unit['k']));actual=actual.reshape(*x.shape[:-1],unit['n']);delta=valid_rows(unit,actual,model.text_mask).float()-dense
                condition_errors[step,uid,gi]=delta.square().sum()/den
        return collect
    for u in model.units:
        for j,n in enumerate(u['targets']):hooks.append(model.native.get_submodule(n).register_forward_hook(hook(u,j)))
    try:
        for i,c in enumerate(model.cal[start:],start):
            condition_errors.fill_(float('nan'));model.generate(c,eager=True)
            if buffers or not torch.isfinite(condition_errors).all():raise RuntimeError('incomplete actual projection map')
            sums+=condition_errors;model.data.pop(c['id'],None)
            temp=partial.with_suffix('.tmp.npz');np.savez(temp,sums=sums.cpu().numpy(),completed=i+1);temp.replace(partial)
            status(output,'cal100_sparse_error_map',completed=i+1,total=100)
        mean=sums.cpu().numpy()/100;np.savez(output/'sensitivity.npz',mean_nl2=mean,conditions=100,mode_order=[4,6,8]);write(output/'maps_complete.json',dict(complete=True,conditions=100,teacher='full same-input Dense',trajectory='Dense',evaluation_used=False,pid=os.getpid()));return mean
    finally:
        for h in hooks:h.remove()


def freeze_policies(config,units,error):
    """Reuse Sigma's mode-priority integer budget, with low incremental error."""
    from .allocation import allocate_phase
    smoothed=np.stack([np.median(error[max(0,t-1):min(len(error),t+2)],axis=0) for t in range(len(error))]);definitions={}
    for percent in config['sparsities']:
        policy=[[0]*len(units) for _ in range(len(error))];phases=[]
        for start,stop in config['phases']:
            rows=[dict(unit=u['unit'],parameters=u['k']*u['n'],errors={str(g):float(smoothed[start:stop,u['unit'],gi].mean()) for gi,g in enumerate((4,6,8))},structural_zeros=u.get('structural_zeros',{str(g):2*(u['k']*u['n']//g) for g in (4,6,8)}),admitted={str(g):True for g in (4,6,8)}) for u in units]
            answer=allocate_phase(rows,percent)
            for t in range(start,stop):policy[t]=answer['policy'].copy()
            phases.append(dict(answer,start=start,stop=stop))
        definitions[f'priority_s{percent:02d}']=dict(policy=policy,target_structural_sparsity=percent/100,actual_structural_sparsity=sum((p['stop']-p['start'])*p['structural_zeros'] for p in phases)/(len(error)*phases[0]['total_parameters']),phases=phases,evaluation_used=False)
    write(config['policy_file'],dict(complete=True,definitions=definitions,calibration_conditions=100,quality_selection=False));return definitions
