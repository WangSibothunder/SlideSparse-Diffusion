"""One resumable study, using the common runtime, runner and metric stages."""
import argparse,os,sys,time,json,subprocess,csv,statistics,traceback,gc
from pathlib import Path
from types import SimpleNamespace
from .io import read,write,sha

def child(config,args,label,command):
    output=Path(args.output);folder=output/'execution';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(label+'.json')
    if path.exists():
        with (folder/'history.jsonl').open('a') as history:history.write(json.dumps(read(path))+'\n')
    with (folder/(label+'.log')).open('ab') as log:
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1'))
        write(path,dict(stage=label,pid=p.pid,parent=os.getpid(),command=command,start_time=time.time()))
        code=p.wait()
    record=read(path);record.update(actual_waitpid_returncode=code,end_time=time.time());write(path,record)
    if code:raise RuntimeError(f'{label} exited {code}; see {folder/(label+".log")}')

def launch(config,args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():return
    write(output/'request.json',dict(model=config['model'],spec=config['spec'],sparsities=config['sparsities'],calibration=100,pilot=4,evaluation=64,bank_method='OBS-Diff fixed-order shared-curvature joint INT8',quality_selection=False,existing_results_not_rerun=True,code_manifest_SHA=sha(Path(__file__).parent/'SOURCE_MANIFEST.json')))
    common=[sys.executable,'-m','sdacc.diffusion_research.study','--config',config['config_path'],'--output',str(output)]
    try:
        if not (output/'prepared_complete.json').exists():child(config,args,'prepare',common+['--worker','prepare'])
        if not (output/'measurement_complete.json').exists():child(config,args,'measure',common+['--worker','measure'])
        groups=[('dense',output/'dense')]+[(f'priority_s{p:02d}',output/'evaluation'/f'priority_s{p:02d}') for p in config['sparsities']]
        for name,dest in groups:
            if not (dest/'decode_complete.json').exists():
                child(config,args,'decode_'+name,[sys.executable,'-m','sdacc.diffusion_research','run','--model',config['model'],'--config',config['config_path'],'--stage','decode','--output',str(dest)])
        for name,dest in groups[1:]:
            if not (dest/'quality_complete.json').exists():
                child(config,args,'score_'+name,[sys.executable,'-m','sdacc.diffusion_research','run','--model',config['model'],'--config',config['config_path'],'--stage','score','--output',str(dest)])
        report(config,output)
    except BaseException as error:
        write(output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),timestamp=time.time()));raise


def measure(config,output):
    import torch
    from .registry import build
    from .prepare import sensitivity,freeze_policies,status
    from .calibration import collect_conditions
    from .resources import Memory,resource_snapshot
    from .runner import execute,policy_book
    model=build(config['model'],config,output/'measurement')
    try:
        if not Path(config['policy_file']).exists():
            errors=sensitivity(model,config,output);freeze_policies(config,model.units,errors)
        policies=policy_book(config)
        if config.get('initial_raw_profile') and not (output/'initial_performance/profile.json').exists():
            from .performance import profile
            status(output,'initial_raw_performance')
            profile(config,output/'initial_performance',model=model,include_affine=False)
        for name,policy in policies.items():
            folder=Path(config['coefficients_directory'])/name
            if (folder/'complete.json').exists():continue
            status(output,'own100cal_affine',policy=name,completed=read(folder/'progress.json').get('completed_conditions',0) if (folder/'progress.json').exists() else 0)
            model.selector=None;collect_conditions(model,policy,folder,Memory)
        speed=output/'benchmark'
        if not (speed/'speed_complete.json').exists():
            if speed.exists() and any(speed.iterdir()):
                previous=read(speed/'context.json');archive=output/'interrupted'/('benchmark_'+str(previous['pid']));archive.parent.mkdir(exist_ok=True);speed.rename(archive)
            status(output,'same_PID_Dense_raw_affine_benchmark',conditions=4,warmups=4,repeats=8)
            execute(config,SimpleNamespace(gpu=None,output=str(speed),stage='benchmark',policy=None,affine=True,runtime=model))
        if not (output/'dense/generation_complete.json').exists():
            status(output,'Dense64_generation');execute(config,SimpleNamespace(gpu=None,output=str(output/'dense'),stage='infer',policy=None,affine=False,runtime=model))
        for name in policies:
            dest=output/'evaluation'/name
            if (dest/'generation_complete.json').exists():continue
            status(output,'Sparse64_affine_generation',policy=name)
            execute(config,SimpleNamespace(gpu=None,output=str(dest),stage='infer',policy=name,affine=True,runtime=model))
        write(output/'measurement_complete.json',dict(complete=True,pid=os.getpid(),calibration=400,evaluation=256,Dense_evaluation=64,resources=resource_snapshot(output)))
    finally:model.close()


def report(config,output):
    source=read(output/'benchmark/speed_complete.json');rows=[];timing=[];steps=config['spec']['steps']
    for c in source['conditions']:timing+=read(output/'benchmark/speed'/(c+'.json'))['records']
    for percent in config['sparsities']:
        name=f'priority_s{percent:02d}';raw=[r for r in timing if r['method']==name+'_raw' and r['stage']=='repeat'];aff=[r for r in timing if r['method']==name+'_affine' and r['stage']=='repeat']
        quality=read(output/'evaluation'/name/'quality_complete.json')['summary'][name+'_affine'];latency=statistics.mean(r['denoiser_ms'] for r in aff)/1000
        row=dict(sparsity_percent=percent,actual_sparsity_percent=100*read(config['policy_file'])['definitions'][name]['actual_structural_sparsity'],raw_speedup=statistics.mean(r['speedup'] for r in raw),affine_speedup=statistics.mean(r['speedup'] for r in aff),affine_denoiser_seconds=latency,affine_steps_per_second=steps/latency,PSNR_dB=quality['psnr']['mean'],Gaussian_wSSIM=quality['wssim']['mean'],CLIP_ViTB16=quality['clip']['mean'],ImageReward_v1=quality['imagereward']['mean'])
        row[f'affine_{steps}_steps_seconds']=latency;rows.append(row)
    write(output/'RESULTS.json',dict(complete=True,rows=rows,timing_PID=source['pid'],protocol=read(output/'request.json')))
    with (output/'RESULTS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    spec=config['spec'];resolution=spec['resolution']
    lines=[f"# {config['model']} / {resolution}×{resolution} / {steps}步",'',
        f"BF16，CFG{spec['guidance']}、text{spec['max_sequence_length']}；保留checkpoint原生scheduler。共享W8A8 SlideSparse，OBS-Diff固定列序、共享100cal曲率与INT8联合补偿；不宣称官方ExactOBC。",'',
        f"四档预先固定；阶段边界{config['phases']}。按100cal实际最小增量误差和整投影整数预算依次6:8→4:6→2:4。每档自身100条未恢复完整{steps}步轨迹拟合FP64逐通道仿射；完整同输入Dense teacher后取≤128有效均匀行，保留全部通道；离线FP32 fold，运行无fit。attention核心及输出投影保持Dense。",'',
        '同一PID、4条件、4warm、8repeat，各次Dense前后配对，报告32个配对比值均值。必要输入、参数选择与拷贝均计时。每档64个独立质量条件，相同prompt/seed的Dense为PSNR/wSSIM参照；CLIP和官方ImageReward用生成图与prompt，evaluation不参与选择。','',
        f'| 目标稀疏率 | 实际稀疏率 | raw加速比 | 仿射加速比 | 仿射{steps}步耗时/s | PSNR/dB | Gaussian wSSIM | CLIP ViT-B/16 | ImageReward v1.0 |','|---|---|---|---|---|---|---|---|---|']
    for r in rows:lines.append(f"| {r['sparsity_percent']}% | {r['actual_sparsity_percent']:.4f}% | {r['raw_speedup']:.4f}× | {r['affine_speedup']:.4f}× | {r['affine_denoiser_seconds']:.4f} | {r['PSNR_dB']:.4f} | {r['Gaussian_wSSIM']:.5f} | {r['CLIP_ViTB16']:.5f} | {r['ImageReward_v1']:.5f} |")
    lines+=['',f'耗时为实际{steps}步去噪累计秒数；不含文本编码、调度器CPU、VAE和评分，不作其他步数的线性折算。','',f"测速PID：{source['pid']}。原始重复在benchmark/speed，质量在evaluation/*/quality_complete.json；已完成的其他模型实验未重跑。"]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n');write(output/'complete.json',dict(complete=True,timestamp=time.time(),rows=4,timing_PID=source['pid'],report_SHA=sha(output/'REPORT.md')))


def supervise(config,output):
    """A real waitpid for a long study; no monitoring or extra experiments."""
    for n in ['started.json','exit.json','failure.json']:
        p=output/n
        if p.exists():
            with (output/'history.jsonl').open('a') as f:f.write(json.dumps(dict(record=n,value=read(p)))+'\n')
    command=[sys.executable,'-m','sdacc.diffusion_research','run','--model',config['model'],'--config',config['config_path'],'--stage','study','--output',str(output)]
    with (output/'worker.log').open('ab') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        stat=Path(f'/proc/{process.pid}/stat').read_text();ticks=int(stat[stat.rindex(')')+2:].split()[19])
        write(output/'started.json',dict(pid=process.pid,parent=os.getpid(),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=ticks,command=command,GPU=int(os.environ['SDACC_PHYSICAL_GPU']),start_time=time.time()))
        for n in ['exit.json','failure.json']:
            p=output/n
            if p.exists():p.unlink()
        code=process.wait();write(output/'exit.json',dict(pid=process.pid,actual_waitpid_returncode=code,end_time=time.time()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=['prepare','measure','supervise'],required=True);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    from .config import load
    config=load(args.config);output=Path(args.output)
    if args.worker=='supervise':
        supervise(config,output);return
    if args.worker=='prepare':
        from .prepare import prepare
        prepare(config,output)
    else:measure(config,output)

if __name__=='__main__':main()
