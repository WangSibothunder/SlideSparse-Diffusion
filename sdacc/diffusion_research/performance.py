"""Targeted denoiser profiling and timing; never calibrates or scores images."""
import argparse
import os
import time
from pathlib import Path
from .io import read, write, sha


def report(study):
    """Summarize the timing-only optimization without relabeling old quality."""
    import csv
    import statistics as st
    study=Path(study);root=study/'performance';run=root/'optimized'
    exit_record=read(run/'exit.json')
    if exit_record['actual_waitpid_returncode']!=0:raise RuntimeError('benchmark did not finish')
    done=read(run/'benchmark/speed_complete.json')
    records=[r for c in done['conditions'] for r in read(run/'benchmark/speed'/(c+'.json'))['records'] if r['stage']=='repeat']
    original=read(root/'baseline.json' if (root/'baseline.json').exists() else study/'RESULTS.json');rows=[]
    steps=original.get('steps',20);label=original.get('model','PixArt-α 256')
    for old in original['rows']:
        p=old['sparsity_percent'];prefix=f'priority_s{p:02d}'
        raw=[r for r in records if r['method']==prefix+'_raw'];affine=[r for r in records if r['method']==prefix+'_affine']
        control={c:st.mean(r['denoiser_ms'] for r in records if r['method']=='dense_graph_control' and r['condition']==c) for c in done['conditions']}
        row=dict(sparsity_percent=p,previous_raw_speedup=old['raw_speedup'],raw_speedup=st.mean(r['speedup'] for r in raw),previous_affine_speedup=old['affine_speedup'],affine_speedup=st.mean(r['speedup'] for r in affine),affine_denoiser_ms=st.mean(r['denoiser_ms'] for r in affine),paired_default_Dense_ms=st.mean(r['paired_Dense_ms'] for r in affine),raw_vs_graph_Dense=st.mean(control[r['condition']]/r['denoiser_ms'] for r in raw),affine_vs_graph_Dense=st.mean(control[r['condition']]/r['denoiser_ms'] for r in affine),affine_parameter_copies=st.mean(sum(t.get('copies',0) for t in r.get('transfers',[])) for r in affine),own_peak_GiB=max(r['own_peak']['own_peak_bytes'] for r in raw+affine)/(1<<30))
        row[f'affine_{steps}_steps_ms']=row['affine_denoiser_ms']
        for kind,group in [('raw',raw),('affine',affine)]:
            row[kind+'_weight_metadata_MiB']=st.mean(sum(t.get('weight_metadata_bytes',0) for t in r.get('bank_transfers',[])) for r in group)/(1<<20)
        rows.append(row)
    value=dict(complete=True,model=label,steps=steps,timing_PID=done['pid'],previous_timing_PID=original['timing_PID'],conditions=len(done['conditions']),warmups=done['warmups'],repeats=done['repeats'],quality_experiments_rerun=False,calibration_reused=True,bank_reconstruction_rerun=False,graphs=done['graphs'],rows=rows,graph_Dense_ms=st.mean(r['denoiser_ms'] for r in records if r['method']=='dense_graph_control'),source_SHA=sha(run/'source_code.tar.gz'))
    write(root/'RESULTS.json',value)
    with (root/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    flat=[]
    for r in records:
        flat.append({k:r.get(k) for k in ['pid','condition','method','repeat','denoiser_ms','paired_Dense_ms','speedup']})
    with (root/'raw_speed.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=[f'# {label}：适配开销与优化后的实际测速','',
        '复用现有校准、固定mask/packed banks和FP64拟合系数。只分析和重测去噪速度；没有重跑质量生成、解码或评分。原实验与质量数据保持原来源。','',
        original.get('optimization_note','复用推荐投影配置，合并参数拷贝，重放完整denoiser。必要输入/参数/银行传输均计入Timer.select。'),'',
        f"同PID {done['pid']}，完整{steps}步，4条件×4warm×8repeat。主表分母为各次前后配对的default compiled BF16 Dense，统计配对比值均值。另测相同完整图优化的Dense，区分通用启动优化与稀疏收益。原测速PID：{original['timing_PID']}，原统计口径保留。",'',
        f'| 目标稀疏率 | 原raw | 新raw | 原仿射 | 新仿射 | 新仿射{steps}步/ms | 仿射/图优化Dense |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['sparsity_percent']}% | {r['previous_raw_speedup']:.3f}× | {r['raw_speedup']:.3f}× | {r['previous_affine_speedup']:.3f}× | {r['affine_speedup']:.3f}× | {r['affine_denoiser_ms']:.2f} | {r['affine_vs_graph_Dense']:.3f}× |")
    lines += ['',f"图优化Dense：{value['graph_Dense_ms']:.2f}ms/{steps}步。实际图重放对compiled caller最大NRMSE：{max(v['max_nrmse'] for v in done['graphs'].values()):.6g}。",'', '旧质量指标直接复用。原始新重复在raw_speed.csv和optimized/benchmark/speed；每个条件污染则整条件归档重测，不拼接跨PID测速。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return value


def profile(config, output, shared=False, model=None, include_affine=True):
    import torch
    from .registry import build
    from .affine import AffinePolicy
    from .runner import policy_book
    from .resources import PhysicalGPU, Memory
    wait = PhysicalGPU.wait_clean;owned=model is None
    if shared:
        # Diagnostic only: foreign activity is recorded, never an accepted speed.
        PhysicalGPU.wait_clean = lambda self: None
    try:
        if owned:model = build(config['model'], config, output / 'runtime')
    finally:
        PhysicalGPU.wait_clean = wait
    recovery = None
    try:
        policy = policy_book(config)['priority_s25']
        variant = model.prepare_variant(policy)
        if include_affine:recovery = AffinePolicy(model, policy, Path(config['coefficients_directory']) / 'priority_s25/coefficients.pt')
        memory = Memory(model)
        condition = model.pilot[0]
        rows = []
        select_original = model.pt.select
        select_events = []
        def select(step):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.profiler.record_function('research.parameter_select'):
                select_original(step)
            end.record()
            select_events.append((start, end))
        model.pt.select = select
        routes=[('dense',None,None),('raw25',variant,None)]
        if recovery:routes.append(('affine25',variant,recovery))
        for name, v, r in routes:
            model.selector = r
            for _ in range(2):
                if r:r.reset()
                model.generate(condition, v)
            select_events.clear()
            if r:r.reset()
            memory.begin()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
                start = time.perf_counter()
                _, result = model.generate(condition, v, timing=True)
                wall = time.perf_counter() - start
            result.update(method=name, wall_seconds=wall, foreign=sorted(model.physical.pids()-{os.getpid()}), own_peak=memory.finish(), select_ms=sum(a.elapsed_time(b) for a,b in select_events))
            events = p.key_averages()
            result['cpu'] = sorted([dict(name=e.key, count=e.count, self_us=e.self_cpu_time_total, total_us=e.cpu_time_total) for e in events], key=lambda x:x['self_us'], reverse=True)[:35]
            result['device'] = sorted([dict(name=e.key, count=e.count, self_us=e.self_device_time_total, total_us=e.device_time_total) for e in events if e.device_time_total], key=lambda x:x['self_us'], reverse=True)[:25]
            rows.append(result)
            write(output/'profile.json', dict(pid=os.getpid(), diagnostic_only=True, shared=shared, rows=rows))
            print(name, result['denoiser_ms'], 'select', result['select_ms'], flush=True)
    finally:
        if 'select_original' in locals():model.pt.select=select_original
        if recovery:recovery.close()
        model.selector=None
        if owned:model.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--shared-diagnostic', action='store_true')
    args=parser.parse_args()
    import subprocess
    uuid=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    os.environ.update(SDACC_PHYSICAL_GPU=str(args.gpu),CUDA_VISIBLE_DEVICES=uuid)
    from .config import load
    profile(load(args.config), Path(args.output), args.shared_diagnostic)


if __name__=='__main__':main()
