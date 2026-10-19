"""Combine new64 metrics with matching accepted64 controls; never regenerate."""
import json
import os
import statistics
from pathlib import Path
try:
    from .run import ROOT, OUT, read, write, sha
except ImportError:
    from run import ROOT, OUT, read, write, sha


def finish():
    plan=read(OUT/'PLAN.json');records=[];summaries=[]
    # Controller dependencies require actual successful exits. Keep their PIDs.
    exits={}
    for tid in plan['task_chain'][:-1]:
        e=read(ROOT/'jobs'/tid/'exit.json')
        assert e.get('actual_waitpid') and e['returncode']==0
        exits[tid]=e
    for worker in plan['workers'].values():
        reference=read(worker['reference']);new=read(Path(worker['output'])/'quality_complete.json')
        assert new['complete'];fresh={(x['method'],x['condition']):x for x in new['records']}
        alpha_dense=None
        if worker['model']=='pixart-alpha':
            q=read(OUT/'reuse/pixart-alpha/quality_complete.json');assert q['complete']
            alpha_dense={x['condition']:x for x in q['records']}
        for cohort in worker['cohorts']:
            for method in cohort['methods']:
                alias=cohort['aliases'][method];selected=[]
                for condition in worker['conditions']:
                    cid=condition['id']
                    if alias['kind']=='existing64':
                        item=reference['images_and_metrics'][alias['method']][cid]
                        row=item['metric'] or alpha_dense[cid]
                        metric_source=item['metric_source'] or str(OUT/'reuse/pixart-alpha/quality_complete.json')
                        reused=True
                    else:
                        row=fresh[alias['method'],cid]
                        metric_source=str(Path(worker['output'])/'quality_complete.json')
                        reused=False
                    dense=reference['images_and_metrics']['dense_default'][cid]['image']['sha']
                    image=row.get('image_SHA',row.get('image_sha'))
                    if row.get('Dense_image_SHA'):assert row['Dense_image_SHA']==dense
                    record=dict(cohort=cohort['name'],model=worker['model'],method=method,condition=cid,
                        source_method=row['method'],image_SHA=image,Dense_image_SHA=dense,
                        generation_PID=row.get('generation_pid'),metric_PID=row.get('metric_pid'),
                        metric_source=metric_source,reused_accepted64=reused,
                        formal_PID=cohort['formal_PID'],formal_source=cohort['formal'],
                        denoiser_seconds_including_affine=cohort['timing_seconds'][method],
                        **{k:row[k] for k in ('psnr','wssim','clip','imagereward')})
                    selected.append(record);records.append(record)
                assert len(selected)==64 and len({x['condition'] for x in selected})==64
                summaries.append(dict(cohort=cohort['name'],model=worker['model'],method=method,samples=64,
                    denoiser_seconds_including_affine=cohort['timing_seconds'][method],
                    **{k:statistics.mean(x[k] for x in selected) for k in ('psnr','wssim','clip','imagereward')}))
    assert len(records)==57*64 and len(summaries)==57
    result=dict(complete=True,scientific_64_complete=True,report_PID=os.getpid(),samples=64,
        rows=records,summary=summaries,plan_SHA=sha(OUT/'PLAN.json'),actual_successful_jobs=exits,
        retimed=False,calibration_repeated=False,schedule_refit=False,
        quality_scope='Frozen original six pilot cohorts extended to independent64; controls reuse accepted64.',
        affine_time_scope='Original CUDA denoiser median including affine; not isolated fit cost.',
        old_4pilot_preserved=str(ROOT/'quality/pilot_missing/quality_complete.json'))
    write(OUT/'quality_complete.json',result)
    lines=['# 原叠加实验64张质量补测','',
        '每种配置64个独立评价输入；与100cal和4pilot分离，沿用原调度/银行/仿射，不重新拟合。已有64张Dense/Sparse及指标按原PID复用。新方法各图只生成一次，未重复测速。耗时为原CUDA去噪中位数（含仿射），不是独立仿射拟合耗时。','']
    for group in dict.fromkeys(x['cohort'] for x in summaries):
        lines+=['## '+group,'','| 配置 | 去噪耗时/s（含仿射） | PSNR/dB | Gaussian wSSIM | CLIP ViT-B/16 | ImageReward v1.0 |','|---|---:|---:|---:|---:|---:|']
        for row in summaries:
            if row['cohort']==group:
                lines.append('| '+row['method']+' | '+' | '.join(f'{row[k]:.5f}' for k in
                    ['denoiser_seconds_including_affine','psnr','wssim','clip','imagereward'])+' |')
        lines.append('')
    (OUT/'QUALITY_TABLES.md').write_text('\n'.join(lines)+'\n')
    combined=[x for x in summaries if '_s' in x['method'] and not x['method'].startswith('sparse_')]
    assert len(combined)==20
    lines=['# 四档叠加方法：64张质量','',
        '每行64图均值；原加速比及去噪时间不重测。全部控制和逐图PID/来源见QUALITY_TABLES.md及quality_complete.json。','']
    for group in dict.fromkeys(x['cohort'] for x in combined):
        lines+=['## '+group,'','| 稀疏度 | 去噪耗时/s（含仿射） | PSNR/dB | Gaussian wSSIM | CLIP ViT-B/16 | ImageReward v1.0 |','|---|---:|---:|---:|---:|---:|']
        for row in combined:
            if row['cohort']==group:
                lines.append('| '+row['method'].rsplit('_s',1)[1]+'% | '+' | '.join(f'{row[k]:.5f}' for k in
                    ['denoiser_seconds_including_affine','psnr','wssim','clip','imagereward'])+' |')
        lines.append('')
    (OUT/'QUALITY_COMBINATIONS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':finish()
