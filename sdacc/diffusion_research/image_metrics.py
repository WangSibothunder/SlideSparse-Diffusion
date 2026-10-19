"""One stock VAE and image metric path; no model-specific metric copies."""
import os,sys,json
from pathlib import Path
from .io import read,write,sha
CODE=Path(__file__).resolve().parent

def memory(torch):
    from .resources import PhysicalGPU,Memory
    class Owner:pass
    owner=Owner();owner.physical=PhysicalGPU(os.environ['SDACC_PHYSICAL_GPU'])
    witness=Memory(owner);witness.external=0;witness.sample();result=witness.finish();owner.physical.close();return result


def decode(config,output,generated):
    import torch,types
    from dataclasses import replace
    from .pipeline import Adapter,SPECS
    from diffusers import AutoencoderKL,FluxPipeline
    from diffusers.image_processor import VaeImageProcessor
    spec=replace(SPECS[config['spec_name']],**config.get('spec',{}))
    with torch.inference_mode():
        vae=AutoencoderKL.from_pretrained(config['checkpoint'],subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).eval().cuda()
        factor=2**(len(vae.config.block_out_channels)-1);decoder=Adapter.__new__(Adapter);decoder.family=spec.family;decoder.device=torch.device('cuda');decoder.spec=spec;decoder.model=torch.nn.Identity()
        decoder.pipe=types.SimpleNamespace(vae=vae,vae_scale_factor=factor,_unpack_latents=FluxPipeline._unpack_latents,image_processor=VaeImageProcessor(vae_scale_factor=factor*(2 if spec.family=='flux' else 1)))
        sources={(r['method'],r['condition']):r for r in generated['latent_sources']};rows=[]
        for method in generated['methods']:
            for first in range(0,len(generated['conditions']),4):
                conditions=generated['conditions'][first:first+4];missing=[]
                for c in conditions:
                    path=output/'images'/method/(c['id']+'.png')
                    if path.exists():
                        row=read(path.with_suffix('.json'))
                        if row['latent_sha']!=sources[method,c['id']]['sha'] or sha(path)!=row['sha']:raise ValueError('committed image differs')
                        rows.append(row)
                    else:missing.append(c)
                if not missing:continue
                packets=[sources[method,c['id']] for c in missing]
                for packet in packets:
                    if sha(packet['path'])!=packet['sha']:raise ValueError('latent changed')
                latents=torch.cat([torch.load(a['path'],weights_only=True,map_location='cpu')['latent'] for a in packets])
                for c,packet,image in zip(missing,packets,decoder.decode(latents)):
                    if image.size!=(spec.resolution,spec.resolution):raise ValueError('unexpected VAE resolution')
                    path=output/'images'/method/(c['id']+'.png');path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');image.save(tmp,format='PNG');os.replace(tmp,path)
                    row=dict(method=method,condition=c['id'],path=str(path),sha=sha(path),latent_sha=packet['sha'],decode_pid=os.getpid(),generation_pid=packet['generation_pid']);write(path.with_suffix('.json'),row);rows.append(row)
                write(output/'decode_progress.json',dict(completed=len(rows),pid=os.getpid(),memory=memory(torch)))
        write(output/'decode_complete.json',dict(complete=True,pid=os.getpid(),generation_SHA=sha(output/'generation_complete.json'),images=rows,stock_VAE=True))

def score(root, output, generated, config):
    ASSETS = Path(config['metric_assets'])
    # Preserve the measured official scalar environment.
    sys.path.insert(0, str(ASSETS / 'obs_rank_schedule_20261006/diagnostics/sigma_imagereward_cpu_20261008/deps'))
    import transformers
    assert transformers.__version__ == '4.30.2'
    import torch
    import numpy as np
    import open_clip
    from PIL import Image
    from skimage.metrics import structural_similarity
    from .metric_assets import load_ir, CLIP_SHA, interval
    from .scalar_metric_contract import official_singletons
    decoded = json.loads((output / 'decode_complete.json').read_text())
    assert decoded['complete'] and decoded['generation_SHA'] == sha(output / 'generation_complete.json')
    dense_dir = Path(config['dense_root'])
    dense_generation = json.loads((Path(config.get('dense_generation', dense_dir / 'complete.json'))).read_text())
    dense_decode = json.loads((Path(config.get('dense_decode', dense_dir / 'decode_complete.json'))).read_text())
    assert dense_generation['complete'] and dense_decode['complete']
    assert dense_generation['conditions'] == generated['conditions']
    dense_images = {r['condition']: r for r in dense_decode['images'] if r['method'] == 'dense_default'}
    for image in dense_images.values():
        assert sha(image['path']) == image['sha']
    images = {(r['method'], r['condition']): r for r in decoded['images']}
    weights = list((ASSETS / 'soda_obs_sensitivity_20261001/metric_weights').glob(
        'models--timm--vit_base_patch16_clip_224.openai/snapshots/*/open_clip_pytorch_model.bin'))
    assert len(weights) == 1 and sha(weights[0]) == CLIP_SHA
    cfg = open_clip.get_pretrained_cfg('ViT-B-16', 'openai')
    clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16',
        pretrained=str(weights[0]), force_quick_gelu=True, image_mean=cfg['mean'],
        image_std=cfg['std'], image_interpolation=cfg['interpolation'],
        image_resize_mode=cfg['resize_mode'])
    clip = clip.cuda().float().eval()
    tokenize = open_clip.get_tokenizer('ViT-B-16')
    reward, reward_receipt = load_ir(ASSETS, torch.device('cuda'))
    records, audits = [], []
    partial = output / 'quality_partial.json'
    if partial.exists():
        prior = json.loads(partial.read_text())
        assert prior['generation_SHA'] == sha(output / 'generation_complete.json')
        assert prior['decode_SHA'] == sha(output / 'decode_complete.json')
        records, audits = prior['records'], prior['ImageReward_scalar_audits']
    completed = {(r['method'], r['condition']) for r in records}
    assert len(completed) == len(records)
    norm = lambda x: torch.nn.functional.normalize(x.float(), dim=-1)
    with torch.inference_mode():
        for method in generated['methods']:
            for first in range(0, 64, 4):
                conditions = [c for c in generated['conditions'][first:first + 4]
                              if (method, c['id']) not in completed]
                if not conditions:
                    continue
                paths = [images[method, c['id']]['path'] for c in conditions]
                for c, path in zip(conditions, paths):
                    assert sha(path) == images[method, c['id']]['sha']
                batch = [Image.open(path).convert('RGB') for path in paths]
                features = norm(clip.encode_image(torch.stack([preprocess(im) for im in batch]).cuda()))
                texts = norm(clip.encode_text(tokenize([c['prompt'] for c in conditions]).cuda()))
                rewards, audit = official_singletons(reward, conditions, batch, paths, torch, 'cuda',
                    verify=config.get('verify_imagereward_scalar', True))
                audits.extend(audit)
                for index, (condition, image) in enumerate(zip(conditions, batch)):
                    target = Image.open(dense_images[condition['id']]['path']).convert('RGB')
                    a = np.asarray(image, dtype=np.float32) / 255
                    d = np.asarray(target, dtype=np.float32) / 255
                    mse = float(np.mean((a - d) ** 2))
                    row = dict(method=method, condition=condition['id'], metric_pid=os.getpid(),
                        generation_pid=images[method, condition['id']]['generation_pid'],
                        image_SHA=images[method, condition['id']]['sha'],
                        Dense_image_SHA=dense_images[condition['id']]['sha'],
                        psnr=120. if mse == 0 else float(-10 * np.log10(mse)), identical=mse == 0,
                        wssim=float(structural_similarity(d, a, channel_axis=2, data_range=1,
                            win_size=11, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)),
                        clip=float(features[index] @ texts[index]), imagereward=float(rewards[index]))
                    assert np.isfinite([row[k] for k in ('psnr', 'wssim', 'clip', 'imagereward')]).all()
                    records.append(row)
                    completed.add((method, condition['id']))
                write(partial, dict(generation_SHA=sha(output / 'generation_complete.json'),
                      decode_SHA=sha(output / 'decode_complete.json'), records=records,
                      ImageReward_scalar_audits=audits, memory=memory(torch)))
    assert len(records) == len(generated['methods']) * len(generated['conditions'])
    summaries = {}
    for method in generated['methods']:
        rows = [r for r in records if r['method'] == method]
        assert len(rows) == 64 and {r['condition'] for r in rows} == {c['id'] for c in generated['conditions']}
        summaries[method] = {k: interval([r[k] for r in rows])
                             for k in ('psnr', 'wssim', 'clip', 'imagereward')}
    write(output / 'quality_complete.json', dict(complete=True, pid=os.getpid(),
          source_manifest_SHA=sha(CODE / 'SOURCE_ORIGINS.json'),
          records=records, summary=summaries, ImageReward_scalar_audits=audits,
          ImageReward_independent_verification=config.get('verify_imagereward_scalar', True),
          metric_assets=dict(clip=CLIP_SHA, imagereward=reward_receipt,
                             transformers=transformers.__version__),
          Dense_generation_SHA=sha(Path(config.get('dense_generation', dense_dir / 'complete.json'))),
          Dense_decode_SHA=sha(Path(config.get('dense_decode', dense_dir / 'decode_complete.json'))),
          generation_SHA=sha(output / 'generation_complete.json'),
          decode_SHA=sha(output / 'decode_complete.json'),
          evaluation_used_for_selection=False, evaluation_used_for_fit=False))
