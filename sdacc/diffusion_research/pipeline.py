"""Pinned eager model adapters for the SODA/OBS study (no VAE in profiling)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
import types

import torch


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    directory: str
    revision: str
    resolution: int
    steps: int
    guidance: float
    pipeline: str
    max_sequence_length: int = 120


SPECS = {
    "dit256": ModelSpec("dit256", "dit", "DiT-XL-2-256", "eab87f77abd5aef071a632f08807fbaab0b704d0", 256, 50, 4., "DiTPipeline"),
    "dit512": ModelSpec("dit512", "dit", "DiT-XL-2-512", "101a3d462b22d64c4afdd4d0c8c59a2c0b961b99", 512, 50, 4., "DiTPipeline"),
    "pixart-alpha": ModelSpec("pixart-alpha", "pixart", "PixArt-XL-2-256x256", "5943a94f78ebef0bcd04b6ea85d888220d696836", 256, 20, 4.5, "PixArtAlphaPipeline"),
    "pixart-sigma": ModelSpec("pixart-sigma", "pixart", "PixArt-Sigma-XL-2-1024-MS", "e102b3591cc82e97071b8b4cb90d834d0c487207", 1024, 20, 4.5, "PixArtSigmaPipeline", 300),
    "sd3-medium": ModelSpec("sd3-medium", "sd3", "stable-diffusion-3-medium-diffusers", "ea42f8cef0f178587cf766dc8129abd379c90671", 512, 25, 7., "StableDiffusion3Pipeline", 256),
    "sd3.5-large": ModelSpec("sd3.5-large", "sd3", "stable-diffusion-3.5-large", "ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f", 512, 25, 3.5, "StableDiffusion3Pipeline", 256),
    "flux-dev": ModelSpec("flux-dev", "flux", "FLUX.1-dev", "7a2a78fe44f3a51c8ab95f1b6de4de8148def560", 1024, 50, 3.5, "FluxPipeline", 512),
    "glm-image": ModelSpec("glm-image", "glm-image", "GLM-Image", "", 1024, 50, 1.5, "GlmImagePipeline", 2048),
}


def serialized_config(value):
    """Preserve scheduler infinities explicitly without emitting invalid JSON."""
    if isinstance(value, dict):
        return {k: serialized_config(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serialized_config(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def weights_identity_equal(left, right) -> bool:
    """Compare verified content, while keeping stat signatures as cache keys.

    File mtimes invalidate a local hash cache but do not identify model weights
    across hosts. Both manifests must contain the same complete hash/size map.
    """
    if left is None or right is None:
        return left is None and right is None
    def content(value):
        if not isinstance(value, dict) or value.get('version_source') != 'content_sha256':
            return None
        hashes = value.get('sha256', {})
        signatures = value.get('file_signature', {})
        if not hashes or set(hashes) != set(signatures):
            return None
        sizes = {}
        for name, digest in hashes.items():
            entry = signatures[name]
            if (not isinstance(digest, str) or len(digest) != 64 or
                    any(c not in '0123456789abcdef' for c in digest) or
                    not isinstance(entry, (list, tuple)) or len(entry) != 2 or
                    not isinstance(entry[0], int) or entry[0] <= 0):
                return None
            sizes[name] = entry[0]
        return hashes, sizes
    a, b = content(left), content(right)
    return a is not None and b is not None and a == b


def checkpoint_receipt(path: Path, spec: ModelSpec, weight_manifest_path=None) -> dict:
    index = json.loads((path / "model_index.json").read_text())
    if index.get("_class_name") != spec.pipeline:
        raise ValueError(f"wrong checkpoint pipeline: {index.get('_class_name')} vs {spec.pipeline}")
    shards = []
    for shard_index in path.rglob("*.safetensors.index.json"):
        mapping = json.loads(shard_index.read_text())["weight_map"]
        for name in sorted(set(mapping.values())):
            shard = shard_index.parent / name
            if not shard.is_file() or shard.stat().st_size == 0:
                raise FileNotFoundError(shard)
            shards.append({"path": str(shard.relative_to(path)), "bytes": shard.stat().st_size})
    receipts = {}
    for name in ("DOWNLOAD_COMPLETE.json", "source_verification_receipt.json"):
        if (path / name).exists():
            receipts[name] = json.loads((path / name).read_text())
    if spec.revision and spec.revision not in str(path) + json.dumps(receipts):
        raise ValueError(f"checkpoint revision is not verified: {spec.revision}")
    config_files=set(path.rglob('*.json'))
    for component in index:
        folder=path/component
        if not component.startswith('_') and folder.is_dir():config_files.update(folder.rglob('*.json'))
    configs = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(config_files) if "config" in p.name or p.name == "model_index.json"}
    weights_identity = None
    if not spec.revision:
        # Legacy GLM folders have no download commit receipt. Bind their actual
        # content instead of inventing a revision or trusting an unchanged path.
        files = sorted([*path.rglob("*.safetensors"), *path.rglob("*.bin")])
        signature = {str(p.relative_to(path)): [p.stat().st_size,p.stat().st_mtime_ns] for p in files}
        cached = Path(weight_manifest_path) if weight_manifest_path else None
        if cached and cached.exists():
            previous = json.loads(cached.read_text())
            if previous["file_signature"] == signature:
                weights_identity = previous
        if weights_identity is None:
            digests = {}
            for p in files:
                h = hashlib.sha256()
                with p.open("rb") as stream:
                    while chunk := stream.read(8 << 20):
                        h.update(chunk)
                digests[str(p.relative_to(path))] = h.hexdigest()
            weights_identity = dict(file_signature=signature,sha256=digests,version_source="content_sha256")
            if cached:
                from .io import write_json
                write_json(cached,weights_identity)
    return {"checkpoint": str(path.resolve()), "spec": asdict(spec), "receipts": receipts,
            "configs_sha256": configs, "shards": shards, "weights_identity": weights_identity}


class Adapter:
    def __init__(self, spec: ModelSpec, checkpoint: Path, device="cuda", steps=None, weight_manifest_path=None,
                 cached_only=False):
        import diffusers
        self.spec, self.device = spec, torch.device(device)
        self.steps = steps or spec.steps
        self.receipt = checkpoint_receipt(checkpoint, spec, weight_manifest_path)
        supplied = {}
        if cached_only:
            components=json.loads((checkpoint/'model_index.json').read_text())
            supplied={name:None for name in ('text_encoder','text_encoder_2','text_encoder_3','vision_language_encoder') if name in components}
        self.pipe = getattr(diffusers, spec.pipeline).from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, local_files_only=True, **supplied)
        self.pipe.set_progress_bar_config(disable=True)
        if spec.family == "dit":
            from diffusers.image_processor import VaeImageProcessor
            self.pipe.image_processor = VaeImageProcessor(vae_scale_factor=8)
        self.model = self.pipe.transformer.eval()
        self.family = spec.family
        self.cfg_batched = self.family in {"dit", "pixart", "sd3"}
        self.calls_per_step = 2 if self.family == "glm-image" else 1
        # The checkpoint's scheduler/config is retained and recorded verbatim.
        scheduler_name = type(self.pipe.scheduler).__name__
        required = {"dit": "DDIMScheduler", "pixart": "DPMSolverMultistepScheduler",
                    "flux": "FlowMatchEulerDiscreteScheduler", "sd3": "FlowMatchEulerDiscreteScheduler", "glm-image": "FlowMatchEulerDiscreteScheduler"}
        if scheduler_name != required[self.family]:
            raise ValueError(f"unexpected scheduler {scheduler_name}")
        if self.family == "pixart" and self.pipe.scheduler.config.algorithm_type != "dpmsolver++":
            raise ValueError("PixArt requires DPM-Solver++")
        self.receipt["scheduler"] = serialized_config(dict(self.pipe.scheduler.config))

    def conditioning_gpu(self):
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3", "vision_language_encoder"):
            model = getattr(self.pipe, name, None)
            if model is not None:
                model.to(self.device).eval()

    def denoiser_gpu(self):
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3", "vision_language_encoder", "vae"):
            model = getattr(self.pipe, name, None)
            if model is not None:
                model.to("cpu")
        torch.cuda.empty_cache()
        self.model.to(self.device)

    @torch.inference_mode()
    def encode(self, condition):
        p, s = self.pipe, self.spec
        if self.family == "dit":
            return {}
        prompt = condition["prompt"]
        if self.family == "pixart":
            emb, mask, neg, negmask = p.encode_prompt(
                prompt, do_classifier_free_guidance=True, device=self.device,
                max_sequence_length=s.max_sequence_length)
            data = dict(prompt_embeds=emb, prompt_attention_mask=mask,
                        negative_prompt_embeds=neg, negative_prompt_attention_mask=negmask)
        elif self.family == "sd3":
            emb,neg,pooled,negpooled=p.encode_prompt(prompt,prompt_2=None,prompt_3=None,device=self.device,
                do_classifier_free_guidance=True,max_sequence_length=s.max_sequence_length)
            def valid(text):
                masks=[t(text,padding="max_length",max_length=77,truncation=True,return_tensors="pt").attention_mask.bool() for t in (p.tokenizer,p.tokenizer_2)]
                t5=p.tokenizer_3(text,padding="max_length",max_length=s.max_sequence_length,truncation=True,return_tensors="pt").attention_mask.bool()
                return torch.cat((masks[0]|masks[1],t5),dim=1)
            data=dict(prompt_embeds=emb,negative_prompt_embeds=neg,pooled_prompt_embeds=pooled,
                negative_pooled_prompt_embeds=negpooled,text_mask=valid(prompt),negative_text_mask=valid(""))
        elif self.family == "flux":
            emb, pooled, _ = p.encode_prompt(prompt, prompt_2=None, device=self.device,
                                           max_sequence_length=s.max_sequence_length)
            mask = p.tokenizer_2(prompt, padding="max_length", max_length=s.max_sequence_length,
                                 truncation=True, return_tensors="pt").attention_mask
            data = dict(prompt_embeds=emb, pooled_prompt_embeds=pooled, text_mask=mask)
        else:
            # Native GLM encodes glyphs (quoted text); AR priors carry prompt semantics.
            prior, prior_images, grid = p.generate_prior_tokens(
                prompt, height=s.resolution, width=s.resolution, device=self.device,
                generator=torch.Generator(device=self.device).manual_seed(condition["seed"]))
            if prior_images is not None or grid is not None:
                raise ValueError("unexpected image conditioning")
            emb, neg = p.encode_prompt(prompt, device=self.device, dtype=torch.bfloat16,
                                      max_sequence_length=s.max_sequence_length)
            glyphs = p.get_glyph_texts([prompt])[0] or [""]
            ids = p.tokenizer(glyphs, max_length=s.max_sequence_length, truncation=True).input_ids
            added = (len(ids)+1) % 2
            valid = [flag for tokens in ids for flag in ([False]*added + [True]*len(tokens))]
            if len(valid) != emb.shape[1]:
                raise RuntimeError("GLM glyph mask does not match native encoding")
            data = dict(prompt_embeds=emb, negative_prompt_embeds=neg, prior_token_ids=prior,
                        text_mask=torch.tensor([valid]), negative_text_mask=torch.ones(neg.shape[:2],dtype=torch.bool))
        return {k: v.detach().cpu() for k, v in data.items()}

    def combine(self, payloads):
        if not payloads or not payloads[0]:
            return {}, None
        # GLM's native single-prompt glyph streams have no padding. Keep their
        # potentially different conditional/unconditional lengths unchanged.
        if self.family == "glm-image" and any(
                p["prompt_embeds"].shape != payloads[0]["prompt_embeds"].shape or
                p["negative_prompt_embeds"].shape != payloads[0]["negative_prompt_embeds"].shape for p in payloads):
            raise ValueError("GLM batch must share glyph lengths to preserve batch-1 inputs")
        data = {k: torch.cat([p[k] for p in payloads]).to(self.device) for k in payloads[0]}
        if self.family == "pixart":
            mask = torch.cat([data["negative_prompt_attention_mask"], data["prompt_attention_mask"]])
        elif self.family == "sd3":
            mask=torch.cat((data.pop("negative_text_mask"),data.pop("text_mask")))
        elif self.family == "flux":
            mask = data.pop("text_mask")
        else:
            mask = {0: data.pop("text_mask"), 1: data.pop("negative_text_mask")}
        return data, mask

    @torch.inference_mode()
    def generate(self, conditions, data=None, trajectory=False, return_cpu=True):
        data = data or {}
        generators = [torch.Generator(device=self.device).manual_seed(c["seed"]) for c in conditions]
        history = []
        # Observe scheduler outputs without altering arguments or return values.
        original_step = self.pipe.scheduler.step
        @wraps(original_step)
        def observe_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            if trajectory:
                latents = result.prev_sample if hasattr(result, "prev_sample") else result[0]
                if self.family == "dit":
                    latents = latents[:len(conditions)]
                history.append(latents.detach().cpu())
            return result
        self.pipe.scheduler.step = observe_step
        try:
            if self.family == "dit":
                latents = self._dit_latents(conditions, generators)
            else:
                args = dict(prompt=None, height=self.spec.resolution, width=self.spec.resolution,
                            num_inference_steps=self.steps, guidance_scale=self.spec.guidance,
                            generator=generators, output_type="latent",
                            max_sequence_length=self.spec.max_sequence_length, **data)
                if self.family == "pixart":
                    args["negative_prompt"] = None
                if self.family == "glm-image":
                    # Cached native streams can have unequal sequence lengths.
                    # check_inputs compares supplied pos/neg shapes, whereas the
                    # native encoder produces unequal shapes inside __call__.
                    # Return both cached streams at that same native encode point.
                    negative = args.pop("negative_prompt_embeds")
                    original_encode = self.pipe.encode_prompt
                    self.pipe.encode_prompt = lambda *a, **k: (data["prompt_embeds"], negative)
                    try:
                        latents = self.pipe(**args).images
                    finally:
                        self.pipe.encode_prompt = original_encode
                else:
                    latents = self.pipe(**args).images
        finally:
            self.pipe.scheduler.step = original_step
        if trajectory and len(history) != self.steps:
            raise RuntimeError(f"scheduler coverage: {len(history)} vs {self.steps}")
        return (latents.detach().cpu() if return_cpu else latents.detach()), history

    def _dit_latents(self, conditions, generators):
        from diffusers.utils.torch_utils import randn_tensor
        p, batch = self.pipe, len(conditions)
        size, channels = p.transformer.config.sample_size, p.transformer.config.in_channels
        latents = randn_tensor((batch, channels, size, size), generator=generators,
                               device=self.device, dtype=p.transformer.dtype)
        current = torch.cat([latents] * 2)
        labels = torch.tensor([c["class_id"] for c in conditions], device=self.device)
        labels = torch.cat([labels, torch.full_like(labels, 1000)])
        p.scheduler.set_timesteps(self.steps)
        for timestep in p.scheduler.timesteps:
            current = torch.cat([current[:batch], current[:batch]])
            current = p.scheduler.scale_model_input(current, timestep)
            t = torch.as_tensor(timestep, device=self.device).reshape(1).expand(current.shape[0])
            prediction = p.transformer(current, timestep=t, class_labels=labels).sample
            eps, rest = prediction[:, :channels], prediction[:, channels:]
            cond, uncond = eps.chunk(2)
            guided = uncond + self.spec.guidance * (cond - uncond)
            prediction = torch.cat([torch.cat([guided, guided]), rest], dim=1)
            if p.transformer.config.out_channels // 2 == channels:
                prediction = prediction.split(channels, dim=1)[0]
            current = p.scheduler.step(prediction, timestep, current).prev_sample
        return current[:batch]

    @torch.inference_mode()
    def validate_dit_native(self, condition):
        if self.family != "dit":
            return None
        latent, _ = self.generate([condition])
        # Native DiT always decodes; capture its VAE input and stop before decode.
        class Captured(Exception):
            pass
        original_decode = self.pipe.vae.decode
        received = []
        def capture(x, *args, **kwargs):
            received.append(x.detach().cpu())
            raise Captured()
        self.pipe.vae.decode = capture
        try:
            try:
                self.pipe(class_labels=[condition["class_id"]], guidance_scale=self.spec.guidance,
                          num_inference_steps=self.steps,
                          generator=torch.Generator(device=self.device).manual_seed(condition["seed"]))
            except Captured:
                pass
        finally:
            self.pipe.vae.decode = original_decode
        expected = latent / self.pipe.vae.config.scaling_factor
        if len(received) != 1 or not torch.equal(expected, received[0]):
            raise RuntimeError("custom DiT latent loop differs from native pipeline")
        return {"native_vae_input_bitwise_equal": True, "steps": self.steps}

    @torch.inference_mode()
    def decode(self, latents):
        p = self.pipe
        self.model.to("cpu")
        torch.cuda.empty_cache()
        p.vae.to(self.device)
        batch = latents.to(self.device, p.vae.dtype)
        if self.family == "flux":
            batch = p._unpack_latents(batch, self.spec.resolution, self.spec.resolution, p.vae_scale_factor)
            batch = batch / p.vae.config.scaling_factor + p.vae.config.shift_factor
        elif self.family == "sd3":
            batch = batch / p.vae.config.scaling_factor + p.vae.config.shift_factor
        elif self.family == "glm-image":
            mean = torch.tensor(p.vae.config.latents_mean, device=self.device, dtype=batch.dtype).view(1, -1, 1, 1)
            std = torch.tensor(p.vae.config.latents_std, device=self.device, dtype=batch.dtype).view(1, -1, 1, 1)
            batch = batch * std + mean
        else:
            batch = batch / p.vae.config.scaling_factor
        decoded = p.vae.decode(batch, return_dict=False)[0]
        return p.image_processor.postprocess(decoded, output_type="pil")
