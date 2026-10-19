# Local assets and reuse

The public repository contains source, static dispatch metadata and path templates. It excludes checkpoints, prepared banks, curvature/activations, coefficients, latents, per-image outputs, internal logs and server configuration. These remain at their original locations in the research installation.

Use an editable installation for the documented native build. `scripts/build_native.py` explicitly builds the bundled SDK into `sdacc/w8a8_slidesparse/native_sm80/` for the current Torch/CUDA ABI. It preserves the original source/dependency checks and writes a build receipt; no prebuilt binary is distributed. The research `library` setting points to the source snapshot containing `SOURCE_BINDING.json` and the locally built `native_sm80/` directory.

Configuration paths resolve relative to the JSON file. Environment variables and home-directory expansion are supported. The supplied templates use local `models/` and `data/<model>/` directories; replace them with existing artifacts as appropriate.

## Conditions

Text-conditioned models use a `conditions.json` with `calibration`, `pilot`, `evaluation` lists. Each entry needs its preserved unique `id`, `prompt` and `seed`; retain any model-specific fields used by the original encoder. DiT uses `class_id` and native class/null CFG instead. Text tensors are stored under `conditioning_directory` using the condition ID. Keep the original condition IDs and seeds when reusing banks and results.

The defaults for complete studies are 100/4/64. Do not infer that simply copying a JSON template provides those data or that a dataset is bundled. For a new study, supply the intended complete fixed condition set; for an existing study, reuse its accepted split and receipts.

## Banks and policies

- PixArt/DiT/MMDiT use the resident-bank format consumed by `banks.load_resident_banks`.
- FLUX uses its existing packed manifest, slot plan, memory plan and mixed attention/FFN format. The public template points to placeholders for those artifacts; it does not synthesize historical bank data.
- `policy_file` fixes the whole-projection sparse choices. `coefficients_directory` holds each policy's own completed calibration and affine coefficients. Never substitute another model's bank or coefficients.
- Completed-stage markers enable reuse in the study driver. Passing `--stage study` authorizes its explicit missing preparation, calibration, timing and quality stages; use `infer` or `benchmark` when only those operations are intended.

## Pinned metric assets

The score path retains the original offline metric loaders and SHA checks. Installing a metric package alone does not provide these pinned assets. Under `metric_assets`, supply:

```text
soda_obs_sensitivity_20261001/metric_weights/
  models--timm--vit_base_patch16_clip_224.openai/snapshots/<revision>/
    open_clip_pytorch_model.bin
obs_rank_schedule_20261006/diagnostics/sigma_imagereward_cpu_20261008/
  official_source_manifest.json
  official/ImageReward/...
  assets/ImageReward.pt
  assets/med_config.json
  assets/tokenizer/...
  deps/                       # optional preserved offline Python dependencies
```

ImageReward source revision: `78f8051059b34fef8e0032ccb40561c23a2ddbbb`. Checkpoint SHA256: `d71dc8d4912bc633c06ab8ef1ef6a81e8199d4369f14b47b29baf1d19de561f6`. CLIP checkpoint SHA256: `52bdba61c0cb19e034fc891b15e6f15291d4c06dcceae69b772776c7bc9f8b35`.

Those relative historical directory names remain part of the existing metric asset contract. No local username, server address or credentials are required. Download/access the model and metric assets from their upstream projects under their own terms; credentials stay in the local environment.

## Optional research integrations

See `sdacc/diffusion_research/integrations/README.md`. The main six-model registry does not automatically load any plugin. Some archive-oriented entrypoints require separately retained sources/configurations and are not self-contained fresh experiments.
