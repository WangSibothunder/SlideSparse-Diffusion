# Third-party sources

The combined source distribution is GPL-3.0. Existing third-party files retain their licenses and copyright headers.

| Component | Source / revision | Distribution |
| --- | --- | --- |
| DuCa adapter | [Shenyi-Z/DuCa](https://github.com/Shenyi-Z/DuCa), `0a7151b7bfea3822295e8d29f26cd2e10756670c` | Derived PixArt adapter in `models/pixart_duca.py`; GPL-3.0 retained |
| SpargeAttn headers | [thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn), `ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a` | Selected SM80 native sources; Apache-2.0 retained |
| SlideSparse reference | [bcacdwk/vllmbench](https://github.com/bcacdwk/vllmbench) | Reference license retained; project-authored W8A8 backend is bundled, not the entire upstream package |
| CUTLASS | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass), `0b55a2f691d69981583568fd9eb69687b1f0de8a` | External build dependency; BSD license and dependency SHA manifest bundled |
| NVML Python bindings | NVIDIA, copyright 2011–2023 | `_nvml.py`, original BSD notice retained |
| CEM | [leaves162/CEM](https://github.com/leaves162/CEM), `ffa2a615ac7567a0103d53c8e8baa2b54898eb3c` | Referenced method and optional bridge; upstream source not redistributed |
| CacheQuant | [BienLuky/CacheQuant](https://github.com/BienLuky/CacheQuant), `9470ddf1bac09337ce3281f797357ac35997a358` | Project-authored arithmetic/OBS adapters; original upstream quantizer source is external, not redistributed |

The local adapters preserve model-native behavior using Diffusers and Transformers APIs. Those packages, model weights, model licenses and optional separately installed plugin dependencies remain external.

Local modifications to redistributed SpargeAttn sources and the DuCa port are identified in `NOTICE`, native provenance and `provenance/EXPORT_ORIGINS.json`. The latter retains original SHA256 identities without exposing private machine paths. Source reorganization is not evidence of a new experiment or remeasurement.
