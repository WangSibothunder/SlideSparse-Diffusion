# Optional research integrations

```bash
python -m sdacc.diffusion_research.integrations
```

This prints the method catalog without importing Torch or launching work. The main six-model runtime enables no plugin automatically.

| Integration | Implementation | Scope |
| --- | --- | --- |
| Plain prediction cache + CEM | `../cache.py`, `../cache_pilot.py` | Shared cached-prediction gate outside compiled calls; not DuCa feature caching |
| Alpha DuCa + CEM + Sparse | `../models/pixart_duca.py`, `../duca_pilot.py` | DuCa-derived internal feature cache, GPL-3.0; original studies retain their sources and results |
| Sigma CEM/DuCa | `sigma_cem.py` | Lazy bridge to the separately preserved `sigma_orthogonal.cem_api` package |
| FLUX SpargeAttn | `../models/flux_sparge*.py`, `../sparge_pilot.py`, `sparge_native/` | Experimental native joint-attention bridge; negative full-attention results are not relabeled as accepted acceleration |
| CacheQuant | `cachequant/`, `../models/sigma_cachequant.py` | Local INT32 arithmetic/OBS research adapters; on hold, no complete model speed/quality claim |
| 64-image backfill | `quality64/` | Archive-oriented driver that requires the original study plan and frozen external cohort sources |

Historical six-cohort evaluation completed 57 configurations × 64 unique evaluation IDs. The published package contains source and method metadata; it does not include private study plans, source archives, model assets, generation logs or per-image data. Historical results remain attributable to their actual source/PID rather than this public export.

## External dependencies

Sigma's bridge requires `SDACC_ORTHOGONAL_ROOT` pointing to the retained original research bundle with `research/cem_duca/src/sigma_orthogonal/cem_api.py`. Importing the bridge does not load that bundle. A missing path is an explicit error, not a fallback to a private server directory.

SpargeAttn needs the pinned upstream Python package `spas_sage_attn` and the explicitly built `sdacc_sparge_sm80_current_stream` extension. The included native source is specialized to SM80/head128. Build, if requested, with:

```bash
SPARGE_STREAM_BUILD=/path/to/local/build \
  python -m sdacc.diffusion_research.integrations.sparge_native.build
```

Add that build directory and the pinned upstream Python package to your local Python path. Neither is built or downloaded by importing the main library. The latest fixed-QKV layout correction was numerically correct in its original checks, but complete attention remained slower than SDPA, so no repaired four-rate sweep is claimed.

CacheQuant uses an externally supplied native quantizer class and the original calibration/provenance assets. This repository does not redistribute that upstream source or turn its hold status into an active experiment.

Archive backfill entrypoints require the original relative study layout and plan. Use the core `run --stage decode/score` interfaces for core-model outputs; no artifact or condition set is fabricated by package installation.
