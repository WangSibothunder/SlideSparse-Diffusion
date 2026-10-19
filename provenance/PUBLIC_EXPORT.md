# Public source export

The reusable library is exported independently from the working research installation. Existing model data, successful frozen sources, SDK releases and experiment outputs are preserved in place.

Selected sources consist of the organized `sdacc/diffusion_research` package and the existing W8A8 SlideSparse 0.3.3 source backend. `EXPORT_ORIGINS.json` records source-relative origin and original SHA256. Exported SHA256 values identify this publication copy separately.

Public-only changes:

- Added package metadata, install/build instructions, licenses and an explicit native build helper.
- Replaced machine-specific configuration values with local path templates; configuration path resolution also expands environment variables and resolves `conditioning_source`/`study_directory` consistently.
- Required an explicit external Sigma research root; archive backfill locates its installed package instead of a private absolute source path.
- Replaced internal task/archive catalogs with a small method/asset-status catalog. No internal logs or whole experiment archive is distributed.
- Removed private path prefixes from operator provenance metadata, preserving shape dispatch, measurements and source/PID identities.

No pruning/quantization arithmetic, calibration sampling, model routing, affine fitting, precision or timing mathematics is changed for publication. No GPU work, inference, generation, scoring, calibration or benchmark is performed as part of this export.
