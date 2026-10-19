# Shared diffusion adapters

See the repository [README](../../README.md) for installation, supported models and entrypoints, and [usage](../../docs/USAGE.md) for local asset contracts.

- Keep model structure in `models/`; calibration, reconstruction, banks, affine selection and timing share one implementation.
- `python -m sdacc.diffusion_research models` reads metadata only.
- `results` reads an index of separately retained experiment artifacts; the public default catalog contains source/asset status, not bundled measurements.
- `run` is an explicit execution command. It does not infer permission to rerun completed studies.
- Optional plugins are documented in [integrations/README.md](integrations/README.md).

The publishing snapshot preserves original computational implementations. Public changes are confined to packaging, documentation, local-path templates and explicit external path selection. Original experiment directories, sources and successful results were not overwritten or remeasured.
