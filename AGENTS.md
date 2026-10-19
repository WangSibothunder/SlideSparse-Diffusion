# Maintenance

- Maintain one shared package; keep model differences in `models/`.
- Preserve existing calibration, banks, affine coefficients, experiment results and original provenance.
- Code organization or publication never authorizes inference, GPU experiments, calibration, quality scoring or benchmark repetition.
- Use small relevant code checks; avoid numbered prototypes and repeated validation chains.
- Keep default compiled Dense as the primary speedup denominator. Record equal graph Dense separately.
- Account for necessary selection/copies in official denoiser timing. Do not infer model quality from kernel measurements.
- Keep optional experimental integrations disabled unless explicitly requested.
