# Experiment and timing contract

- Preserve native model conditioning, scheduler, CFG, Q/K normalization and VAE scale/shift. Resolutions and step counts are model-specific, as listed in the root README.
- Fixed targets are 15%, 25%, 35%, 45%. Measure and report actual structural sparsity separately; whole-projection integer budgets and tail padding can differ from the target.
- Each policy uses its own 100 complete uncorrected calibration trajectories. Dense teacher input matches the actual sparse input; sample at most 128 uniform valid rows while retaining every channel. Accumulate FP64 moments, solve channel-wise closed-form affine coefficients only after the calibration prefix is complete, fold scale/bias in FP32 offline.
- Four pilot conditions are separate from the 64 independent evaluation conditions. Do not use evaluation data for policy selection or coefficient fitting.
- Official performance uses same-PID Dense/raw/affine pairs, four conditions, four warmups and eight repeats. Keep anomalous/raw repeats and reject an entire contaminated condition rather than selecting individual favorable repeats.
- Include necessary bank/parameter selection and H2D/D2D in `Timer.select`. Loading, first compile and offline calibration are separate costs, not hidden speedups.
- Default compiled BF16 Dense is the primary denominator. When graph replay or generic launch optimization is enabled, also retain an equally optimized Dense comparison.
- PSNR and Gaussian11/σ1.5 weighted SSIM use matched Dense references; CLIP is ViT-B/16; ImageReward is official v1.0. Class-conditional DiT requires an explicitly suitable prompt/metric protocol rather than fabricated text conditioning.
- Historical results belong to their actual source, producer PID and conditions. Publishing or refactoring source does not revalidate them. Retain original failed evidence and already completed artifacts; resume only missing work when an experiment is explicitly requested.

Operator complete-projection speedups, denoiser speedups, end-to-end latency and image-quality metrics are distinct measurements. No kernel result alone establishes model-level acceleration or quality preservation.
