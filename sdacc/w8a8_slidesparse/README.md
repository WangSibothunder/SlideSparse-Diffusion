# W8A8 SlideSparse source backend

This directory contains the existing 0.3.3 SM80 operator implementation and its qualified static dispatch profiles. The public export excludes compiled binaries, historical experiment directories and bulky raw traces.

- INT8 `2:4`, `4:6`, `6:8`; dynamic per-token activation quantization.
- Offline weight packing, per-output-channel weight scales, BF16 output.
- Quant/lift frontend plus fused GEMM scale/bias, same-input fused QKV/KV, protected BF16 addend support.
- Static shape/config choices; no forward-time search, packing or fitting.

```python
from sdacc.w8a8_slidesparse import build_native, load_native, PackedLinear

build_native('local-build', '/path/to/pinned/cutlass', fused_projection=True)
load_native('local-build')
# qweight is already pruned and quantized CPU INT8 [out,in].
# op = PackedLinear.prepare(qweight, scales, bias, mode='4:6').cuda()
# op.configure()
```

For the research runtime's bound source snapshot, use root `scripts/build_native.py`, which builds into this directory's `native_sm80/`. Build using the current local Torch/CUDA ABI; never load a binary from another ABI. CUTLASS pin and retained BSD license live in `reference/`.

Profiles retain historical measurement identities and are not new public-export benchmarks. Synthetic operator speedups are not denoiser or model-quality claims. The root README defines the shared model runtime and measurement boundaries.
