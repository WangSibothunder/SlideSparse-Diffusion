# Optional SM80 SpargeAttn specialization

Selected upstream headers from `thu-ml/SpargeAttn` at `ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a` retain their copyright notices and Apache-2.0 license. Local `current_stream.cu` uses an explicit current PyTorch stream for the fixed noncausal SM80/head128 specialization. See `PROVENANCE.json` and root `NOTICE`.

Compilation is explicit through this directory's `build.py` and `SPARGE_STREAM_BUILD`. No binary is bundled. This optional native component is not required by the six core model adapters, and its source availability is not a complete-model performance claim.
