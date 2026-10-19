"""Build one existing SM80 specialization with an explicit PyTorch stream."""
import hashlib
import json
import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parent
output = Path(os.environ['SPARGE_STREAM_BUILD'])
output.mkdir(parents=True, exist_ok=True)
assert not torch.cuda.is_initialized()
os.environ['MAX_JOBS'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
module = load(
    name='sdacc_sparge_sm80_current_stream',
    sources=[str(root / 'current_stream.cu')], build_directory=str(output),
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math',
        '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
        '-Xcompiler', '-include,cassert', '-diag-suppress=174'], verbose=True)
assert not torch.cuda.is_initialized()
binary = Path(module.__file__)
receipt = dict(complete=True, cuda_initialized=False, torch=torch.__version__,
    binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
    source_manifest_sha256=hashlib.sha256((root/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
    scope='native SM80 head128 BF16, per-block QK, noncausal; explicit current stream only')
(output/'complete.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt), flush=True)
