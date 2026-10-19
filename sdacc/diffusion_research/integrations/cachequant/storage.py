"""Preserved offline grid hashing/storage, never an online activation observer."""
import os, hashlib
from pathlib import Path
import torch

def tensor_sha(value):
    # Byte reinterpretation requires a dimension for BF16/FP32 scalar slots.
    # Flatten only the HASH view; descriptors retain original shape/stride,
    # tensor values/layout/math remain untouched and non-scalar bytes agree.
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    torch.save(value, tmp)
    with tmp.open('rb') as stream:
        os.fsync(stream.fileno())
    tmp.replace(path)
