"""Lazy access to the preserved Sigma CEM research API without cloning its runtime.

The original frozen package remains the evidence-bearing implementation. Loading
this module alone never imports Torch, loads weights, or starts CUDA.
"""
import importlib
import os
import sys
from pathlib import Path


def _api(root=None):
    root = root or os.environ.get('SDACC_ORTHOGONAL_ROOT')
    if not root:
        raise ValueError('Set SDACC_ORTHOGONAL_ROOT to the preserved Sigma research bundle.')
    root = Path(root).expanduser()
    source = (root / 'research/cem_duca/src').resolve()
    if not (source / 'sigma_orthogonal/cem_api.py').is_file():
        raise FileNotFoundError(source / 'sigma_orthogonal/cem_api.py')
    loaded = sys.modules.get('sigma_orthogonal')
    if loaded is not None and Path(loaded.__file__).resolve().parent != source / 'sigma_orthogonal':
        raise RuntimeError('A different frozen sigma_orthogonal runtime is already loaded; use a fresh process.')
    sys.path.insert(0, str(source))
    try:
        return importlib.import_module('sigma_orthogonal.cem_api')
    finally:
        sys.path.remove(str(source))


def load_model(paths, output, *, root=None):
    return _api(root).load_model(paths, output)


def prepare(model, definition, *, root=None):
    return _api(root).prepare(model, definition)


def generate(model, condition, variant, *, root=None):
    return _api(root).generate(model, condition, variant)
