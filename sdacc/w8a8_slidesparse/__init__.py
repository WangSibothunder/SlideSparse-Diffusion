"""Dynamic-token W8A8 SlideSparse projections for NVIDIA SM80.

Packing is offline. No pruning, fitting, search, or compilation in forward.
"""
from .backend import build_native, load_native
from .packing import lifted_k, slide_weight
from .runtime import PackedLinear, quantize_lift

__all__ = ["PackedLinear", "quantize_lift", "slide_weight", "lifted_k",
           "build_native", "load_native", "compile_projection"]

from .compilation import compile_projection
