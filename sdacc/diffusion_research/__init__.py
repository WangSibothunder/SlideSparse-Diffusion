"""Shared diffusion research adapters; importing this package never loads CUDA."""
from .registry import MODELS, definition, build, inventory

__all__=['MODELS','definition','build','inventory']
