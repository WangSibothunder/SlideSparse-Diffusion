"""Experiment integrations and provenance; importing this module never loads Torch."""
import json
from pathlib import Path


def catalog():
    """Read the preserved experiment catalog without executing any driver."""
    return json.loads((Path(__file__).parent / 'CATALOG.json').read_text())


__all__ = ['catalog']
