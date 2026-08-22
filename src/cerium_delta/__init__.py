"""
Cerium Delta
============

Neural network observability and Neural Vitality analysis toolkit.

Public API
----------
NVS
    Direct Neural Vitality analysis for users who already have
    the required model/state data.

Bridge
    Framework integration layer for users who want Cerium Delta
    to extract model information from supported ML frameworks.
"""

from .meterics.brain import NVS
from .exporters.dev import Bridge

__version__ = "1.0"

__all__ = [
    "NVS",
    "Bridge",
    "__version__",
]
