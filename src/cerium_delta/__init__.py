```python
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

visualizer
    Static visualization interface for analyzing Neural Vitality
    and layer-level observability results.

Statistical
    Statistical analysis utilities used for inspecting and
    analyzing Neural Vitality data.
"""

from .metrics.brain import NVS
from .exporters.dev import bridge
from .visualizers.static_viz import visualizer
from .visualizers.stats_method import Statistical

__version__ = "1.1.0"

__all__ = [
    "NVS",
    "bridge",
    "visualizer",
    "Statistical",
    "__version__",
]
```
