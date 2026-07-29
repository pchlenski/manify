"""Project-wide default constants.

These centralize a few values that are shared across many classes and functions so they can be set in one
place. They are only used as *default* argument values; every object stores its own resolved ``dtype``/``device``
and reads from that instance attribute, so passing an explicit value always takes precedence.
"""

from __future__ import annotations

import torch

# Default floating-point dtype for manifolds and the operations built on them. float64 is the sensible default
# for manifold math: the ambient hyperboloid/sphere coordinates are cosh/sin of the tangent norm and overflow
# float32 once |curvature| * sigma^2 * dim grows. Pass an explicit dtype to trade this headroom for memory.
DEFAULT_DTYPE: torch.dtype = torch.float64

# Default device for manifolds, samples, and models.
DEFAULT_DEVICE: str = "cpu"
