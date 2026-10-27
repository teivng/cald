"""CALD: calibrated anchor-loss disagreement — a theory-driven shift detector
for in-context tabular foundation models.

The detector fires when the post-finetune anchor loss is LOW; that direction
is part of the method and is never flipped (see cald.detector).
"""

from .adapters import CALDBackbone, get_adapter, register_adapter
from .detector import CALD, CALDResult

__version__ = "0.1.0"
__all__ = ["CALD", "CALDResult", "CALDBackbone", "get_adapter",
           "register_adapter", "__version__"]
