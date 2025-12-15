"""Weight synchronization abstractions for distributed RL training."""

from .base import WeightChunk
from .weight_extractor import WeightExtractor

__all__ = [
    "WeightChunk",
    "WeightExtractor",
]
