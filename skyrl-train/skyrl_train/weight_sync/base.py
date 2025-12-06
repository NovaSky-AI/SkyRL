"""Base data structures for weight synchronization."""

from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional

import torch


@dataclass
class WeightChunk:
    """Represents one or more model parameters to be transferred.

    A WeightChunk can contain multiple parameters grouped together for efficient
    transfer (e.g., Q/K/V projections for FlashRL fusion).

    Attributes:
        names: List of parameter names (e.g., ["model.layer.0.weight"])
        dtypes: List of dtype strings (e.g., ["torch.bfloat16"])
        shapes: List of tensor shapes (e.g., [[4096, 4096]])
        tensors: List of actual tensor data (populated during extraction)
        module_name: Optional module identifier for grouped parameters
        total_size_bytes: Total memory footprint (cached property, auto-calculated)
    """

    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    tensors: List[torch.Tensor]
    module_name: Optional[str] = None

    @cached_property
    def total_size_bytes(self) -> int:
        """Calculate total memory footprint in bytes."""
        return sum(t.numel() * t.element_size() for t in self.tensors)
