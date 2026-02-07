"""Base data structures for weight synchronization."""

from typing import Any, Dict
from dataclasses import dataclass, asdict
from functools import cached_property
from typing import List

import torch


@dataclass
class WeightUpdateRequest:
    """Base class for weight update requests.

    Each transfer strategy has its own request type with strategy-specific fields.
    """

    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]

    def __post_init__(self):
        lengths = [len(self.names), len(self.dtypes), len(self.shapes)]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"names, dtypes, shapes must have the same length. "
                f"Got names={len(self.names)}, dtypes={len(self.dtypes)}, shapes={len(self.shapes)}"
            )

    def __len__(self) -> int:
        return len(self.names)

    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize the request to JSON."""
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "WeightUpdateRequest":
        """Deserialize the request from JSON."""
        return cls(**data)


class LoraLoadRequest(WeightUpdateRequest):
    """Request to load LoRA weights from disk.

    This is a special request type used for loading pre-trained LoRA adapters
    from disk rather than transferring weights from training. Unlike other
    WeightUpdateRequest subclasses, this doesn't transfer weights - it tells
    the inference engine to load LoRA from a path.
    """

    def __init__(self, lora_path: str):
        super().__init__(names=[], dtypes=[], shapes=[])
        self.lora_path = lora_path


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
        total_numel: Total number of elements (cached property, auto-calculated)
        total_size_bytes: Total memory footprint (cached property, auto-calculated)
    """

    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    tensors: List[torch.Tensor]

    def __post_init__(self):
        """Validate that all input lists have the same length."""
        lengths = [len(self.names), len(self.dtypes), len(self.shapes), len(self.tensors)]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"All lists must have the same length. Got names={len(self.names)}, "
                f"dtypes={len(self.dtypes)}, shapes={len(self.shapes)}, tensors={len(self.tensors)}"
            )

    def __len__(self) -> int:
        """Return the number of parameters in this chunk."""
        return len(self.names)

    @cached_property
    def total_numel(self) -> int:
        """Calculate total number of elements across all tensors."""
        return sum(t.numel() for t in self.tensors)

    @cached_property
    def total_size_bytes(self) -> int:
        """Calculate total memory footprint in bytes."""
        return sum(t.numel() * t.element_size() for t in self.tensors)
