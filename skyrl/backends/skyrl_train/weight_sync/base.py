"""Base data structures for weight synchronization."""

from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Dict, Iterator, List

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


@dataclass
class LoraLoadRequest(WeightUpdateRequest):
    """Request to load LoRA weights from disk.

    This is a special request type used for loading LoRA adapters
    from disk rather than transferring weights over network in training. Unlike other
    WeightUpdateRequest subclasses, this doesn't transfer weights - it tells
    the inference engine to load LoRA from a path.

    ``lora_name`` is the name vLLM should register the adapter under and is
    what callers later pass as ``model=<lora_name>`` when sampling. Empty
    string preserves the legacy single-tenant behavior where the engine
    generates a numeric name itself.
    """

    names: List[str] = field(default_factory=list)
    dtypes: List[str] = field(default_factory=list)
    shapes: List[List[int]] = field(default_factory=list)
    lora_path: str = ""
    lora_name: str = ""


@dataclass
class WeightChunk:
    """Represents one or more model parameters to be transferred.

    A WeightChunk can contain multiple parameters grouped together for efficient
    transfer (e.g., Q/K/V projections for fused-weight loaders).

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


def torch_dtype_name(dtype: torch.dtype) -> str:
    """Return the dtype spelling expected by vLLM weight-transfer metadata."""
    return str(dtype).split(".")[-1]


def cuda_uuid_to_str(uuid: str | bytes) -> str:
    """Normalize CUDA UUIDs identically on both sides of an IPC transfer."""
    return uuid.decode("ascii") if isinstance(uuid, bytes) else str(uuid)


def iter_single_dtype_chunks(chunk: WeightChunk) -> Iterator[WeightChunk]:
    """Split one logical chunk into dtype-homogeneous chunks.

    Serialized FP8 checkpoint-format sync includes FP8 weights, FP32 block
    scales, and BF16 tensors that intentionally remain unquantized. IPC buffers
    must describe one concrete tensor dtype at a time. Preserve the
    first-seen dtype order so every training rank follows the same deterministic
    update sequence. vLLM's NCCL path byte-packs mixed dtypes and does not need
    this partitioning.
    """
    by_dtype: Dict[torch.dtype, Dict[str, list]] = {}
    dtype_order: List[torch.dtype] = []

    for name, tensor in zip(chunk.names, chunk.tensors):
        dtype = tensor.dtype
        if dtype not in by_dtype:
            dtype_order.append(dtype)
            by_dtype[dtype] = {"names": [], "dtypes": [], "shapes": [], "tensors": []}
        group = by_dtype[dtype]
        group["names"].append(name)
        group["dtypes"].append(str(dtype))
        group["shapes"].append(list(tensor.shape))
        group["tensors"].append(tensor)

    for dtype in dtype_order:
        group = by_dtype[dtype]
        yield WeightChunk(
            names=group["names"],
            dtypes=group["dtypes"],
            shapes=group["shapes"],
            tensors=group["tensors"],
        )


def get_weight_chunk_metadata(chunk: WeightChunk) -> Dict[str, List]:
    """Return vLLM metadata from the tensors that will actually be transferred."""
    return {
        "names": list(chunk.names),
        "dtype_names": [torch_dtype_name(tensor.dtype) for tensor in chunk.tensors],
        "shapes": [list(tensor.shape) for tensor in chunk.tensors],
    }
