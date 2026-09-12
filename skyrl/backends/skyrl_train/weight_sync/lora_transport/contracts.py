"""Immutable generation contracts for native LoRA transport."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol


class _LoRALayout(Protocol):
    adapter_name: str
    layout_digest: str
    source_dtype: str


@dataclass(frozen=True)
class LoRASourceSlice:
    """A rectangular selection from one producer-owned adapter tensor."""

    key: str
    starts: tuple[int, ...]
    stops: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.key or not self.starts or len(self.starts) != len(self.stops):
            raise ValueError("LoRA source slices require a key and matching nonempty bounds")
        if any(start < 0 or stop <= start for start, stop in zip(self.starts, self.stops)):
            raise ValueError("LoRA source slice bounds must be nonnegative and nonempty")

    def validate_shape(self, shape: tuple[int, ...]) -> None:
        if len(shape) != len(self.stops) or any(stop > size for stop, size in zip(self.stops, shape)):
            raise ValueError(f"LoRA source slice {self.key!r} exceeds source shape {shape}")

    @property
    def indices(self) -> tuple[slice, ...]:
        return tuple(slice(start, stop) for start, stop in zip(self.starts, self.stops))


@dataclass(frozen=True)
class LoRAUpdateRequest:
    """A single named-adapter publication generation."""

    adapter_name: str
    generation: int
    layout_digest: str
    source_dtype: str

    @classmethod
    def from_layout(cls, layout: _LoRALayout, generation: int) -> "LoRAUpdateRequest":
        return cls(
            adapter_name=layout.adapter_name,
            generation=generation,
            layout_digest=layout.layout_digest,
            source_dtype=layout.source_dtype,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "LoRAUpdateRequest":
        return cls(**data)

    def __post_init__(self) -> None:
        if not self.adapter_name:
            raise ValueError("LoRA update requests require a non-empty adapter name")
        if self.generation < 0:
            raise ValueError(f"LoRA generation must be non-negative, got {self.generation}")
        if self.source_dtype != "float32":
            raise ValueError(f"lora_transport requires float32 sources, got {self.source_dtype!r}")
        if len(self.layout_digest) != 64:
            raise ValueError("LoRA update requests require a SHA-256 layout digest")


@dataclass(frozen=True)
class LoRAReceiverGeneration:
    """Retain the fixed receiver contract with its active or staged generation."""

    request: LoRAUpdateRequest
    adapter_id: int
    adapter_config_json: str
