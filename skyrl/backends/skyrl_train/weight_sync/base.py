"""Base data structures for weight synchronization."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class LoraLoadRequest:
    """Request to load LoRA weights from disk.

    Not a weight *transfer*: it tells the inference engine to load an adapter
    from a path rather than moving any tensor.

    ``lora_name`` is the name vLLM registers the adapter under, and what callers
    later pass as ``model=<lora_name>`` when sampling. Empty string preserves the
    legacy single-tenant behavior where the engine generates a numeric name.
    """

    lora_path: str = ""
    lora_name: str = ""

    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize the request to JSON."""
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "LoraLoadRequest":
        """Deserialize the request from JSON."""
        return cls(**data)
