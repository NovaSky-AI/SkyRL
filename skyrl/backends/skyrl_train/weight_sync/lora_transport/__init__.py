"""Shared contracts for native named-adapter transport backends."""

from .bridge_sources import (
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    extract_lora_bridge_sources,
    validate_lora_bridge_source_layout,
)
from .contracts import (
    LoRAUpdateRequest,
)
from .server_lifecycle import (
    LoRATransportRollbackError,
    LoRATransportServerLifecycle,
)
from .vllm_adapter import (
    activate_staged_vllm_lora_model,
    build_vllm_lora_consumer_plan,
    discard_staged_vllm_lora_model,
)

__all__ = [
    "LoRABridgeSource",
    "LoRABridgeSourceLayout",
    "LoRATransportRollbackError",
    "LoRATransportServerLifecycle",
    "LoRAUpdateRequest",
    "activate_staged_vllm_lora_model",
    "build_vllm_lora_consumer_plan",
    "discard_staged_vllm_lora_model",
    "extract_lora_bridge_sources",
    "validate_lora_bridge_source_layout",
]
