"""Build and stage vLLM adapter objects from native LoRA transfers."""

from typing import Any, Mapping

import torch

from .bridge_sources import LoRABridgeSourceLayout
from .consumer_plan import LoRAConsumerPlan, build_lora_consumer_plan


def activate_staged_vllm_lora_model(model_runner: Any, adapter_id: int) -> None:
    """Activate a previously staged adapter id in an unused vLLM LoRA slot."""
    manager = _get_vllm_lora_manager(model_runner)
    if adapter_id not in manager.list_adapters():
        raise ValueError(f"LoRA adapter id {adapter_id} was not staged")
    # vLLM returns False when this adapter is already active, including rollback.
    manager.activate_adapter(adapter_id)


def discard_staged_vllm_lora_model(model_runner: Any, adapter_id: int) -> None:
    """Release a staged adapter id after a failed generation."""
    manager = _get_vllm_lora_manager(model_runner)
    manager.remove_adapter(adapter_id)


def _get_vllm_lora_manager(model_runner: Any) -> Any:
    manager = getattr(model_runner, "lora_manager", None)
    if manager is None:
        raise RuntimeError(
            "lora_transport requires a vLLM model runner with LoRA enabled"
        )
    return manager


def get_vllm_local_lora_plan(
    model_runner: Any, adapter_config: Mapping[str, Any]
) -> Any:
    """Bind adapter targets to the actual vLLM rank's local buffers."""
    from vllm.lora.peft_helper import PEFTHelper

    manager = _get_vllm_lora_manager(model_runner)
    if not all(
        hasattr(manager, method)
        for method in (
            "get_local_adapter_plan",
            "add_local_adapter",
            "activate_adapter",
        )
    ):
        raise RuntimeError(
            "lora_transport requires vLLM local-adapter plan and registration APIs"
        )
    return manager.get_local_adapter_plan(PEFTHelper.from_dict(dict(adapter_config)))


def build_vllm_lora_consumer_plan(
    layout: LoRABridgeSourceLayout,
    adapter_config: Mapping[str, Any],
    model_runner: Any,
) -> LoRAConsumerPlan:
    """Bind canonical source ownership to one vLLM rank's local buffers."""
    return build_lora_consumer_plan(
        layout,
        get_vllm_local_lora_plan(model_runner, adapter_config),
    )


def stage_vllm_local_lora_factors(
    model_runner: Any,
    adapter_id: int,
    receiver_plan: Any,
    factors: Mapping[str, tuple[list[torch.Tensor], list[torch.Tensor]]],
) -> None:
    """Register independent local factors without activating a GPU slot."""
    manager = _get_vllm_lora_manager(model_runner)
    if not manager.add_local_adapter(adapter_id, receiver_plan, dict(factors)):
        raise RuntimeError(f"vLLM declined local LoRA adapter id {adapter_id}")
