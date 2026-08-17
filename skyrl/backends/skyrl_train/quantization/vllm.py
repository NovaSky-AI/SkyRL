"""vLLM receiver mappings for serialized quantized weights."""

from __future__ import annotations

from functools import cache

from .base import ModelQuantizationLayout, QuantizationStrategy
from .model_layouts import MODEL_QUANTIZATION_LAYOUTS
from .registry import get_serialized_weight_strategy


def resolve_vllm_receiver_target(
    strategy: QuantizationStrategy,
    layout: ModelQuantizationLayout,
    checkpoint_name: str,
) -> tuple[str, str] | None:
    """Resolve a checkpoint tensor to a fused vLLM parameter and shard."""

    for target in layout.targets:
        if target.checkpoint_component is None or target.vllm_parameter is None or target.vllm_shard_id is None:
            continue
        for checkpoint_suffix, parameter_suffix in strategy.receiver_suffixes.items():
            suffix = f".experts.{target.checkpoint_component}{checkpoint_suffix}"
            if checkpoint_name.endswith(suffix):
                parameter_name = (
                    checkpoint_name[: -len(suffix)] + f".experts.{target.vllm_parameter}" + parameter_suffix
                )
                return parameter_name, target.vllm_shard_id
    return None


@cache
def resolve_registered_vllm_receiver_target(
    mode: str,
    checkpoint_name: str,
) -> tuple[str, str] | None:
    """Resolve against every built-in layout and reject conflicting mappings."""

    strategy = get_serialized_weight_strategy(mode)
    resolved = set()
    for layout in {id(layout): layout for layout in MODEL_QUANTIZATION_LAYOUTS.values()}.values():
        target = resolve_vllm_receiver_target(strategy, layout, checkpoint_name)
        if target is not None:
            resolved.add(target)
    if len(resolved) > 1:
        raise ValueError(f"Conflicting vLLM receiver targets for {checkpoint_name!r}: {sorted(resolved)}")
    return next(iter(resolved), None)
