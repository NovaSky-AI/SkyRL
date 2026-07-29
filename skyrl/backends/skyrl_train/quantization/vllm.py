"""vLLM receiver mappings for serialized quantized weights."""

from __future__ import annotations

from .base import QuantizationStrategy, QuantizedModelLayout, WeightCategory

VLLM_EXPERT_TARGETS: dict[WeightCategory, tuple[str, str]] = {
    "routed_expert_gate": (".experts.w13_weight", "w1"),
    "routed_expert_up": (".experts.w13_weight", "w3"),
    "routed_expert_down": (".experts.w2_weight", "w2"),
}


def resolve_vllm_receiver_target(
    strategy: QuantizationStrategy,
    layout: QuantizedModelLayout,
    checkpoint_name: str,
) -> tuple[str, str] | None:
    """Resolve a checkpoint tensor name to a fused vLLM parameter and shard."""

    for expert_spec in layout.moe_expert_specs:
        for weight in expert_spec.output_weights:
            runtime_target = VLLM_EXPERT_TARGETS.get(weight.category)
            if runtime_target is None:
                continue
            runtime_parameter_suffix, shard_id = runtime_target
            for checkpoint_suffix, parameter_suffix in strategy.receiver_suffixes.items():
                suffix = f".experts.{weight.checkpoint_component}{checkpoint_suffix}"
                if checkpoint_name.endswith(suffix):
                    parameter_name = checkpoint_name[: -len(suffix)] + runtime_parameter_suffix + parameter_suffix
                    return parameter_name, shard_id
    return None
