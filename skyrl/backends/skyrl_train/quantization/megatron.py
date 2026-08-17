"""Megatron adapters for model quantization strategies."""

from __future__ import annotations

from types import MethodType

from .base import ExpertExportLayout, ModelQuantizationLayout, QuantizationStrategy


def build_high_precision_te_recipe() -> dict:
    return {
        "transformer_engine_config_type": "TEQuantizationParams",
        "training_recipe": {"override_quantized_autocast": True},
        "evaluation_recipe": {"override_quantized_autocast": True},
    }


def build_megatron_matchers(
    strategy: QuantizationStrategy,
    layout: ModelQuantizationLayout,
) -> dict:
    """Translate layout-owned target patterns into Megatron matchers."""

    layout.validate_targets(strategy.target_names)
    pattern_owners: dict[str, set[str]] = {}
    for target in layout.targets:
        for pattern in target.megatron_patterns:
            pattern_owners.setdefault(pattern, set()).add(target.name)

    matchers = {}
    for index, (pattern, owners) in enumerate(pattern_owners.items()):
        selected = owners & strategy.target_names
        if selected and selected != owners:
            raise ValueError(
                f"Megatron pattern {pattern!r} represents inseparable targets {sorted(owners)}; "
                f"selected only {sorted(selected)}"
            )
        if selected:
            matchers[f"target_{index}"] = {
                "type": "glob",
                "pattern": pattern,
                "config": "quantized",
                "enabled": True,
            }
    matchers["default"] = {
        "type": "glob",
        "pattern": "*",
        "config": "high_precision",
        "enabled": True,
    }
    return matchers


def build_megatron_quantization_recipe(
    strategy: QuantizationStrategy,
    layout: ModelQuantizationLayout,
    *,
    persistent: bool,
) -> dict:
    return {
        "configs": {
            "quantized": strategy.build_te_recipe(persistent=persistent),
            "high_precision": build_high_precision_te_recipe(),
        },
        "matchers": build_megatron_matchers(strategy, layout),
    }


def configure_megatron_quantization(
    provider,
    strategy: QuantizationStrategy,
    layout: ModelQuantizationLayout,
    *,
    persistent: bool,
) -> None:
    """Apply a strategy and model layout to a Megatron provider."""

    from megatron.core.quantization.quant_config import RecipeConfig

    strategy.validate_layout(layout)
    strategy.configure_megatron_provider(provider)
    provider.quant_recipe = RecipeConfig.from_config_dict(
        build_megatron_quantization_recipe(strategy, layout, persistent=persistent)
    )


def get_packed_qwen3_moe_conversion_tasks(auto_bridge, model):
    """Build Qwen3 MoE export tasks with experts packed by layer."""

    from megatron.bridge.models.conversion.mapping_registry import (
        MegatronMappingRegistry,
    )
    from megatron.bridge.models.conversion.param_mapping import (
        FusedExpertMapping,
        FusedGatedExpertMapping,
    )

    model_bridge = auto_bridge._model_bridge
    mappings = [mapping for mapping in model_bridge.mapping_registry() if ".mlp.experts." not in mapping.megatron_param]
    mappings.extend(
        [
            FusedGatedExpertMapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                hf_param="model.layers.*.mlp.experts.gate_up_proj",
            ),
            FusedExpertMapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                hf_param="model.layers.*.mlp.experts.down_proj",
            ),
            FusedGatedExpertMapping(
                megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                hf_param="model.layers.*.mlp.experts.gate_up_proj",
            ),
            FusedExpertMapping(
                megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                hf_param="model.layers.*.mlp.experts.down_proj",
            ),
        ]
    )
    registry = MegatronMappingRegistry(*mappings)
    model_bridge.mapping_registry = MethodType(lambda _self: registry, model_bridge)
    model_list = model if isinstance(model, list) else [model]
    return model_bridge.build_conversion_tasks(auto_bridge.hf_pretrained, model_list)


def get_quantized_conversion_tasks(auto_bridge, model, layout: ModelQuantizationLayout):
    """Build Bridge export tasks required by a model layout."""

    if layout.expert_export_layout is ExpertExportLayout.PACKED:
        return get_packed_qwen3_moe_conversion_tasks(auto_bridge, model)
    return auto_bridge.get_conversion_tasks(model)
