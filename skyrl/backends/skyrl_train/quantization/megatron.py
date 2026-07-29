"""Megatron training and Bridge adapters for model quantization policies."""

from __future__ import annotations

from types import MethodType

from .base import ExpertExportLayout, QuantizationStrategy, QuantizedModelLayout


def build_high_precision_te_recipe() -> dict:
    """Build Transformer Engine settings for modules left high precision."""

    return {
        "transformer_engine_config_type": "TEQuantizationParams",
        "training_recipe": {"override_quantized_autocast": True},
        "evaluation_recipe": {"override_quantized_autocast": True},
    }


def build_megatron_matchers(strategy: QuantizationStrategy) -> dict:
    """Translate strategy categories into Megatron module-name matchers."""

    gate = "routed_expert_gate" in strategy.quantized_categories
    up = "routed_expert_up" in strategy.quantized_categories
    if gate != up:
        raise ValueError("Megatron fused expert FC1 requires gate and up to use the same precision")

    matchers = {}
    if gate:
        matchers["routed_fc1"] = {
            "type": "glob",
            "pattern": "*.experts.linear_fc1",
            "config": "quantized",
            "enabled": True,
        }
    if "routed_expert_down" in strategy.quantized_categories:
        matchers["routed_fc2"] = {
            "type": "glob",
            "pattern": "*.experts.linear_fc2",
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


def configure_megatron_quantization(
    provider,
    strategy: QuantizationStrategy,
    *,
    persistent: bool,
) -> None:
    """Apply a quantization strategy to a Megatron model provider."""

    from megatron.core.quantization.quant_config import RecipeConfig

    recipe = build_megatron_quantization_recipe(strategy, persistent=persistent)
    strategy.configure_megatron_provider(provider)
    provider.quant_recipe = RecipeConfig.from_config_dict(recipe)


def build_megatron_quantization_recipe(
    strategy: QuantizationStrategy,
    *,
    persistent: bool,
) -> dict:
    """Build a complete Megatron per-module quantization recipe."""

    return {
        "configs": {
            "quantized": strategy.build_te_recipe(persistent=persistent),
            "high_precision": build_high_precision_te_recipe(),
        },
        "matchers": build_megatron_matchers(strategy),
    }


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
    mappings = [
        mapping
        for mapping in model_bridge.mapping_registry()
        if ".mlp.experts." not in mapping.megatron_param
    ]
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


def get_quantized_conversion_tasks(auto_bridge, model, layout: QuantizedModelLayout):
    """Build Bridge export tasks required by a quantized model layout."""

    if layout.expert_export_layout is ExpertExportLayout.PACKED:
        return get_packed_qwen3_moe_conversion_tasks(auto_bridge, model)
    return auto_bridge.get_conversion_tasks(model)
