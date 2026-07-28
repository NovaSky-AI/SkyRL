"""Megatron training and Bridge adapters for model quantization policies."""

from __future__ import annotations

from types import MethodType

from .base import ExpertExportLayout, ModelQuantizationPolicy
from .mxfp8 import build_mxfp8_te_recipe, configure_mxfp8_provider


def build_high_precision_te_recipe() -> dict:
    """Build Transformer Engine settings for modules left high precision."""

    return {
        "transformer_engine_config_type": "TEQuantizationParams",
        "training_recipe": {"override_quantized_autocast": True},
        "evaluation_recipe": {"override_quantized_autocast": True},
    }


def build_megatron_matchers(policy: ModelQuantizationPolicy) -> dict:
    """Translate policy categories into Megatron module-name matchers."""

    gate = policy.quantizes_category("routed_expert_gate")
    up = policy.quantizes_category("routed_expert_up")
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
    if policy.quantizes_category("routed_expert_down"):
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
    policy: ModelQuantizationPolicy,
    *,
    format_name: str,
    persistent: bool,
) -> None:
    """Apply a policy and format to a Megatron model provider."""

    from megatron.core.quantization.quant_config import RecipeConfig

    recipe = build_megatron_quantization_recipe(policy, format_name=format_name, persistent=persistent)
    configure_mxfp8_provider(provider)
    provider.quant_recipe = RecipeConfig.from_config_dict(recipe)


def build_megatron_quantization_recipe(
    policy: ModelQuantizationPolicy,
    *,
    format_name: str,
    persistent: bool,
) -> dict:
    """Build a complete Megatron per-module quantization recipe."""

    if format_name != "mxfp8":
        raise ValueError(f"Unsupported Megatron quantization format {format_name!r}")
    return {
        "configs": {
            "quantized": build_mxfp8_te_recipe(persistent=persistent),
            "high_precision": build_high_precision_te_recipe(),
        },
        "matchers": build_megatron_matchers(policy),
    }


def get_packed_qwen3_moe_conversion_tasks(auto_bridge, model):
    """Build Qwen3 MoE export tasks with experts packed by layer."""

    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.param_mapping import FusedExpertMapping, FusedGatedExpertMapping

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


def get_quantized_conversion_tasks(auto_bridge, model, policy: ModelQuantizationPolicy):
    """Build Bridge export tasks required by a model quantization policy."""

    if policy.expert_export_layout is ExpertExportLayout.PACKED:
        return get_packed_qwen3_moe_conversion_tasks(auto_bridge, model)
    return auto_bridge.get_conversion_tasks(model)
