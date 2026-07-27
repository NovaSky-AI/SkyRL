"""Megatron-Bridge mapping overrides used by serialized weight sync."""

from types import MethodType


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
