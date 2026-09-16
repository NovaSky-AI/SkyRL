"""Canonical PEFT layouts shared by disk and remote adapter publication."""

from collections.abc import Mapping

import torch


def convert_moe_expert_lora_key(key: str, tensor_ndim: int) -> str:
    """Map fused expert keys to the flat PEFT names expected by vLLM."""
    if tensor_ndim != 3:
        return key
    if ".mlp.experts.gate_up_proj." in key:
        return key.replace(".mlp.experts.gate_up_proj.", ".mlp.experts.base_layer.")
    if ".mlp.experts.down_proj." in key:
        return key.replace(".mlp.experts.down_proj.", ".mlp.experts.")
    return key


def convert_moe_experts_lora_to_vllm(
    adapter_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Flatten expert-major A and expert-minor B values for vLLM's PEFT loader."""
    converted = {}
    for key, tensor in adapter_state.items():
        converted_key = convert_moe_expert_lora_key(key, tensor.ndim)
        if converted_key != key:
            if key.endswith(".lora_A.weight"):
                tensor = tensor.reshape(-1, tensor.shape[-1]).contiguous()
            elif key.endswith(".lora_B.weight"):
                tensor = tensor.permute(1, 2, 0).contiguous().reshape(tensor.shape[1], -1)
        converted[converted_key] = tensor
    return converted
