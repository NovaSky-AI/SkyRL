"""Expert-only MXFP8 setup for Megatron."""

from __future__ import annotations

import torch
from loguru import logger


def expert_mxfp8_recipe_dict() -> dict:
    """Return the per-module Transformer Engine recipe."""

    high_precision = {
        "transformer_engine_config_type": "TEQuantizationParams",
        "training_recipe": {"override_quantized_autocast": True},
        "evaluation_recipe": {"override_quantized_autocast": True},
    }
    mxfp8 = {
        "transformer_engine_config_type": "TEQuantizationParams",
        "training_recipe": {
            "fp8_quantization_recipe": "mxfp8",
            "fp8_format": "e4m3",
            "override_quantized_autocast": True,
            "override_nonquantized_autocast": True,
        },
        "evaluation_recipe": {
            "fp8_quantization_recipe": "mxfp8",
            "fp8_format": "e4m3",
            "override_quantized_autocast": True,
            "override_nonquantized_autocast": True,
        },
    }
    return {
        "configs": {"expert_mxfp8": mxfp8, "high_precision": high_precision},
        "matchers": {
            "routed_fc1": {
                "type": "glob",
                "pattern": "*.experts.linear_fc1",
                "config": "expert_mxfp8",
                "enabled": True,
            },
            "routed_fc2": {
                "type": "glob",
                "pattern": "*.experts.linear_fc2",
                "config": "expert_mxfp8",
                "enabled": True,
            },
            "default": {
                "type": "glob",
                "pattern": "*",
                "config": "high_precision",
                "enabled": True,
            },
        },
    }


def configure_expert_mxfp8_provider(provider) -> None:
    """Configure routed experts for MXFP8 compute."""

    from megatron.core.quantization.quant_config import RecipeConfig

    if getattr(provider, "quant_recipe", None) is not None:
        raise ValueError("Expert MXFP8 conflicts with an existing quant_recipe")
    if getattr(provider, "fp8", None) not in (None, False):
        raise ValueError("Expert MXFP8 conflicts with global FP8 configuration")
    if not provider.moe_grouped_gemm:
        raise ValueError("Expert MXFP8 requires moe_grouped_gemm=true")
    if provider.moe_router_dtype != "fp32":
        raise ValueError("Expert MXFP8 requires moe_router_dtype=fp32")
    for option in ("fp8_dot_product_attention", "fp8_multi_head_attention", "fp8_output_proj"):
        if getattr(provider, option, False):
            raise ValueError(f"Expert MXFP8 requires {option}=false")
    provider.fp8 = "e4m3"
    provider.fp8_recipe = "mxfp8"
    provider.moe_router_padding_for_quantization = True
    provider.quant_recipe = RecipeConfig.from_config_dict(expert_mxfp8_recipe_dict())


def validate_expert_mxfp8_hardware() -> None:
    """Require native Blackwell MXFP8 support."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError(f"Expert MXFP8 requires SM100 or SM103, got SM{capability[0]}{capability[1]}")


def is_routed_expert_linear(name: str) -> bool:
    """Return whether a module name identifies routed expert FC1 or FC2."""

    parts = name.split(".")
    return any(
        parts[index] == "experts" and parts[index + 1] in ("linear_fc1", "linear_fc2")
        for index in range(len(parts) - 1)
    )


def audit_expert_mxfp8_modules(model_chunks) -> int:
    """Verify that only routed expert linears execute quantized."""

    matched = []
    errors = []
    for chunk in model_chunks:
        for name, module in chunk.named_modules():
            if not hasattr(module, "will_execute_quantized"):
                continue
            targeted = is_routed_expert_linear(name)
            quantized = module.will_execute_quantized(True)
            quant_params = getattr(module, "te_quant_params", None)
            recipe = getattr(getattr(quant_params, "training_recipe", None), "fp8_quantization_recipe", None)
            uses_mxfp8 = getattr(recipe, "value", recipe) == "mxfp8"
            if targeted and (not quantized or not uses_mxfp8):
                errors.append(f"{name} did not enable MXFP8")
            elif not targeted and (quantized or uses_mxfp8):
                errors.append(f"{name} unexpectedly enabled quantization")
            elif targeted:
                matched.append(name)

    state = torch.tensor(
        [bool(errors), len(matched)],
        device=torch.cuda.current_device(),
        dtype=torch.int32,
    )
    torch.distributed.all_reduce(state)
    if state[0].item():
        rank_errors = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(rank_errors, errors)
        messages = [message for rank in rank_errors for message in rank]
        raise RuntimeError("; ".join(messages))
    if state[1].item() == 0:
        raise RuntimeError("Expert MXFP8 recipe matched no routed expert modules")
    if torch.distributed.get_rank() == 0:
        logger.info(f"Expert MXFP8 enabled for {state[1].item()} routed grouped-linear modules")
    return state[1].item()
