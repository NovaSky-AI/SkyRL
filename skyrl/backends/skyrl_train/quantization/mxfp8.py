"""Expert MXFP8 training and serialized weight encoding."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar

import torch

from .base import ModelQuantizationLayout, QuantizationStrategy
from .blockwise_fp8 import batched_blockwise_cast_to_fp8, blockwise_cast_to_fp8

MXFP8 = "mxfp8"

EXPERT_ONLY_MXFP8_IGNORED_MODULES = (
    "*.self_attn.*",
    "*.linear_attn.*",
    "*.mlp.gate",
    "*.mlp.gate_up_proj",
    "*.mlp.down_proj",
    "*.mlp.shared_expert*",
    "*lm_head*",
    "*.visual.*",
    "mtp.*",
)


def build_mxfp8_te_recipe(*, persistent: bool) -> dict:
    """Build Transformer Engine settings for expert MXFP8 modules."""

    recipe = {
        "transformer_engine_config_type": "TEQuantizationParams",
        "training_recipe": {
            "fp8_quantization_recipe": "mxfp8",
            "fp8_format": "e4m3",
            "override_quantized_autocast": True,
            "override_nonquantized_autocast": True,
            "fp8_param": persistent,
        },
        "evaluation_recipe": {
            "fp8_quantization_recipe": "mxfp8",
            "fp8_format": "e4m3",
            "override_quantized_autocast": True,
            "override_nonquantized_autocast": True,
            "fp8_param": persistent,
        },
    }
    return recipe


def configure_mxfp8_provider(provider) -> None:
    """Set Megatron provider fields required by expert MXFP8."""

    if getattr(provider, "quant_recipe", None) is not None:
        raise ValueError("Expert MXFP8 conflicts with an existing quant_recipe")
    if getattr(provider, "fp8", None) not in (None, False):
        raise ValueError("Expert MXFP8 conflicts with global FP8 configuration")
    if not provider.moe_grouped_gemm:
        raise ValueError("Expert MXFP8 requires moe_grouped_gemm=true")
    if provider.moe_router_dtype != "fp32":
        raise ValueError("Expert MXFP8 requires moe_router_dtype=fp32")
    for option in (
        "fp8_dot_product_attention",
        "fp8_multi_head_attention",
        "fp8_output_proj",
    ):
        if getattr(provider, option, False):
            raise ValueError(f"Expert MXFP8 requires {option}=false")

    provider.fp8 = "e4m3"
    provider.fp8_recipe = "mxfp8"
    provider.moe_router_padding_for_quantization = False


def validate_mxfp8_hardware() -> None:
    """Require native Blackwell MXFP8 support."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError(f"Expert MXFP8 requires SM100 or SM103, got SM{capability[0]}{capability[1]}")


def is_routed_expert_linear(name: str) -> bool:
    parts = name.split(".")
    return any(
        parts[index] == "experts" and parts[index + 1] in ("linear_fc1", "linear_fc2")
        for index in range(len(parts) - 1)
    )


def audit_expert_mxfp8_modules(model_chunks, *, persistent: bool) -> int:
    """Verify expert-only MXFP8 execution and optional parameter storage."""

    from megatron.core.fp8_utils import is_mxfp8tensor

    matched = []
    errors = []
    for chunk in model_chunks:
        for name, module in chunk.named_modules():
            if not hasattr(module, "will_execute_quantized"):
                continue
            targeted = is_routed_expert_linear(name)
            quantized = module.will_execute_quantized(True)
            quant_params = getattr(module, "te_quant_params", None)
            recipe = getattr(
                getattr(quant_params, "training_recipe", None),
                "fp8_quantization_recipe",
                None,
            )
            uses_mxfp8 = getattr(recipe, "value", recipe) == "mxfp8"
            persistent_params = any(is_mxfp8tensor(param) for param in module.parameters(recurse=False))
            if targeted and (not quantized or not uses_mxfp8):
                errors.append(f"{name} did not enable MXFP8")
            elif not targeted and (quantized or uses_mxfp8 or persistent_params):
                errors.append(f"{name} unexpectedly enabled MXFP8")
            elif targeted and persistent != persistent_params:
                errors.append(f"{name} persistent MXFP8 storage was {persistent_params}, expected {persistent}")
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
        raise RuntimeError("; ".join(message for rank in rank_errors for message in rank))
    if state[1].item() == 0:
        raise RuntimeError("Expert MXFP8 strategy matched no routed expert modules")
    return state[1].item()


def mxfp8_scale_name_for_weight(name: str) -> str:
    if not name.endswith(".weight"):
        raise ValueError(f"MXFP8 scale can only be derived from .weight tensors: {name}")
    return name[: -len(".weight")] + ".weight_scale"


def power_2_scales_to_e8m0(scales: torch.Tensor) -> torch.Tensor:
    """Encode positive power-of-two FP32 scales as E8M0 bytes."""

    exponent_bits = (scales.contiguous().view(torch.int32) >> 23) & 0xFF
    return exponent_bits.to(torch.uint8)


def mxfp8_cast_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to E4M3 data and E8M0 scales."""

    if weight.ndim != 2:
        raise ValueError(f"MXFP8 expects a 2D tensor, got shape={tuple(weight.shape)}")
    if weight.shape[-1] % 32 != 0:
        raise ValueError(f"MXFP8 requires the last dimension to be divisible by 32, got shape={tuple(weight.shape)}")
    q_weight, scales = blockwise_cast_to_fp8(weight, (1, 32), power_2_scale=True)
    return q_weight, power_2_scales_to_e8m0(scales)


def batched_mxfp8_cast_to_fp8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 3D expert tensor to E4M3 data and E8M0 scales."""

    if weight.ndim != 3:
        raise ValueError(f"Batched MXFP8 expects a 3D tensor, got shape={tuple(weight.shape)}")
    if weight.shape[-1] % 32 != 0:
        raise ValueError(f"MXFP8 requires the last dimension to be divisible by 32, got shape={tuple(weight.shape)}")
    q_weight, scales = batched_blockwise_cast_to_fp8(weight, (1, 32), power_2_scale=True)
    return q_weight, power_2_scales_to_e8m0(scales)


class Mxfp8ExpertStrategy(QuantizationStrategy):
    """Apply MXFP8 to routed expert weights."""

    mode: ClassVar[str] = MXFP8
    target_names: ClassVar[frozenset[str]] = frozenset({"routed_expert_gate", "routed_expert_up", "routed_expert_down"})
    receiver_suffixes: ClassVar[dict[str, str]] = {
        ".weight": "",
        ".weight_scale": "_scale",
    }
    reject_unknown_routed_experts: ClassVar[bool] = True
    required_model_dtype: ClassVar[str | None] = "bfloat16"
    vllm_quantization: ClassVar[str] = "modelopt_mxfp8"

    def build_te_recipe(self, *, persistent: bool) -> dict:
        return build_mxfp8_te_recipe(persistent=persistent)

    def configure_megatron_provider(self, provider) -> None:
        configure_mxfp8_provider(provider)

    def build_runtime_env(self) -> dict[str, str]:
        return {
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "0",
            "VLLM_USE_DEEP_GEMM_E8M0": "1",
        }

    def serialize_weight(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        batched_experts: bool,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        q_weight, scale = batched_mxfp8_cast_to_fp8(tensor) if batched_experts else mxfp8_cast_to_fp8(tensor)
        yield name, q_weight
        yield mxfp8_scale_name_for_weight(name), scale

    def vllm_quantization_config(
        self,
        inference_config: Any,
        hf_config: Any,
        layout: ModelQuantizationLayout,
    ) -> dict[str, Any]:
        del inference_config, hf_config, layout
        return {
            "quant_method": "modelopt",
            "quant_algo": "MXFP8",
            "ignore": list(EXPERT_ONLY_MXFP8_IGNORED_MODULES),
        }
