"""MXFP8 training integration and serialized weight encoding."""

from __future__ import annotations

from typing import Any, ClassVar, Iterator

import torch

from .base import QuantizationStrategy, QuantizedModelLayout, WeightCategory
from .blockwise_fp8 import batched_blockwise_cast_to_fp8, blockwise_cast_to_fp8

SERIALIZED_MXFP8 = "serialized_mxfp8"
MXFP8_1X32 = "mxfp8_1x32"

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
    """Build Transformer Engine settings for MXFP8 modules."""

    recipe = {
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
    if persistent:
        recipe["training_recipe"]["fp8_param"] = True
        recipe["evaluation_recipe"]["fp8_param"] = True
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
    for option in ("fp8_dot_product_attention", "fp8_multi_head_attention", "fp8_output_proj"):
        if getattr(provider, option, False):
            raise ValueError(f"Expert MXFP8 requires {option}=false")

    provider.fp8 = "e4m3"
    provider.fp8_recipe = "mxfp8"
    provider.moe_router_padding_for_quantization = False


def validate_mxfp8_hardware() -> None:
    """Require a Blackwell GPU with native MXFP8 support."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError(f"Expert MXFP8 requires SM100 or SM103, got SM{capability[0]}{capability[1]}")


def get_serialized_mxfp8_quantization_config() -> dict:
    """Return ModelOpt configuration for expert-only MXFP8 loading."""

    return {
        "quant_method": "modelopt",
        "quant_algo": "MXFP8",
        "ignore": list(EXPERT_ONLY_MXFP8_IGNORED_MODULES),
    }


def mxfp8_scale_name_for_weight(name: str) -> str:
    """Return the checkpoint name for an MXFP8 E8M0 scale."""

    if not name.endswith(".weight"):
        raise ValueError(f"MXFP8 scale can only be derived from .weight tensors: {name}")
    return name[: -len(".weight")] + ".weight_scale"


def power_2_scales_to_e8m0(scales: torch.Tensor) -> torch.Tensor:
    """Encode positive power-of-two FP32 scales as E8M0 bytes."""

    exponent_bits = (scales.contiguous().view(torch.int32) >> 23) & 0xFF
    return exponent_bits.to(torch.uint8)


def mxfp8_cast_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to E4M3 weights and E8M0 scales."""

    if weight.ndim != 2:
        raise ValueError(f"MXFP8 expects a 2D tensor, got shape={tuple(weight.shape)}")
    if weight.shape[-1] % 32 != 0:
        raise ValueError(f"MXFP8 requires the last dimension to be divisible by 32, got shape={tuple(weight.shape)}")
    q_weight, scales = blockwise_cast_to_fp8(weight, (1, 32), power_2_scale=True)
    return q_weight, power_2_scales_to_e8m0(scales)


def batched_mxfp8_cast_to_fp8(
    weight: torch.Tensor,
    expert_batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 3D expert tensor to E4M3 weights and E8M0 scales."""

    if weight.ndim != 3:
        raise ValueError(f"Batched MXFP8 expects a 3D tensor, got shape={tuple(weight.shape)}")
    if isinstance(expert_batch_size, bool) or not isinstance(expert_batch_size, int) or expert_batch_size <= 0:
        raise ValueError(f"expert_batch_size must be a positive integer, got {expert_batch_size!r}")
    if weight.shape[-1] % 32 != 0:
        raise ValueError(f"MXFP8 requires the last dimension to be divisible by 32, got shape={tuple(weight.shape)}")

    q_weight, scales = batched_blockwise_cast_to_fp8(
        weight,
        (1, 32),
        power_2_scale=True,
        expert_batch_size=expert_batch_size,
    )
    return q_weight, power_2_scales_to_e8m0(scales)


class Mxfp8ExpertStrategy(QuantizationStrategy):
    """Apply MXFP8 to routed expert weights."""

    mode: ClassVar[str] = SERIALIZED_MXFP8
    quantized_categories: ClassVar[frozenset[WeightCategory]] = frozenset(
        {"routed_expert_gate", "routed_expert_up", "routed_expert_down"}
    )
    receiver_suffixes: ClassVar[dict[str, str]] = {
        ".weight": "",
        ".weight_scale": "_scale",
    }
    supported_model_types: ClassVar[frozenset[str]] = frozenset(
        {"qwen3_moe", "qwen3_5_moe", "qwen3_5_moe_text"}
    )
    reject_unknown_routed_experts: ClassVar[bool] = True
    required_model_dtype: ClassVar[str | None] = "bfloat16"
    vllm_quantization: ClassVar[str] = "modelopt_mxfp8"

    def validate_model_type(self, model_type: str, model_path: str | None = None) -> None:
        """Require a registered Qwen MoE layout."""

        if not self.supports_model_type(model_type):
            supported = ", ".join(sorted(self.supported_model_types))
            raise ValueError(f"Serialized MXFP8 does not support model_type={model_type!r}; supported: {supported}")

    def build_te_recipe(self, *, persistent: bool) -> dict:
        """Build Transformer Engine MXFP8 settings."""

        return build_mxfp8_te_recipe(persistent=persistent)

    def configure_megatron_provider(self, provider) -> None:
        """Set provider fields required by MXFP8 training."""

        configure_mxfp8_provider(provider)

    def serialize_weight(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        batched_experts: bool,
        context: Any = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield an MXFP8 weight and its E8M0 scale tensor."""

        del context
        if batched_experts:
            q_weight, scale = batched_mxfp8_cast_to_fp8(tensor)
        else:
            q_weight, scale = mxfp8_cast_to_fp8(tensor)
        yield name, q_weight
        yield mxfp8_scale_name_for_weight(name), scale

    def vllm_quantization_config(
        self,
        inference_config: Any,
        hf_config: Any,
        layout: QuantizedModelLayout,
    ) -> dict[str, Any]:
        """Return ModelOpt MXFP8 settings for vLLM."""

        del inference_config, hf_config, layout
        return get_serialized_mxfp8_quantization_config()
