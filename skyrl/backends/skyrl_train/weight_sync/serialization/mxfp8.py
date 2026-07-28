"""Serialized expert MXFP8 strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Iterator

import torch

from .base import (
    ModelQuantizationSpec,
    QuantizationTarget,
    ReceiverTensorRole,
    SerializedWeightStrategy,
    WeightKind,
)
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


def get_serialized_mxfp8_quantization_config() -> dict:
    return {
        "quant_method": "modelopt",
        "quant_algo": "MXFP8",
        "ignore": list(EXPERT_ONLY_MXFP8_IGNORED_MODULES),
    }


def mxfp8_scale_name_for_weight(name: str) -> str:
    if not name.endswith(".weight"):
        raise ValueError(f"MXFP8 scale can only be derived from .weight tensors: {name}")
    return name[: -len(".weight")] + ".weight_scale"


def power_2_scales_to_e8m0(scales: torch.Tensor) -> torch.Tensor:
    """Encode positive power-of-two FP32 scales as biased E8M0 exponents."""

    exponent_bits = (scales.contiguous().view(torch.int32) >> 23) & 0xFF
    return exponent_bits.to(torch.uint8)


def mxfp8_cast_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


@dataclass(frozen=True)
class Mxfp8Strategy(SerializedWeightStrategy):
    expert_only: bool = True

    mode: ClassVar[str] = SERIALIZED_MXFP8
    receiver_tensor_roles: ClassVar[tuple[ReceiverTensorRole, ...]] = (
        ReceiverTensorRole(".weight", ""),
        ReceiverTensorRole(".weight_scale", "_scale"),
    )
    supported_model_types: ClassVar[frozenset[str]] = frozenset({"qwen3_moe", "qwen3_5_moe", "qwen3_5_moe_text"})
    reject_unknown_routed_experts: ClassVar[bool] = True
    required_model_dtype: ClassVar[str | None] = "bfloat16"
    vllm_quantization: ClassVar[str] = "modelopt_mxfp8"
    use_legacy_wire_prefix: ClassVar[bool] = True

    def supports(self, target: QuantizationTarget) -> bool:
        if target.kind is WeightKind.ROUTED_EXPERT:
            return True
        return target.kind is WeightKind.LINEAR and not self.expert_only

    def validate_model_type(self, model_type: str, model_path: str | None = None) -> None:
        if not self.supports_model_type(model_type):
            supported = ", ".join(sorted(self.supported_model_types))
            raise ValueError(f"Serialized MXFP8 does not support model_type={model_type!r}; supported: {supported}")

    def serialize(self, target: QuantizationTarget) -> Iterator[tuple[str, torch.Tensor]]:
        if target.batched_experts:
            q_weight, scale = batched_mxfp8_cast_to_fp8(target.tensor)
        else:
            q_weight, scale = mxfp8_cast_to_fp8(target.tensor)
        yield target.checkpoint_name, q_weight
        yield mxfp8_scale_name_for_weight(target.checkpoint_name), scale

    def vllm_quantization_config(
        self,
        inference_config: Any,
        hf_config: Any,
        model_spec: ModelQuantizationSpec,
    ) -> dict[str, Any]:
        del inference_config, hf_config, model_spec
        return get_serialized_mxfp8_quantization_config()
