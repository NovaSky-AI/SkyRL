"""Compatibility API for serialized FP8 weight formats."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterator, Sequence

import torch

from .serialization import (
    BLOCKWISE_128X128,
    LEGACY_BATCHED_MOE_PREFIX,
    MXFP8_1X32,
    QWEN35_QUANTIZATION_SPEC,
    SERIALIZED_BLOCKWISE_FP8,
    SERIALIZED_MXFP8,
    BlockwiseFp8Strategy,
    Mxfp8Strategy,
    batched_blockwise_cast_to_fp8,
    batched_mxfp8_cast_to_fp8,
    blockwise_cast_to_fp8,
    get_hf_model_type,
    get_qwen35_fp8_ignored_layers,
    get_serialized_fp8_quantization_config,
    get_serialized_mxfp8_quantization_config,
    is_qwen35_config,
    iter_serialized_weight_tensors,
    mxfp8_cast_to_fp8,
    mxfp8_scale_name_for_weight,
    power_2_scales_to_e8m0,
    resolve_model_quantization_spec,
    scale_name_for_weight,
    should_use_serialized_weights,
    use_amax_epsilon_default,
    use_power_2_scales_default,
)
from .serialization.blockwise_fp8 import normalize_block_size
from .serialization.mxfp8 import EXPERT_ONLY_MXFP8_IGNORED_MODULES

SKYRL_BATCHED_MOE_FP8_PREFIX = LEGACY_BATCHED_MOE_PREFIX
_EXPERT_ONLY_MXFP8_IGNORED_MODULES = EXPERT_ONLY_MXFP8_IGNORED_MODULES
_power_2_scales_to_e8m0 = power_2_scales_to_e8m0
_normalize_block_size = normalize_block_size


@dataclass(frozen=True)
class MoeArchitectureSpec:
    gate_up_suffixes: tuple[str, ...]
    down_suffix: str
    gate_up_fused: bool
    vllm_projection_names: tuple[str, ...]
    batched: bool


_QWEN35_MOE_SPEC = MoeArchitectureSpec(
    gate_up_suffixes=(".gate_up_proj",),
    down_suffix=".down_proj",
    gate_up_fused=True,
    vllm_projection_names=("gate_proj", "up_proj", "down_proj"),
    batched=True,
)
_QWEN3_MOE_SPEC = _QWEN35_MOE_SPEC
MOE_ARCHITECTURE_SPECS = {
    "qwen3_moe": _QWEN3_MOE_SPEC,
    "qwen3_5_moe": _QWEN35_MOE_SPEC,
    "qwen3_5_moe_text": _QWEN35_MOE_SPEC,
}


def get_moe_architecture_spec(model_type: str) -> MoeArchitectureSpec:
    try:
        return MOE_ARCHITECTURE_SPECS[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(MOE_ARCHITECTURE_SPECS))
        raise ValueError(
            f"Serialized MXFP8 does not support model_type={model_type!r}; supported: {supported}"
        ) from exc


@dataclass(frozen=True)
class SerializedFp8Config:
    """Compatibility configuration for serialized FP8 rollout weights."""

    scaling_mode: str = BLOCKWISE_128X128
    expert_only: bool = False
    model_type: str | None = None
    weight_block_size: tuple[int, int] = (128, 128)
    power_2_scale: bool = field(default_factory=use_power_2_scales_default)
    amax_epsilon: float = field(default_factory=use_amax_epsilon_default)

    def __post_init__(self) -> None:
        if self.scaling_mode not in (BLOCKWISE_128X128, MXFP8_1X32):
            raise ValueError(f"scaling_mode must be {BLOCKWISE_128X128!r} or {MXFP8_1X32!r}, got {self.scaling_mode!r}")
        if type(self.expert_only) is not bool:
            raise ValueError(f"expert_only must be a bool, got {self.expert_only!r}")
        object.__setattr__(self, "weight_block_size", normalize_block_size(self.weight_block_size))
        if type(self.power_2_scale) is not bool:
            raise ValueError(f"power_2_scale must be a bool, got {self.power_2_scale!r}")
        if not isfinite(self.amax_epsilon) or self.amax_epsilon < 0:
            raise ValueError(f"amax_epsilon must be finite and non-negative, got {self.amax_epsilon}")
        if self.expert_only:
            if not self.model_type:
                raise ValueError("expert_only serialized FP8 requires model_type")
            get_moe_architecture_spec(self.model_type)


def serialized_fp8_config_for_mode(mode: str, *, model_type: str | None = None) -> SerializedFp8Config:
    if mode == SERIALIZED_BLOCKWISE_FP8:
        return SerializedFp8Config(model_type=model_type)
    if mode == SERIALIZED_MXFP8:
        return SerializedFp8Config(
            scaling_mode=MXFP8_1X32,
            expert_only=True,
            model_type=model_type,
            weight_block_size=(1, 32),
        )
    raise ValueError(f"Unsupported fp8_weight_sync_mode={mode!r}")


def should_use_serialized_fp8(mode: str | None) -> bool:
    return should_use_serialized_weights(mode)


def is_quantizable_weight(name: str, tensor: torch.Tensor, *, expert_only: bool = False) -> bool:
    if expert_only or not name.endswith(".weight") or tensor.ndim != 2:
        return False
    return is_quantizable_weight_shape(name, tensor.shape)


def is_quantizable_weight_shape(name: str, shape: Sequence[int]) -> bool:
    return QWEN35_QUANTIZATION_SPEC.should_quantize(name, shape)


def batched_moe_expert_spec(
    name: str,
    model_type: str | None = None,
) -> tuple[str, tuple[str, ...], bool] | None:
    legacy_spec = _QWEN35_MOE_SPEC if model_type is None else get_moe_architecture_spec(model_type)
    if not legacy_spec.batched or ".mlp.experts" not in name:
        return None
    gate_up_suffix = legacy_spec.gate_up_suffixes[0]
    if name.endswith(gate_up_suffix):
        return name[: -len(gate_up_suffix)], legacy_spec.vllm_projection_names[:2], legacy_spec.gate_up_fused
    if name.endswith(legacy_spec.down_suffix):
        return name[: -len(legacy_spec.down_suffix)], (legacy_spec.vllm_projection_names[-1],), False
    return None


def _strategy_for_config(config: SerializedFp8Config):
    if config.scaling_mode == MXFP8_1X32:
        return Mxfp8Strategy(expert_only=config.expert_only)
    return BlockwiseFp8Strategy(
        weight_block_size=config.weight_block_size,
        power_2_scale=config.power_2_scale,
        amax_epsilon=config.amax_epsilon,
        expert_only=config.expert_only,
    )


def _model_spec_for_config(config: SerializedFp8Config):
    if config.model_type is None:
        return QWEN35_QUANTIZATION_SPEC
    model_spec = resolve_model_quantization_spec(config.model_type)
    if model_spec is None:
        get_moe_architecture_spec(config.model_type)
    return model_spec


def iter_batched_moe_expert_fp8_tensors(
    name: str,
    tensor: torch.Tensor,
    config: SerializedFp8Config,
) -> Iterator[tuple[str, torch.Tensor]]:
    if batched_moe_expert_spec(name, config.model_type) is None:
        raise ValueError(f"Not a batched MoE expert tensor: {name}")
    yield from iter_serialized_fp8_tensors(name, tensor, torch.bfloat16, config)


def iter_serialized_fp8_tensors(
    name: str,
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    config: SerializedFp8Config,
) -> Iterator[tuple[str, torch.Tensor]]:
    model_spec = _model_spec_for_config(config)
    model_type = config.model_type or "qwen3_5_moe_text"
    yield from iter_serialized_weight_tensors(
        name,
        tensor,
        target_dtype,
        model_spec,
        _strategy_for_config(config),
        model_type=model_type,
    )


__all__ = [
    "BLOCKWISE_128X128",
    "MOE_ARCHITECTURE_SPECS",
    "MXFP8_1X32",
    "SERIALIZED_BLOCKWISE_FP8",
    "SERIALIZED_MXFP8",
    "SKYRL_BATCHED_MOE_FP8_PREFIX",
    "MoeArchitectureSpec",
    "SerializedFp8Config",
    "batched_blockwise_cast_to_fp8",
    "batched_moe_expert_spec",
    "batched_mxfp8_cast_to_fp8",
    "blockwise_cast_to_fp8",
    "get_hf_model_type",
    "get_moe_architecture_spec",
    "get_qwen35_fp8_ignored_layers",
    "get_serialized_fp8_quantization_config",
    "get_serialized_mxfp8_quantization_config",
    "is_quantizable_weight",
    "is_quantizable_weight_shape",
    "is_qwen35_config",
    "iter_batched_moe_expert_fp8_tensors",
    "iter_serialized_fp8_tensors",
    "mxfp8_cast_to_fp8",
    "mxfp8_scale_name_for_weight",
    "scale_name_for_weight",
    "serialized_fp8_config_for_mode",
    "should_use_serialized_fp8",
    "use_amax_epsilon_default",
    "use_power_2_scales_default",
]
