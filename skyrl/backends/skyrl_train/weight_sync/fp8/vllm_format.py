"""Serialized wire format for vLLM blockwise FP8 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterator, Sequence

import torch

from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import ModelFp8Spec
from skyrl.backends.skyrl_train.weight_sync.fp8.quantize import (
    batched_blockwise_cast_to_fp8,
    blockwise_cast_to_fp8,
    normalize_block_size,
    use_amax_epsilon_default,
    use_power_2_scales_default,
)

SERIALIZED_BLOCKWISE_FP8 = "serialized_blockwise"
# Internal wire-format marker for Qwen3.5 MoE tensors that remain batched over
# experts. The receiver strips this marker and routes the tensor directly to
# vLLM's fused-MoE parameter loader instead of the ordinary HF-name loader.
SKYRL_BATCHED_MOE_FP8_PREFIX = "__skyrl_batched_moe_fp8__:"


@dataclass(frozen=True)
class SerializedFp8Config:
    """Configuration for serialized FP8 rollout weight sync.

    ``spec`` is the per-model quantization policy, resolved once from the HF
    config via ``resolve_fp8_spec``; the tensor iterators require it.
    """

    weight_block_size: tuple[int, int] = (128, 128)
    power_2_scale: bool = field(default_factory=use_power_2_scales_default)
    amax_epsilon: float = field(default_factory=use_amax_epsilon_default)
    spec: ModelFp8Spec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight_block_size", normalize_block_size(self.weight_block_size))
        if type(self.power_2_scale) is not bool:
            raise ValueError(f"power_2_scale must be a bool, got {self.power_2_scale!r}")
        if not isfinite(self.amax_epsilon) or self.amax_epsilon < 0:
            raise ValueError(f"amax_epsilon must be finite and non-negative, got {self.amax_epsilon}")

    def require_spec(self) -> ModelFp8Spec:
        if self.spec is None:
            raise ValueError(
                "SerializedFp8Config.spec is not set; resolve the model spec with "
                "resolve_fp8_spec(hf_config) before serializing weights"
            )
        return self.spec


def get_serialized_fp8_quantization_config(
    weight_block_size: Sequence[int] = (128, 128),
    ignored_layers: Sequence[str] | None = None,
) -> dict:
    """Return vLLM's Hugging Face quantization config for serialized FP8."""

    block_m, block_n = normalize_block_size(weight_block_size)
    qconfig = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [block_m, block_n],
    }
    if ignored_layers:
        qconfig["ignored_layers"] = list(ignored_layers)
    return qconfig


def should_use_serialized_fp8(mode: str | None) -> bool:
    return mode == SERIALIZED_BLOCKWISE_FP8


def scale_name_for_weight(name: str) -> str:
    if not name.endswith(".weight"):
        raise ValueError(f"FP8 scale can only be derived from .weight tensors: {name}")
    return name[: -len(".weight")] + ".weight_scale_inv"


def iter_batched_moe_expert_fp8_tensors(
    name: str,
    tensor: torch.Tensor,
    config: SerializedFp8Config,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Convert a batched expert tensor without expanding expert names.

    The old wire format emitted one weight and one scale tensor for every
    expert/projection pair. Keeping the expert dimension intact reduces each
    routed MoE layer from ``6 * num_experts`` tensors to six and lets vLLM use
    its fused 3D loader.
    """
    moe_spec = config.require_spec().moe_expert_spec(name)
    if moe_spec is None:
        raise ValueError(f"Not a batched MoE expert tensor: {name}")
    if tensor.ndim != 3:
        raise ValueError(f"Batched MoE expert tensor must be 3D, got shape={tuple(tensor.shape)}")
    if moe_spec.split_dim is not None:
        num_projections = len(moe_spec.projections)
        if tensor.shape[moe_spec.split_dim] % num_projections != 0:
            raise ValueError(
                f"Batched MoE tensor dim {moe_spec.split_dim} must split evenly across "
                f"{num_projections} projections, got shape={tuple(tensor.shape)}"
            )
        projection_tensors = torch.chunk(tensor, num_projections, dim=moe_spec.split_dim)
    else:
        projection_tensors = (tensor,)

    for proj, projection_tensor in zip(moe_spec.projections, projection_tensors):
        q_weight, scale = batched_blockwise_cast_to_fp8(
            projection_tensor,
            config.weight_block_size,
            config.power_2_scale,
            config.amax_epsilon,
        )
        weight_name = f"{SKYRL_BATCHED_MOE_FP8_PREFIX}{moe_spec.experts_base}.{proj.hf_name}.weight"
        yield weight_name, q_weight
        yield scale_name_for_weight(weight_name), scale


def iter_serialized_fp8_tensors(
    name: str,
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    config: SerializedFp8Config,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield vLLM checkpoint tensors for one Megatron-exported weight."""

    spec = config.require_spec()
    if spec.moe_expert_spec(name) is not None:
        yield from iter_batched_moe_expert_fp8_tensors(name, tensor, config)
        return

    if tensor.ndim == 2 and spec.should_quantize(name, tuple(tensor.shape)):
        q_weight, scale = blockwise_cast_to_fp8(
            tensor,
            config.weight_block_size,
            config.power_2_scale,
            config.amax_epsilon,
        )
        yield name, q_weight
        yield scale_name_for_weight(name), scale
        return

    yield name, tensor.to(dtype=target_dtype)
