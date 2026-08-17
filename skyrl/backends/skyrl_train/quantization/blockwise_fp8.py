"""Blockwise FP8 serialization and vLLM configuration."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from operator import index
from typing import Any, ClassVar

import torch

from .base import ModelQuantizationLayout, QuantizationStrategy

BLOCKWISE_FP8 = "blockwise"


def use_power_2_scales_default() -> bool:
    scale_mode = os.getenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    if scale_mode not in {"0", "1"}:
        raise ValueError(
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES must be '0' (power-of-2) "
            f"or '1' (FP32 scales), got {scale_mode!r}"
        )
    return scale_mode == "0"


def normalize_block_size(block_size: Sequence[int]) -> tuple[int, int]:
    try:
        raw_values = tuple(block_size)
        if any(isinstance(value, bool) for value in raw_values):
            raise TypeError
        values = tuple(index(value) for value in raw_values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"weight_block_size must contain exactly two positive integers, got {block_size!r}") from exc
    if len(values) != 2 or any(value <= 0 for value in values):
        raise ValueError(f"weight_block_size must contain exactly two positive integers, got {block_size!r}")
    return values


def blockwise_cast_to_fp8(
    weight: torch.Tensor,
    block_size: Sequence[int],
    power_2_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to E4M3 data and block inverse scales."""

    if weight.ndim != 2:
        raise ValueError(f"Blockwise FP8 expects a 2D tensor, got shape={tuple(weight.shape)}")

    block_m, block_n = normalize_block_size(block_size)
    rows, cols = weight.shape
    padded_rows = ((rows + block_m - 1) // block_m) * block_m
    padded_cols = ((cols + block_n - 1) // block_n) * block_n

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    weight_fp32 = weight.detach().to(torch.float32)
    if padded_rows != rows or padded_cols != cols:
        padded = weight_fp32.new_zeros((padded_rows, padded_cols))
        padded[:rows, :cols].copy_(weight_fp32)
    else:
        padded = weight_fp32

    blocks = padded.view(padded_rows // block_m, block_m, padded_cols // block_n, block_n)
    blocks = blocks.permute(0, 2, 1, 3)
    scale = blocks.abs().amax(dim=(2, 3)).clamp(min=1e-10) / fp8_info.max
    if power_2_scale:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    q_blocks = (blocks / scale[:, :, None, None]).clamp(min=fp8_info.min, max=fp8_info.max)
    q_padded = q_blocks.to(torch.float8_e4m3fn).permute(0, 2, 1, 3).contiguous().view(padded_rows, padded_cols)
    return q_padded[:rows, :cols].contiguous(), scale.to(torch.float32).contiguous()


def batched_blockwise_cast_to_fp8(
    weight: torch.Tensor,
    block_size: Sequence[int],
    power_2_scale: bool = False,
    expert_batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 3D expert tensor in bounded expert batches."""

    if weight.ndim != 3:
        raise ValueError(f"Batched blockwise FP8 expects a 3D tensor, got shape={tuple(weight.shape)}")
    if isinstance(expert_batch_size, bool) or not isinstance(expert_batch_size, int) or expert_batch_size <= 0:
        raise ValueError(f"expert_batch_size must be a positive integer, got {expert_batch_size!r}")

    block_m, block_n = normalize_block_size(block_size)
    num_experts, rows, cols = weight.shape
    padded_rows = ((rows + block_m - 1) // block_m) * block_m
    padded_cols = ((cols + block_n - 1) // block_n) * block_n
    row_blocks = padded_rows // block_m
    col_blocks = padded_cols // block_n

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    q_weight = torch.empty(weight.shape, dtype=torch.float8_e4m3fn, device=weight.device)
    scales = torch.empty((num_experts, row_blocks, col_blocks), dtype=torch.float32, device=weight.device)

    for start in range(0, num_experts, expert_batch_size):
        end = min(start + expert_batch_size, num_experts)
        weight_fp32 = weight[start:end].detach().to(torch.float32).contiguous()
        if padded_rows != rows or padded_cols != cols:
            padded = weight_fp32.new_zeros((end - start, padded_rows, padded_cols))
            padded[:, :rows, :cols].copy_(weight_fp32)
        else:
            padded = weight_fp32

        blocks = padded.view(end - start, row_blocks, block_m, col_blocks, block_n)
        blocks = blocks.permute(0, 1, 3, 2, 4)
        scale = blocks.abs().amax(dim=(3, 4)).clamp(min=1e-10) / fp8_info.max
        if power_2_scale:
            scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
        q_blocks = (blocks / scale[:, :, :, None, None]).clamp(min=fp8_info.min, max=fp8_info.max)
        q_padded = (
            q_blocks.to(torch.float8_e4m3fn)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(end - start, padded_rows, padded_cols)
        )
        q_weight[start:end].copy_(q_padded[:, :rows, :cols])
        scales[start:end].copy_(scale)

    return q_weight, scales


def scale_name_for_weight(name: str) -> str:
    if not name.endswith(".weight"):
        raise ValueError(f"FP8 scale can only be derived from .weight tensors: {name}")
    return name[: -len(".weight")] + ".weight_scale_inv"


@dataclass(frozen=True)
class BlockwiseFp8Strategy(QuantizationStrategy):
    """Apply 128x128 blockwise FP8 to supported linear weights."""

    weight_block_size: tuple[int, int] = (128, 128)
    power_2_scale: bool = field(default_factory=use_power_2_scales_default)

    mode: ClassVar[str] = BLOCKWISE_FP8
    target_names: ClassVar[frozenset[str]] = frozenset(
        {"linear", "routed_expert_gate", "routed_expert_up", "routed_expert_down"}
    )
    receiver_suffixes: ClassVar[dict[str, str]] = {
        ".weight": "",
        ".weight_scale_inv": "_scale_inv",
    }
    vllm_quantization: ClassVar[str] = "fp8"

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight_block_size", normalize_block_size(self.weight_block_size))
        if type(self.power_2_scale) is not bool:
            raise ValueError(f"power_2_scale must be a bool, got {self.power_2_scale!r}")

    def build_runtime_env(self) -> dict[str, str]:
        env = {"NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "0" if self.power_2_scale else "1"}
        expected_e8m0 = "1" if self.power_2_scale else "0"
        configured_e8m0 = os.getenv("VLLM_USE_DEEP_GEMM_E8M0", expected_e8m0)
        if configured_e8m0 != expected_e8m0:
            raise ValueError(
                f"Blockwise FP8 with power_2_scale={self.power_2_scale} requires "
                f"VLLM_USE_DEEP_GEMM_E8M0={expected_e8m0}"
            )
        env["VLLM_USE_DEEP_GEMM_E8M0"] = configured_e8m0
        return env

    def serialize_weight(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        batched_experts: bool,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        if batched_experts:
            q_weight, scale = batched_blockwise_cast_to_fp8(
                tensor,
                self.weight_block_size,
                self.power_2_scale,
            )
        else:
            q_weight, scale = blockwise_cast_to_fp8(
                tensor,
                self.weight_block_size,
                self.power_2_scale,
            )
        yield name, q_weight
        yield scale_name_for_weight(name), scale

    def vllm_quantization_config(
        self,
        inference_config: Any,
        hf_config: Any,
        layout: ModelQuantizationLayout,
    ) -> dict[str, Any]:
        del inference_config
        config: dict[str, Any] = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": list(self.weight_block_size),
        }
        ignored_layers = layout.ignored_layers(hf_config)
        if ignored_layers:
            config["ignored_layers"] = ignored_layers
        return config
