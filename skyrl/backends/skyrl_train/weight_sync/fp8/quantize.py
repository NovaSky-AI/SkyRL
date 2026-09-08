"""Blockwise FP8 quantization kernels and scale-mode helpers."""

from __future__ import annotations

import os
from operator import index
from typing import Sequence

import torch


def use_power_2_scales_default() -> bool:
    """Return whether rollout weights use power-of-two block scales.

    The setting must match Transformer Engine. Hopper defaults to FP32 scales;
    Blackwell launchers select power-of-two scales by setting
    ``NVTE_FP8_BLOCK_SCALING_FP32_SCALES=0``.
    """

    scale_mode = os.getenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    if scale_mode not in {"0", "1"}:
        raise ValueError(
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES must be '0' (power-of-2) " f"or '1' (FP32 scales), got {scale_mode!r}"
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
    """Quantize a 2D tensor to vLLM's blockwise E4M3 checkpoint format.

    Returns ``weight_scale_inv`` such that
    ``weight ~= qweight.float() * scale``. Power-of-two mode rounds scales up
    to match Transformer Engine's UE8M0 rule.
    """

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
    # Nonzero floor keeps all-zero blocks from degenerating the scale.
    scale = blocks.abs().amax(dim=(2, 3)).clamp(min=1e-10) / fp8_info.max
    if power_2_scale:
        # Rounding up preserves range and matches TE's power-of-two scale rule.
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    q_blocks = (blocks / scale[:, :, None, None]).clamp(min=fp8_info.min, max=fp8_info.max)
    q_blocks = q_blocks.to(torch.float8_e4m3fn)
    q_padded = q_blocks.permute(0, 2, 1, 3).contiguous().view(padded_rows, padded_cols)
    q_weight = q_padded[:rows, :cols].contiguous()
    return q_weight, scale.to(torch.float32).contiguous()


def batched_blockwise_cast_to_fp8(
    weight: torch.Tensor,
    block_size: Sequence[int],
    power_2_scale: bool = False,
    expert_batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 3D ``[experts, rows, cols]`` tensor blockwise.

    Quantizing several experts per operation avoids launching the full 2D
    conversion pipeline once per expert, while bounded batches limit peak FP32
    workspace.
    """

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
    scales = torch.empty(
        (num_experts, row_blocks, col_blocks),
        dtype=torch.float32,
        device=weight.device,
    )

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
        q_blocks = q_blocks.to(torch.float8_e4m3fn)
        q_padded = q_blocks.permute(0, 1, 3, 2, 4).contiguous().view(end - start, padded_rows, padded_cols)
        q_weight[start:end].copy_(q_padded[:, :rows, :cols])
        scales[start:end].copy_(scale)

    return q_weight, scales
