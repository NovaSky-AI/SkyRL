"""Blockwise FP8 serialization and vLLM configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import isfinite
from operator import index
from typing import Any, ClassVar, Iterator, Sequence

import torch

from .base import QuantizationStrategy, QuantizedModelLayout, WeightCategory
from .model_layouts import is_qwen35_config

SERIALIZED_BLOCKWISE_FP8 = "serialized_blockwise"
BLOCKWISE_128X128 = "blockwise_128x128"

_QWEN35_UNQUANTIZED_LINEAR_SUFFIXES = (".in_proj_b", ".in_proj_a")
_QWEN35_LINEAR_ATTN_PREFIX_TEMPLATES = (
    "{model_prefix}.layers.{layer_idx}.linear_attn",
    "{model_prefix}.language_model.layers.{layer_idx}.linear_attn",
)
_QWEN35_VISION_ATTN_PROJ_PREFIX_TEMPLATES = ("{model_prefix}.visual.blocks.{block_idx}.attn.proj",)


def use_power_2_scales_default() -> bool:
    """Return whether rollout weights use power-of-two block scales."""

    scale_mode = os.getenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    if scale_mode not in {"0", "1"}:
        raise ValueError(
            f"NVTE_FP8_BLOCK_SCALING_FP32_SCALES must be '0' (power-of-2) or '1' (FP32 scales), got {scale_mode!r}"
        )
    return scale_mode == "0"


def use_amax_epsilon_default() -> float:
    """Read the TE blockwise amax floor used by the training quantizer."""

    raw_value = os.getenv("NVTE_FP8_BLOCK_AMAX_EPSILON", "0")
    try:
        epsilon = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"NVTE_FP8_BLOCK_AMAX_EPSILON must be a float, got {raw_value!r}") from exc
    if not isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"NVTE_FP8_BLOCK_AMAX_EPSILON must be finite and non-negative, got {epsilon}")
    return epsilon


def normalize_block_size(block_size: Sequence[int]) -> tuple[int, int]:
    """Validate and normalize a two-dimensional quantization block size."""

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


def scale_name_for_weight(name: str) -> str:
    """Return the checkpoint name for a blockwise FP8 inverse scale."""

    if not name.endswith(".weight"):
        raise ValueError(f"FP8 scale can only be derived from .weight tensors: {name}")
    return name[: -len(".weight")] + ".weight_scale_inv"


def get_serialized_fp8_quantization_config(
    weight_block_size: Sequence[int] = (128, 128),
    ignored_layers: Sequence[str] | None = None,
) -> dict:
    """Return vLLM's blockwise FP8 checkpoint configuration."""

    block_m, block_n = normalize_block_size(weight_block_size)
    config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [block_m, block_n],
    }
    if ignored_layers:
        config["ignored_layers"] = list(ignored_layers)
    return config


def get_qwen35_fp8_ignored_layers(hf_config: Any, model_prefix: str = "model") -> list[str]:
    """Return Qwen3.5 modules incompatible with blockwise FP8 loading."""

    if not is_qwen35_config(hf_config):
        return []
    text_config = getattr(hf_config, "text_config", None) or getattr(hf_config, "language_config", None) or hf_config
    ignored: list[str] = []
    for layer_idx, layer_type in enumerate(list(getattr(text_config, "layer_types", []) or [])):
        if layer_type != "linear_attention":
            continue
        for template in _QWEN35_LINEAR_ATTN_PREFIX_TEMPLATES:
            prefix = template.format(model_prefix=model_prefix, layer_idx=layer_idx)
            ignored.extend(f"{prefix}{suffix}" for suffix in _QWEN35_UNQUANTIZED_LINEAR_SUFFIXES)

    vision_config = getattr(hf_config, "vision_config", None) or getattr(hf_config, "visual_config", None)
    vision_depth = 0
    if vision_config is not None:
        for attr in ("depth", "num_hidden_layers", "num_layers"):
            value = getattr(vision_config, attr, None)
            if isinstance(value, int) and value > 0:
                vision_depth = value
                break
    for block_idx in range(vision_depth):
        for template in _QWEN35_VISION_ATTN_PROJ_PREFIX_TEMPLATES:
            ignored.append(template.format(model_prefix=model_prefix, block_idx=block_idx))
    return ignored


def blockwise_cast_to_fp8(
    weight: torch.Tensor,
    block_size: Sequence[int],
    power_2_scale: bool = False,
    amax_epsilon: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to blockwise E4M3 weights and FP32 scales."""

    if weight.ndim != 2:
        raise ValueError(f"Blockwise FP8 expects a 2D tensor, got shape={tuple(weight.shape)}")
    if not isfinite(amax_epsilon) or amax_epsilon < 0:
        raise ValueError(f"amax_epsilon must be finite and non-negative, got {amax_epsilon}")

    block_m, block_n = normalize_block_size(block_size)
    rows, cols = weight.shape
    padded_rows = ((rows + block_m - 1) // block_m) * block_m
    padded_cols = ((cols + block_n - 1) // block_n) * block_n

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    weight_fp32 = weight.detach().to(torch.float32).contiguous() 
    if padded_rows != rows or padded_cols != cols:
        padded = weight_fp32.new_zeros((padded_rows, padded_cols))
        padded[:rows, :cols].copy_(weight_fp32)
    else:
        padded = weight_fp32

    blocks = padded.view(padded_rows // block_m, block_m, padded_cols // block_n, block_n).permute(0, 2, 1, 3)
    scale = blocks.abs().amax(dim=(2, 3)).clamp(min=max(amax_epsilon, 1e-10)) / fp8_info.max
    if power_2_scale:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    q_blocks = (blocks / scale[:, :, None, None]).clamp(min=fp8_info.min, max=fp8_info.max)
    q_padded = q_blocks.to(torch.float8_e4m3fn).permute(0, 2, 1, 3).contiguous().view(padded_rows, padded_cols)
    return q_padded[:rows, :cols].contiguous(), scale.to(torch.float32).contiguous()


def batched_blockwise_cast_to_fp8(
    weight: torch.Tensor,
    block_size: Sequence[int],
    power_2_scale: bool = False,
    amax_epsilon: float = 0.0,
    expert_batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 3D expert tensor in bounded expert batches."""

    if weight.ndim != 3:
        raise ValueError(f"Batched blockwise FP8 expects a 3D tensor, got shape={tuple(weight.shape)}")
    if not isfinite(amax_epsilon) or amax_epsilon < 0:
        raise ValueError(f"amax_epsilon must be finite and non-negative, got {amax_epsilon}")
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

        blocks = padded.view(end - start, row_blocks, block_m, col_blocks, block_n).permute(0, 1, 3, 2, 4)
        scale = blocks.abs().amax(dim=(3, 4)).clamp(min=max(amax_epsilon, 1e-10)) / fp8_info.max
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


@dataclass(frozen=True)
class BlockwiseFp8LinearStrategy(QuantizationStrategy):
    """Apply 128x128 blockwise FP8 to supported linear weights."""

    weight_block_size: tuple[int, int] = (128, 128)
    power_2_scale: bool = field(default_factory=use_power_2_scales_default)
    amax_epsilon: float = field(default_factory=use_amax_epsilon_default)

    mode: ClassVar[str] = SERIALIZED_BLOCKWISE_FP8
    quantized_categories: ClassVar[frozenset[WeightCategory]] = frozenset(
        {"linear", "routed_expert_gate", "routed_expert_up", "routed_expert_down"}
    )
    receiver_suffixes: ClassVar[dict[str, str]] = {
        ".weight": "",
        ".weight_scale_inv": "_scale_inv",
    }
    supported_model_types: ClassVar[frozenset[str]] = frozenset(
        {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
    )
    vllm_quantization: ClassVar[str] = "fp8"

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight_block_size", normalize_block_size(self.weight_block_size))
        if type(self.power_2_scale) is not bool:
            raise ValueError(f"power_2_scale must be a bool, got {self.power_2_scale!r}")
        if not isfinite(self.amax_epsilon) or self.amax_epsilon < 0:
            raise ValueError(f"amax_epsilon must be finite and non-negative, got {self.amax_epsilon}")

    def validate_model_type(self, model_type: str, model_path: str | None = None) -> None:
        """Require a Qwen3.5 checkpoint layout."""

        if not self.supports_model_type(model_type):
            raise ValueError(
                "Serialized FP8 weight sync currently supports only Qwen3.5 checkpoint layouts; "
                f"model_path={model_path!r}"
            )

    def build_runtime_env(self) -> dict[str, str]:
        """Return the shared TE/vLLM block-scale environment."""

        env = {
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "0" if self.power_2_scale else "1",
            "NVTE_FP8_BLOCK_AMAX_EPSILON": str(self.amax_epsilon),
        }
        if not self.power_2_scale:
            e8m0_mode = os.getenv("VLLM_USE_DEEP_GEMM_E8M0", "0")
            if e8m0_mode != "0":
                raise ValueError(
                    "FP32 block scales require VLLM_USE_DEEP_GEMM_E8M0=0 so vLLM "
                    "does not requantize them to power-of-2 scales."
                )
            env["VLLM_USE_DEEP_GEMM_E8M0"] = e8m0_mode
        return env

    def serialize_weight(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        batched_experts: bool,
        context: Any = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield a quantized weight and its inverse-scale tensor."""

        del context
        if batched_experts:
            q_weight, scale = batched_blockwise_cast_to_fp8(
                tensor,
                self.weight_block_size,
                self.power_2_scale,
                self.amax_epsilon,
            )
        else:
            q_weight, scale = blockwise_cast_to_fp8(
                tensor,
                self.weight_block_size,
                self.power_2_scale,
                self.amax_epsilon,
            )
        yield name, q_weight
        yield scale_name_for_weight(name), scale

    def vllm_quantization_config(
        self,
        inference_config: Any,
        hf_config: Any,
        layout: QuantizedModelLayout,
    ) -> dict[str, Any]:
        """Return blockwise FP8 settings for vLLM checkpoint loading."""

        del inference_config, layout
        return get_serialized_fp8_quantization_config(
            self.weight_block_size,
            get_qwen35_fp8_ignored_layers(hf_config),
        )
