"""Serialized FP8 weight-sync helpers.

This module prepares Megatron-exported HF/vLLM weights for rollout engines that
expect an FP8 checkpoint-style payload: FP8 weight tensors plus explicit scale
tensors. The sync payload is quantized from the exported master weights whether
Megatron uses BF16 or persistent FP8 compute parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import isfinite
from operator import index
from typing import Any, Iterator, Sequence

import torch

SERIALIZED_BLOCKWISE_FP8 = "serialized_blockwise"


def use_power_2_scales_default() -> bool:
    """Whether serialized rollout weights should use power-of-2 block scales.

    Hopper uses exact FP32 block scales by default. Blackwell requires
    power-of-2 (UE8M0-representable) block scales and must explicitly set
    ``NVTE_FP8_BLOCK_SCALING_FP32_SCALES=0``. The rollout weight sync must use
    the *same* scale format as the Megatron-TE training forward, otherwise the
    effective rollout and training weights differ by up to 2x per block.
    """

    scale_mode = os.getenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    if scale_mode not in {"0", "1"}:
        raise ValueError(
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES must be '0' (power-of-2) " f"or '1' (FP32 scales), got {scale_mode!r}"
        )
    return scale_mode == "0"


def use_amax_epsilon_default() -> float:
    """Read the TE blockwise amax floor used by the training weight quantizer."""

    raw_value = os.getenv("NVTE_FP8_BLOCK_AMAX_EPSILON", "0")
    try:
        epsilon = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"NVTE_FP8_BLOCK_AMAX_EPSILON must be a float, got {raw_value!r}") from exc
    if not isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"NVTE_FP8_BLOCK_AMAX_EPSILON must be finite and non-negative, got {epsilon}")
    return epsilon


_QWEN35_FP8_WEIGHT_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
    ".linear_attn.in_proj_qkv.weight",
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.out_proj.weight",
    # Qwen3.5-MoE dense-shaped shared expert (vLLM builds it WITH quant_config, so
    # FP8; the router ``mlp.gate`` and ``shared_expert_gate`` stay BF16 by omission).
    ".mlp.shared_expert.gate_proj.weight",
    ".mlp.shared_expert.up_proj.weight",
    ".mlp.shared_expert.down_proj.weight",
)
# Qwen3.5-MoE routed experts. Megatron-Bridge exports them as batched 3D tensors
# ``mlp.experts.gate_up_proj`` [E, 2*moe_inter, hidden] and ``mlp.experts.down_proj``
# [E, hidden, moe_inter]. vLLM's proven blockwise-FP8 MoE path is per-expert
# (``experts.N.<proj>.weight`` + ``.weight_scale_inv`` -> fused ``w13/w2_weight_scale_inv``
# via make_expert_params_mapping), so we un-batch + un-fuse into per-expert 2D linears.
_QWEN35_MOE_GATE_UP_SUFFIX = ".mlp.experts.gate_up_proj"
_QWEN35_MOE_DOWN_SUFFIX = ".mlp.experts.down_proj"
_QWEN35_UNQUANTIZED_LINEAR_SUFFIXES = (
    ".in_proj_b",
    ".in_proj_a",
)
_QWEN35_LINEAR_ATTN_PREFIX_TEMPLATES = (
    "{model_prefix}.layers.{layer_idx}.linear_attn",
    "{model_prefix}.language_model.layers.{layer_idx}.linear_attn",
)
_QWEN35_VISION_ATTN_PROJ_PREFIX_TEMPLATES = ("{model_prefix}.visual.blocks.{block_idx}.attn.proj",)


def _normalize_block_size(block_size: Sequence[int]) -> tuple[int, int]:
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


@dataclass(frozen=True)
class SerializedFp8Config:
    """Configuration for serialized FP8 rollout weight sync."""

    weight_block_size: tuple[int, int] = (128, 128)
    power_2_scale: bool = field(default_factory=use_power_2_scales_default)
    amax_epsilon: float = field(default_factory=use_amax_epsilon_default)

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight_block_size", _normalize_block_size(self.weight_block_size))
        if type(self.power_2_scale) is not bool:
            raise ValueError(f"power_2_scale must be a bool, got {self.power_2_scale!r}")
        if not isfinite(self.amax_epsilon) or self.amax_epsilon < 0:
            raise ValueError(f"amax_epsilon must be finite and non-negative, got {self.amax_epsilon}")


def get_serialized_fp8_quantization_config(
    weight_block_size: Sequence[int] = (128, 128),
    ignored_layers: Sequence[str] | None = None,
) -> dict:
    """Return the HF quantization_config needed for vLLM serialized FP8."""

    block_m, block_n = _normalize_block_size(weight_block_size)
    qconfig = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [block_m, block_n],
    }
    if ignored_layers:
        qconfig["ignored_layers"] = list(ignored_layers)
    return qconfig


def is_qwen35_config(hf_config: Any) -> bool:
    """Return whether an HF config uses the supported Qwen3.5 text layout."""

    text_config = getattr(hf_config, "text_config", None) or getattr(hf_config, "language_config", None) or hf_config
    model_type = str(getattr(text_config, "model_type", "") or getattr(hf_config, "model_type", ""))
    return model_type in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}


def get_qwen35_fp8_ignored_layers(hf_config: Any, model_prefix: str = "model") -> list[str]:
    """Return Qwen3.5 vLLM module prefixes excluded from serialized FP8.

    Serialized FP8 sync quantizes a narrow Megatron-name allow-list. The GDN
    ``in_proj_b``/``in_proj_a`` projection is intentionally absent from that
    allow-list, while vLLM would otherwise quantize the fused ``in_proj_ba``
    module under a global FP8 config. vLLM's skip logic checks full module
    prefixes and requires every shard of a fused module to have the same scheme,
    so emit both shard prefixes for each linear-attention layer.

    The Qwen3.5 text model and full conditional-generation wrapper use
    different checkpoint prefix families. vLLM applies its HF-to-vLLM mapper
    to these ignored-layer names before matching runtime modules.
    """

    text_config = getattr(hf_config, "text_config", None) or getattr(hf_config, "language_config", None) or hf_config
    if not is_qwen35_config(hf_config):
        return []

    layer_types = list(getattr(text_config, "layer_types", []) or [])
    ignored: list[str] = []
    for layer_idx, layer_type in enumerate(layer_types):
        if layer_type != "linear_attention":
            continue
        layer_prefixes = []
        for template in _QWEN35_LINEAR_ATTN_PREFIX_TEMPLATES:
            prefix = template.format(model_prefix=model_prefix, layer_idx=layer_idx)
            if prefix not in layer_prefixes:
                layer_prefixes.append(prefix)

        for layer_prefix in layer_prefixes:
            for suffix in _QWEN35_UNQUANTIZED_LINEAR_SUFFIXES:
                ignored.append(f"{layer_prefix}{suffix}")

    # Qwen3.5 text-only RL runs set language_model_only=true, but vLLM 0.23 can
    # still instantiate the unused vision tower before the multimodal limits take
    # effect. Its row-parallel attention output projection is 1152 wide and fails
    # the 128-wide FP8 block divisibility check under TP2, so keep it BF16. vLLM's
    # FP8 ignore matcher is exact-match by default.
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


def should_use_serialized_fp8(mode: str | None) -> bool:
    return mode == SERIALIZED_BLOCKWISE_FP8


def is_quantizable_weight(name: str, tensor: torch.Tensor) -> bool:
    """Return whether an exported HF tensor should be serialized as FP8.

    vLLM's FP8 config applies to Linear modules. HF checkpoints also contain 2D
    embedding/output weights, so keep known non-Linear weight tables unquantized.
    """

    if not name.endswith(".weight") or tensor.ndim != 2:
        return False

    return is_quantizable_weight_shape(name, tensor.shape)


def is_quantizable_weight_shape(name: str, shape: Sequence[int]) -> bool:
    if not name.endswith(".weight") or len(shape) != 2:
        return False
    return name.endswith(_QWEN35_FP8_WEIGHT_SUFFIXES)


def scale_name_for_weight(name: str) -> str:
    if not name.endswith(".weight"):
        raise ValueError(f"FP8 scale can only be derived from .weight tensors: {name}")
    return name[: -len(".weight")] + ".weight_scale_inv"


def blockwise_cast_to_fp8(
    weight: torch.Tensor,
    block_size: Sequence[int],
    power_2_scale: bool = False,
    amax_epsilon: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to FP8 with FP32 block scales.

    The returned scale follows the vLLM checkpoint ``weight_scale_inv``
    convention for blockwise FP8:

    ``dequantized_weight ~= qweight.float() * scale``.

    Exact FP32 scales are the Hopper default and require Megatron-TE to use
    ``NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1``. When ``power_2_scale`` is True,
    the dequant scale is rounded up with the same
    ``2**ceil(log2(amax/fp8_max))`` rule used by UE8M0 kernels; this is the
    required Blackwell mode. Mixing the scale modes between rollout and
    training makes their effective weights disagree by up to 2x per block.
    """

    if weight.ndim != 2:
        raise ValueError(f"Blockwise FP8 expects a 2D tensor, got shape={tuple(weight.shape)}")
    if not isfinite(amax_epsilon) or amax_epsilon < 0:
        raise ValueError(f"amax_epsilon must be finite and non-negative, got {amax_epsilon}")

    block_m, block_n = _normalize_block_size(block_size)
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
    # Keep the serialized rollout quantizer on the same amax floor as TE's
    # training weight quantizer. The tiny fallback only handles an all-zero
    # block when TE's optional floor is disabled.
    scale = blocks.abs().amax(dim=(2, 3)).clamp(min=max(amax_epsilon, 1e-10)) / fp8_info.max
    if power_2_scale:
        # Round the dequant scale up to the next power of two so the rollout
        # engine and the Megatron-TE training forward quantize weights against
        # identical block scales. Rounding up keeps amax / scale <= fp8_max.
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    q_blocks = (blocks / scale[:, :, None, None]).clamp(min=fp8_info.min, max=fp8_info.max)
    q_blocks = q_blocks.to(torch.float8_e4m3fn)
    q_padded = q_blocks.permute(0, 2, 1, 3).contiguous().view(padded_rows, padded_cols)
    q_weight = q_padded[:rows, :cols].contiguous()
    return q_weight, scale.to(torch.float32).contiguous()


def batched_moe_expert_spec(name: str) -> tuple[str, tuple[str, ...], bool] | None:
    """Recognize a batched Qwen3.5-MoE expert tensor exported by Megatron-Bridge.

    Returns ``(experts_base, proj_names, is_gate_up)`` where ``experts_base`` is the
    ``...mlp.experts`` prefix and per-expert HF names are
    ``f"{experts_base}.{expert_idx}.{proj}.weight"``. ``gate_up_proj`` is split along
    the output dim into ``gate_proj`` (first half) and ``up_proj`` (second half),
    matching vLLM's ``stacked``/``expert`` mappings. Returns ``None`` for non-experts.
    """

    if name.endswith(_QWEN35_MOE_GATE_UP_SUFFIX):
        return name[: -len(".gate_up_proj")], ("gate_proj", "up_proj"), True
    if name.endswith(_QWEN35_MOE_DOWN_SUFFIX):
        return name[: -len(".down_proj")], ("down_proj",), False
    return None


def _per_expert_2d_slices(mat: torch.Tensor, is_gate_up: bool) -> list[torch.Tensor]:
    """Split one expert's 2D matrix into the per-projection 2D linears."""
    if is_gate_up:
        if mat.shape[0] % 2 != 0:
            raise ValueError("Batched MoE gate_up_proj output dimension must be even, " f"got shape={tuple(mat.shape)}")
        half = mat.shape[0] // 2
        return [mat[:half], mat[half:]]
    return [mat]


def iter_batched_moe_expert_fp8_tensors(
    name: str,
    tensor: torch.Tensor,
    config: SerializedFp8Config,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Un-batch a 3D expert tensor into per-expert FP8 blockwise weights + scales."""
    spec = batched_moe_expert_spec(name)
    if spec is None:
        raise ValueError(f"Not a batched MoE expert tensor: {name}")
    if tensor.ndim != 3:
        raise ValueError(f"Batched MoE expert tensor must be 3D, got shape={tuple(tensor.shape)}")
    experts_base, proj_names, is_gate_up = spec
    num_experts = tensor.shape[0]
    for expert_idx in range(num_experts):
        mat = tensor[expert_idx]
        for proj, sub in zip(proj_names, _per_expert_2d_slices(mat, is_gate_up)):
            weight_name = f"{experts_base}.{expert_idx}.{proj}.weight"
            q_weight, scale = blockwise_cast_to_fp8(
                sub.contiguous(),
                config.weight_block_size,
                config.power_2_scale,
                config.amax_epsilon,
            )
            yield weight_name, q_weight
            yield scale_name_for_weight(weight_name), scale


def iter_serialized_fp8_tensors(
    name: str,
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    config: SerializedFp8Config,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield one or more tensors for a single exported HF weight."""

    if batched_moe_expert_spec(name) is not None:
        yield from iter_batched_moe_expert_fp8_tensors(name, tensor, config)
        return

    if is_quantizable_weight(name, tensor):
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
