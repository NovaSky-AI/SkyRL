"""Convert Megatron-exported tensors to serialized vLLM FP8 formats."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from math import isfinite
from operator import index
from typing import Any, Iterator, Sequence

import torch

SERIALIZED_BLOCKWISE_FP8 = "serialized_blockwise"
SERIALIZED_MXFP8 = "serialized_mxfp8"
BLOCKWISE_128X128 = "blockwise_128x128"
MXFP8_1X32 = "mxfp8_1x32"
# Internal wire-format marker for Qwen3.5 MoE tensors that remain batched over
# experts. The receiver strips this marker and routes the tensor directly to
# vLLM's fused-MoE parameter loader instead of the ordinary HF-name loader.
SKYRL_BATCHED_MOE_FP8_PREFIX = "__skyrl_batched_moe_fp8__:"


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
    # Shared-expert linears use FP8; router and shared-expert gates remain BF16.
    ".mlp.shared_expert.gate_proj.weight",
    ".mlp.shared_expert.up_proj.weight",
    ".mlp.shared_expert.down_proj.weight",
)
_QWEN35_UNQUANTIZED_LINEAR_SUFFIXES = (
    ".in_proj_b",
    ".in_proj_a",
)
_QWEN35_LINEAR_ATTN_PREFIX_TEMPLATES = (
    "{model_prefix}.layers.{layer_idx}.linear_attn",
    "{model_prefix}.language_model.layers.{layer_idx}.linear_attn",
)
_QWEN35_VISION_ATTN_PROJ_PREFIX_TEMPLATES = ("{model_prefix}.visual.blocks.{block_idx}.attn.proj",)


@dataclass(frozen=True)
class MoeArchitectureSpec:
    """HF expert export and vLLM projection naming for one model family."""

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
_QWEN3_MOE_SPEC = MoeArchitectureSpec(
    gate_up_suffixes=(".gate_proj.weight", ".up_proj.weight"),
    down_suffix=".down_proj.weight",
    gate_up_fused=False,
    vllm_projection_names=("gate_proj", "up_proj", "down_proj"),
    batched=False,
)
MOE_ARCHITECTURE_SPECS = {
    "qwen3_moe": _QWEN3_MOE_SPEC,
    "qwen3_5_moe": _QWEN35_MOE_SPEC,
    "qwen3_5_moe_text": _QWEN35_MOE_SPEC,
}
_EXPERT_ONLY_MXFP8_IGNORED_MODULES = (
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
_PER_EXPERT_WEIGHT_PATTERN = re.compile(
    r"^(?P<base>.+\.mlp\.experts\.\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


def get_hf_model_type(hf_config: Any) -> str:
    """Return the text model type used for expert serialization dispatch."""

    text_config = getattr(hf_config, "text_config", None) or getattr(hf_config, "language_config", None) or hf_config
    return str(getattr(text_config, "model_type", "") or getattr(hf_config, "model_type", ""))


def get_moe_architecture_spec(model_type: str) -> MoeArchitectureSpec:
    """Return the registered MoE export layout."""

    try:
        return MOE_ARCHITECTURE_SPECS[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(MOE_ARCHITECTURE_SPECS))
        raise ValueError(
            f"Serialized MXFP8 does not support model_type={model_type!r}; supported: {supported}"
        ) from exc


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
        object.__setattr__(self, "weight_block_size", _normalize_block_size(self.weight_block_size))
        if type(self.power_2_scale) is not bool:
            raise ValueError(f"power_2_scale must be a bool, got {self.power_2_scale!r}")
        if not isfinite(self.amax_epsilon) or self.amax_epsilon < 0:
            raise ValueError(f"amax_epsilon must be finite and non-negative, got {self.amax_epsilon}")
        if self.expert_only:
            if not self.model_type:
                raise ValueError("expert_only serialized FP8 requires model_type")
            get_moe_architecture_spec(self.model_type)


def serialized_fp8_config_for_mode(mode: str, *, model_type: str | None = None) -> SerializedFp8Config:
    """Build the serializer configuration for a weight-sync mode."""

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


def get_serialized_fp8_quantization_config(
    weight_block_size: Sequence[int] = (128, 128),
    ignored_layers: Sequence[str] | None = None,
) -> dict:
    """Return vLLM's Hugging Face quantization config for serialized FP8."""

    block_m, block_n = _normalize_block_size(weight_block_size)
    qconfig = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [block_m, block_n],
    }
    if ignored_layers:
        qconfig["ignored_layers"] = list(ignored_layers)
    return qconfig


def get_serialized_mxfp8_quantization_config() -> dict:
    """Return vLLM's ModelOpt config for expert-only serialized MXFP8."""

    return {
        "quant_method": "modelopt",
        "quant_algo": "MXFP8",
        "ignore": list(_EXPERT_ONLY_MXFP8_IGNORED_MODULES),
    }


def is_qwen35_config(hf_config: Any) -> bool:
    """Return whether an HF config uses the supported Qwen3.5 text layout."""

    return get_hf_model_type(hf_config) in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}


def get_qwen35_fp8_ignored_layers(hf_config: Any, model_prefix: str = "model") -> list[str]:
    """Return Qwen3.5 vLLM module prefixes excluded from serialized FP8.

    Serialized sync excludes GDN ``in_proj_a`` and ``in_proj_b``. vLLM requires
    every shard of the fused module to share a quantization scheme, so both
    prefixes are ignored for text-only and conditional-generation checkpoints.
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

    # vLLM 0.23 may instantiate the vision tower for text-only runs. Its TP2
    # attention output is incompatible with 128-wide blocks, and ignore matching
    # requires each block's exact module prefix.
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
    return mode in (SERIALIZED_BLOCKWISE_FP8, SERIALIZED_MXFP8)


def is_quantizable_weight(name: str, tensor: torch.Tensor, *, expert_only: bool = False) -> bool:
    """Return whether an exported HF tensor should be serialized as FP8.

    vLLM's FP8 config applies to Linear modules. HF checkpoints also contain 2D
    embedding/output weights, so keep known non-Linear weight tables unquantized.
    """

    if expert_only or not name.endswith(".weight") or tensor.ndim != 2:
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


def mxfp8_scale_name_for_weight(name: str) -> str:
    if not name.endswith(".weight"):
        raise ValueError(f"MXFP8 scale can only be derived from .weight tensors: {name}")
    return name[: -len(".weight")] + ".weight_scale"


def blockwise_cast_to_fp8(
    weight: torch.Tensor,
    block_size: Sequence[int],
    power_2_scale: bool = False,
    amax_epsilon: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to vLLM's blockwise E4M3 checkpoint format.

    Returns ``weight_scale_inv`` such that
    ``weight ~= qweight.float() * scale``. Power-of-two mode rounds scales up
    to match Transformer Engine's UE8M0 rule.
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
    # Match TE's amax floor, with a nonzero fallback for all-zero blocks.
    scale = blocks.abs().amax(dim=(2, 3)).clamp(min=max(amax_epsilon, 1e-10)) / fp8_info.max
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
    amax_epsilon: float = 0.0,
    expert_batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 3D ``[experts, rows, cols]`` tensor blockwise.

    Quantizing several experts per operation avoids launching the full 2D
    conversion pipeline once per expert, while bounded batches limit peak FP32
    workspace.
    """

    if weight.ndim != 3:
        raise ValueError(f"Batched blockwise FP8 expects a 3D tensor, got shape={tuple(weight.shape)}")
    if not isfinite(amax_epsilon) or amax_epsilon < 0:
        raise ValueError(f"amax_epsilon must be finite and non-negative, got {amax_epsilon}")
    if isinstance(expert_batch_size, bool) or not isinstance(expert_batch_size, int) or expert_batch_size <= 0:
        raise ValueError(f"expert_batch_size must be a positive integer, got {expert_batch_size!r}")

    block_m, block_n = _normalize_block_size(block_size)
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
        scale = blocks.abs().amax(dim=(3, 4)).clamp(min=max(amax_epsilon, 1e-10)) / fp8_info.max
        if power_2_scale:
            scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
        q_blocks = (blocks / scale[:, :, :, None, None]).clamp(min=fp8_info.min, max=fp8_info.max)
        q_blocks = q_blocks.to(torch.float8_e4m3fn)
        q_padded = q_blocks.permute(0, 1, 3, 2, 4).contiguous().view(end - start, padded_rows, padded_cols)
        q_weight[start:end].copy_(q_padded[:, :rows, :cols])
        scales[start:end].copy_(scale)

    return q_weight, scales


def _power_2_scales_to_e8m0(scales: torch.Tensor) -> torch.Tensor:
    """Encode positive power-of-two FP32 scales as biased E8M0 exponents."""

    exponent_bits = (scales.contiguous().view(torch.int32) >> 23) & 0xFF
    return exponent_bits.to(torch.uint8)


def mxfp8_cast_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to vLLM's row-major MXFP8 checkpoint format."""

    if weight.ndim != 2:
        raise ValueError(f"MXFP8 expects a 2D tensor, got shape={tuple(weight.shape)}")
    if weight.shape[-1] % 32 != 0:
        raise ValueError(f"MXFP8 requires the last dimension to be divisible by 32, got shape={tuple(weight.shape)}")
    q_weight, scales = blockwise_cast_to_fp8(weight, (1, 32), power_2_scale=True)
    return q_weight, _power_2_scales_to_e8m0(scales)


def batched_mxfp8_cast_to_fp8(
    weight: torch.Tensor,
    expert_batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 3D expert tensor to row-major MXFP8 weights and E8M0 scales."""

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
    return q_weight, _power_2_scales_to_e8m0(scales)


def batched_moe_expert_spec(
    name: str,
    model_type: str | None = None,
) -> tuple[str, tuple[str, ...], bool] | None:
    """Parse a registered Megatron Bridge batched MoE tensor name.

    Returns ``(experts_base, projection_names, split_gate_up)`` or ``None``.
    """

    spec = _QWEN35_MOE_SPEC if model_type is None else get_moe_architecture_spec(model_type)
    if not spec.batched or ".mlp.experts" not in name:
        return None
    gate_up_suffix = spec.gate_up_suffixes[0]
    if name.endswith(gate_up_suffix):
        return name[: -len(gate_up_suffix)], spec.vllm_projection_names[:2], spec.gate_up_fused
    if name.endswith(spec.down_suffix):
        return name[: -len(spec.down_suffix)], (spec.vllm_projection_names[-1],), False
    return None


def is_per_expert_moe_weight(name: str, model_type: str) -> bool:
    """Return whether a tensor is a registered per-expert HF projection."""

    spec = get_moe_architecture_spec(model_type)
    if spec.batched:
        return False
    match = _PER_EXPERT_WEIGHT_PATTERN.fullmatch(name)
    suffixes = (*spec.gate_up_suffixes, spec.down_suffix)
    return (
        match is not None
        and match.group("projection") in spec.vllm_projection_names
        and name.endswith(suffixes)
    )


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
    spec = batched_moe_expert_spec(name, config.model_type)
    if spec is None:
        raise ValueError(f"Not a batched MoE expert tensor: {name}")
    if tensor.ndim != 3:
        raise ValueError(f"Batched MoE expert tensor must be 3D, got shape={tuple(tensor.shape)}")
    experts_base, proj_names, is_gate_up = spec
    if is_gate_up:
        if tensor.shape[1] % 2 != 0:
            raise ValueError(f"Batched MoE gate_up_proj output dimension must be even, got shape={tuple(tensor.shape)}")
        half = tensor.shape[1] // 2
        projection_tensors = (tensor[:, :half], tensor[:, half:])
    else:
        projection_tensors = (tensor,)

    for proj, projection_tensor in zip(proj_names, projection_tensors):
        if config.scaling_mode == MXFP8_1X32:
            q_weight, scale = batched_mxfp8_cast_to_fp8(projection_tensor)
        else:
            q_weight, scale = batched_blockwise_cast_to_fp8(
                projection_tensor,
                config.weight_block_size,
                config.power_2_scale,
                config.amax_epsilon,
            )
        weight_name = f"{SKYRL_BATCHED_MOE_FP8_PREFIX}{experts_base}.{proj}.weight"
        yield weight_name, q_weight
        scale_name = (
            mxfp8_scale_name_for_weight(weight_name)
            if config.scaling_mode == MXFP8_1X32
            else scale_name_for_weight(weight_name)
        )
        yield scale_name, scale


def iter_serialized_fp8_tensors(
    name: str,
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    config: SerializedFp8Config,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield vLLM checkpoint tensors for one Megatron-exported weight."""

    if batched_moe_expert_spec(name, config.model_type) is not None:
        yield from iter_batched_moe_expert_fp8_tensors(name, tensor, config)
        return

    if config.expert_only and config.model_type and is_per_expert_moe_weight(name, config.model_type):
        q_weight, scale = mxfp8_cast_to_fp8(tensor)
        yield name, q_weight
        yield mxfp8_scale_name_for_weight(name), scale
        return

    if config.expert_only and ".mlp.experts." in name and tensor.ndim >= 2:
        raise ValueError(f"Unsupported routed-expert export tensor for model_type={config.model_type!r}: {name}")

    if is_quantizable_weight(name, tensor, expert_only=config.expert_only):
        if config.scaling_mode == MXFP8_1X32:
            q_weight, scale = mxfp8_cast_to_fp8(tensor)
            scale_name = mxfp8_scale_name_for_weight(name)
        else:
            q_weight, scale = blockwise_cast_to_fp8(
                tensor,
                config.weight_block_size,
                config.power_2_scale,
                config.amax_epsilon,
            )
            scale_name = scale_name_for_weight(name)
        yield name, q_weight
        yield scale_name, scale
        return

    yield name, tensor.to(dtype=target_dtype)
