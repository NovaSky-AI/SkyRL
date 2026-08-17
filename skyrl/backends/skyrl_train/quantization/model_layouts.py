"""Built-in model layouts shared by quantization strategies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import (
    ExpertExportLayout,
    ExpertWeight,
    ModelQuantizationLayout,
    MoeExpertSpec,
    QuantizationTarget,
)

_QWEN_LINEAR_SUFFIXES = (
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
    ".mlp.shared_expert.gate_proj.weight",
    ".mlp.shared_expert.up_proj.weight",
    ".mlp.shared_expert.down_proj.weight",
)
_ROUTED_EXPERT_SPECS = (
    MoeExpertSpec(
        source_suffix=".gate_up_proj",
        split_dim=1,
        output_weights=(
            ExpertWeight("gate_proj", "routed_expert_gate"),
            ExpertWeight("up_proj", "routed_expert_up"),
        ),
    ),
    MoeExpertSpec(
        source_suffix=".down_proj",
        split_dim=None,
        output_weights=(ExpertWeight("down_proj", "routed_expert_down"),),
    ),
)
_QWEN35_UNQUANTIZED_LINEAR_SUFFIXES = (".in_proj_b", ".in_proj_a")
_QWEN35_LINEAR_ATTN_PREFIX_TEMPLATES = (
    "{model_prefix}.layers.{layer_idx}.linear_attn",
    "{model_prefix}.language_model.layers.{layer_idx}.linear_attn",
)
_QWEN35_VISION_BLOCK_PREFIX_TEMPLATES = (
    "{model_prefix}.visual.blocks.{block_idx}.attn.proj",
    "{model_prefix}.visual.blocks.{block_idx}.mlp.linear_fc1",
    "{model_prefix}.visual.blocks.{block_idx}.mlp.linear_fc2",
)


def _matches_suffixes(*suffixes: str, dimensions: tuple[int, ...] = (2, 3)):
    def matches(name: str, shape: Sequence[int]) -> bool:
        return len(shape) in dimensions and name.endswith(suffixes)

    return matches


_QWEN_TARGETS = (
    QuantizationTarget(
        name="linear",
        matches_exported_weight=_matches_suffixes(*_QWEN_LINEAR_SUFFIXES, dimensions=(2,)),
    ),
    QuantizationTarget(
        name="routed_expert_gate",
        matches_exported_weight=_matches_suffixes(".mlp.experts.gate_proj.weight"),
        megatron_patterns=("*.experts.linear_fc1",),
        checkpoint_component="gate_proj",
        vllm_parameter="w13_weight",
        vllm_shard_id="w1",
    ),
    QuantizationTarget(
        name="routed_expert_up",
        matches_exported_weight=_matches_suffixes(".mlp.experts.up_proj.weight"),
        megatron_patterns=("*.experts.linear_fc1",),
        checkpoint_component="up_proj",
        vllm_parameter="w13_weight",
        vllm_shard_id="w3",
    ),
    QuantizationTarget(
        name="routed_expert_down",
        matches_exported_weight=_matches_suffixes(".mlp.experts.down_proj.weight"),
        megatron_patterns=("*.experts.linear_fc2",),
        checkpoint_component="down_proj",
        vllm_parameter="w2_weight",
        vllm_shard_id="w2",
    ),
)


def get_hf_model_type(hf_config: Any) -> str:
    """Return the text-model type used for layout lookup."""

    text_config = getattr(hf_config, "text_config", None) or getattr(hf_config, "language_config", None) or hf_config
    return str(getattr(text_config, "model_type", "") or getattr(hf_config, "model_type", ""))


def get_qwen35_ignored_layers(hf_config: Any, model_prefix: str = "model") -> list[str]:
    """Return Qwen3.5 module prefixes incompatible with blockwise loading."""

    if get_hf_model_type(hf_config) not in QWEN35_LAYOUT.model_types:
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
        for template in _QWEN35_VISION_BLOCK_PREFIX_TEMPLATES:
            ignored.append(template.format(model_prefix=model_prefix, block_idx=block_idx))
    return ignored


QWEN3_LAYOUT = ModelQuantizationLayout(
    name="qwen3_moe",
    model_types=frozenset({"qwen3_moe"}),
    targets=_QWEN_TARGETS,
    moe_expert_specs=_ROUTED_EXPERT_SPECS,
    expert_export_layout=ExpertExportLayout.PACKED,
)

QWEN35_LAYOUT = ModelQuantizationLayout(
    name="qwen3_5",
    model_types=frozenset({"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}),
    targets=_QWEN_TARGETS,
    moe_expert_specs=_ROUTED_EXPERT_SPECS,
    ignored_layers=get_qwen35_ignored_layers,
)

MODEL_QUANTIZATION_LAYOUTS = {
    model_type: layout for layout in (QWEN3_LAYOUT, QWEN35_LAYOUT) for model_type in layout.model_types
}


def get_quantized_model_layout(hf_config_or_model_type: Any) -> ModelQuantizationLayout:
    """Resolve a model layout from a Hugging Face config or model type."""

    model_type = (
        hf_config_or_model_type
        if isinstance(hf_config_or_model_type, str)
        else get_hf_model_type(hf_config_or_model_type)
    )
    try:
        return MODEL_QUANTIZATION_LAYOUTS[model_type]
    except KeyError as exc:
        raise ValueError(
            f"No quantized model layout for model_type={model_type!r}; "
            f"supported: {sorted(MODEL_QUANTIZATION_LAYOUTS)}"
        ) from exc
