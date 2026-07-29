"""Built-in model layouts used by quantization strategies."""

from __future__ import annotations

from typing import Any

from .base import ExpertExportLayout, ExpertWeight, MoeExpertSpec, QuantizedModelLayout

_QWEN_QUANTIZABLE_WEIGHT_SUFFIXES = (
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
        split_dim=1,
        output_weights=(ExpertWeight("down_proj", "routed_expert_down"),),
    ),
)

QWEN3_LAYOUT = QuantizedModelLayout(
    name="qwen3_moe",
    model_types=frozenset({"qwen3_moe"}),
    quantizable_weight_suffixes=_QWEN_QUANTIZABLE_WEIGHT_SUFFIXES,
    moe_expert_specs=_ROUTED_EXPERT_SPECS,
    expert_export_layout=ExpertExportLayout.PACKED,
)

QWEN35_LAYOUT = QuantizedModelLayout(
    name="qwen3_5",
    model_types=frozenset({"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}),
    quantizable_weight_suffixes=_QWEN_QUANTIZABLE_WEIGHT_SUFFIXES,
    moe_expert_specs=_ROUTED_EXPERT_SPECS,
)

MODEL_QUANTIZATION_LAYOUTS = {
    model_type: layout for layout in (QWEN3_LAYOUT, QWEN35_LAYOUT) for model_type in layout.model_types
}


def get_hf_model_type(hf_config: Any) -> str:
    """Return the text-model type used for layout lookup."""

    text_config = getattr(hf_config, "text_config", None) or getattr(hf_config, "language_config", None) or hf_config
    return str(getattr(text_config, "model_type", "") or getattr(hf_config, "model_type", ""))


def get_quantized_model_layout(hf_config_or_model_type: Any) -> QuantizedModelLayout:
    """Resolve the quantized model layout for an HF config or model type."""

    model_type = (
        hf_config_or_model_type
        if isinstance(hf_config_or_model_type, str)
        else get_hf_model_type(hf_config_or_model_type)
    )
    try:
        return MODEL_QUANTIZATION_LAYOUTS[model_type]
    except KeyError as exc:
        raise ValueError(
            f"No quantized model layout for model_type={model_type!r}; supported: {sorted(MODEL_QUANTIZATION_LAYOUTS)}"
        ) from exc


def is_qwen35_config(hf_config: Any) -> bool:
    """Return whether an HF config uses a Qwen3.5 layout."""

    return get_hf_model_type(hf_config) in QWEN35_LAYOUT.model_types
