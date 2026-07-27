"""Model-specific serialized quantization layouts."""

from __future__ import annotations

from typing import Any

from .base import (
    ExpertExportLayout,
    ModelQuantizationSpec,
    MoeExpertSpec,
    MoeProjection,
)

_QWEN35_QUANTIZABLE_WEIGHT_SUFFIXES = (
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
_QWEN35_UNQUANTIZED_LINEAR_SUFFIXES = (".in_proj_b", ".in_proj_a")
_QWEN35_LINEAR_ATTN_PREFIX_TEMPLATES = (
    "{model_prefix}.layers.{layer_idx}.linear_attn",
    "{model_prefix}.language_model.layers.{layer_idx}.linear_attn",
)
_QWEN35_VISION_ATTN_PROJ_PREFIX_TEMPLATES = ("{model_prefix}.visual.blocks.{block_idx}.attn.proj",)

_ROUTED_EXPERT_SPECS = (
    MoeExpertSpec(
        source_suffix=".gate_up_proj",
        split_dim=1,
        projections=(
            MoeProjection(
                hf_name="gate_proj",
                vllm_param=".experts.w13_weight",
                shard_id="w1",
            ),
            MoeProjection(
                hf_name="up_proj",
                vllm_param=".experts.w13_weight",
                shard_id="w3",
            ),
        ),
    ),
    MoeExpertSpec(
        source_suffix=".down_proj",
        split_dim=1,
        projections=(
            MoeProjection(
                hf_name="down_proj",
                vllm_param=".experts.w2_weight",
                shard_id="w2",
            ),
        ),
    ),
)


def get_hf_model_type(hf_config: Any) -> str:
    """Return the text model type used for serialization dispatch."""

    text_config = getattr(hf_config, "text_config", None) or getattr(hf_config, "language_config", None) or hf_config
    return str(getattr(text_config, "model_type", "") or getattr(hf_config, "model_type", ""))


def _no_ignored_layers(_hf_config: Any) -> list[str]:
    return []


def _qwen35_ignored_layers(hf_config: Any, model_prefix: str = "model") -> list[str]:
    text_config = getattr(hf_config, "text_config", None) or getattr(hf_config, "language_config", None) or hf_config
    layer_types = list(getattr(text_config, "layer_types", []) or [])
    ignored: list[str] = []
    for layer_idx, layer_type in enumerate(layer_types):
        if layer_type != "linear_attention":
            continue
        for template in _QWEN35_LINEAR_ATTN_PREFIX_TEMPLATES:
            layer_prefix = template.format(model_prefix=model_prefix, layer_idx=layer_idx)
            for suffix in _QWEN35_UNQUANTIZED_LINEAR_SUFFIXES:
                ignored.append(f"{layer_prefix}{suffix}")

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


QWEN3_QUANTIZATION_SPEC = ModelQuantizationSpec(
    name="qwen3_moe",
    model_types=frozenset({"qwen3_moe"}),
    quantizable_weight_suffixes=_QWEN35_QUANTIZABLE_WEIGHT_SUFFIXES,
    moe_expert_specs=_ROUTED_EXPERT_SPECS,
    ignored_layers_fn=_no_ignored_layers,
    expert_export_layout=ExpertExportLayout.PACKED,
)

QWEN35_QUANTIZATION_SPEC = ModelQuantizationSpec(
    name="qwen3_5",
    model_types=frozenset({"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}),
    quantizable_weight_suffixes=_QWEN35_QUANTIZABLE_WEIGHT_SUFFIXES,
    moe_expert_specs=_ROUTED_EXPERT_SPECS,
    ignored_layers_fn=_qwen35_ignored_layers,
)

MODEL_QUANTIZATION_SPECS: list[ModelQuantizationSpec] = []


def register_model_quantization_spec(spec: ModelQuantizationSpec) -> None:
    if any(registered.name == spec.name for registered in MODEL_QUANTIZATION_SPECS):
        raise ValueError(f"Duplicate model quantization spec: {spec.name}")
    MODEL_QUANTIZATION_SPECS.append(spec)


def model_quantization_specs() -> tuple[ModelQuantizationSpec, ...]:
    return tuple(MODEL_QUANTIZATION_SPECS)


def resolve_model_quantization_spec(hf_config_or_model_type: Any) -> ModelQuantizationSpec | None:
    model_type = (
        hf_config_or_model_type
        if isinstance(hf_config_or_model_type, str)
        else get_hf_model_type(hf_config_or_model_type)
    )
    matches = [spec for spec in MODEL_QUANTIZATION_SPECS if spec.matches(model_type)]
    if len(matches) > 1:
        raise ValueError(f"Multiple model quantization specs matched model_type={model_type!r}")
    return matches[0] if matches else None


def get_model_quantization_spec(hf_config_or_model_type: Any) -> ModelQuantizationSpec:
    model_type = (
        hf_config_or_model_type
        if isinstance(hf_config_or_model_type, str)
        else get_hf_model_type(hf_config_or_model_type)
    )
    spec = resolve_model_quantization_spec(model_type)
    if spec is None:
        supported = sorted(model_type for item in MODEL_QUANTIZATION_SPECS for model_type in item.model_types)
        raise ValueError(f"No serialized quantization spec for model_type={model_type!r}; supported: {supported}")
    return spec


def is_qwen35_config(hf_config: Any) -> bool:
    return QWEN35_QUANTIZATION_SPEC.matches(get_hf_model_type(hf_config))


def get_qwen35_fp8_ignored_layers(hf_config: Any, model_prefix: str = "model") -> list[str]:
    if not is_qwen35_config(hf_config):
        return []
    return _qwen35_ignored_layers(hf_config, model_prefix=model_prefix)


register_model_quantization_spec(QWEN3_QUANTIZATION_SPEC)
register_model_quantization_spec(QWEN35_QUANTIZATION_SPEC)
