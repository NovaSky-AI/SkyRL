"""Qwen3.5 ``ModelFp8Spec`` for serialized FP8 weight sync (blockwise and MXFP8)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import (
    BLOCKWISE_FP8,
    MXFP8,
    ModelFp8Spec,
    MoeExpertSpec,
    MoeProjection,
    register_fp8_spec,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.quantize import MXFP8_GROUP_SIZE

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
# Megatron Bridge exports routed experts in batched tensors. Keep the expert
# dimension intact on the wire so the receiver can use vLLM's fused MoE loader.
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
# Vision attention output and both vision MLP linears carry dims (e.g. 4304)
# that stop being 128-divisible once vLLM TP-shards them, so all three must be
# ignored for the engine to build at TP>1. The weight-sync spec keeps the
# vision tower BF16 regardless.
_QWEN35_VISION_BLOCK_PREFIX_TEMPLATES = (
    "{model_prefix}.visual.blocks.{block_idx}.attn.proj",
    "{model_prefix}.visual.blocks.{block_idx}.mlp.linear_fc1",
    "{model_prefix}.visual.blocks.{block_idx}.mlp.linear_fc2",
)
# MXFP8 rejects the rest of the vision tower as well: its kernels require the
# reduction dim to be a multiple of 32, and the vision intermediate size (4304)
# leaves a remainder of 16. The tower is not evaluated on a text-only RL
# rollout, so excluding all of it costs nothing measurable.
_QWEN35_MXFP8_EXTRA_VISION_BLOCK_TEMPLATES = ("{model_prefix}.visual.blocks.{block_idx}.attn.qkv",)
_QWEN35_MXFP8_VISION_MERGER_TEMPLATES = (
    "{model_prefix}.visual.merger.linear_fc1",
    "{model_prefix}.visual.merger.linear_fc2",
)

_MOE_GATE = MoeProjection(hf_name="gate_proj", vllm_param="w13_weight", shard_id="w1")
_MOE_UP = MoeProjection(hf_name="up_proj", vllm_param="w13_weight", shard_id="w3")
_MOE_DOWN = MoeProjection(hf_name="down_proj", vllm_param="w2_weight", shard_id="w2")


def is_qwen35_config(hf_config: Any) -> bool:
    """Return whether an HF config uses the supported Qwen3.5 text layout."""

    text_config = getattr(hf_config, "text_config", None) or getattr(hf_config, "language_config", None) or hf_config
    model_type = str(getattr(text_config, "model_type", "") or getattr(hf_config, "model_type", ""))
    return model_type in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}


def get_qwen35_fp8_ignored_layers(
    hf_config: Any,
    wire_format: str = BLOCKWISE_FP8,
    model_prefix: str = "model",
) -> list[str]:
    """Return Qwen3.5 vLLM module prefixes excluded from serialized FP8.

    Both wire formats exclude GDN ``in_proj_a`` and ``in_proj_b`` — blockwise
    because the 32-row shard is not 128-divisible, MXFP8 because its kernels
    require ``out_features >= 128``. vLLM requires every shard of a fused module
    to share a quantization scheme, so both prefixes are ignored for text-only
    and conditional-generation checkpoints.

    MXFP8 additionally excludes the whole vision tower; see the template
    definitions above. The blockwise list is a strict subset of the MXFP8 one.
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

    # vLLM instantiates the vision tower even for text-only runs
    # (language_model_only only affects multimodal weight loading), and ignore
    # matching requires each block's exact module prefix.
    vision_config = getattr(hf_config, "vision_config", None) or getattr(hf_config, "visual_config", None)
    vision_depth = 0
    if vision_config is not None:
        for attr in ("depth", "num_hidden_layers", "num_layers"):
            value = getattr(vision_config, attr, None)
            if isinstance(value, int) and value > 0:
                vision_depth = value
                break
    block_templates = _QWEN35_VISION_BLOCK_PREFIX_TEMPLATES
    if wire_format == MXFP8:
        block_templates = block_templates + _QWEN35_MXFP8_EXTRA_VISION_BLOCK_TEMPLATES
    for block_idx in range(vision_depth):
        for template in block_templates:
            ignored.append(template.format(model_prefix=model_prefix, block_idx=block_idx))
    if wire_format == MXFP8 and vision_depth:
        for template in _QWEN35_MXFP8_VISION_MERGER_TEMPLATES:
            ignored.append(template.format(model_prefix=model_prefix))
    return ignored


def is_quantizable_weight_shape(name: str, shape: Sequence[int], wire_format: str = BLOCKWISE_FP8) -> bool:
    """Return whether an exported HF weight should be serialized as FP8.

    vLLM's FP8 config applies to Linear modules. HF checkpoints also contain 2D
    embedding/output weights, so keep known non-Linear weight tables unquantized.

    MXFP8 additionally requires the reduction dimension to be a multiple of 32;
    a weight that fails it has no valid group layout on the wire.
    """

    if not name.endswith(".weight") or len(shape) != 2:
        return False
    if not name.endswith(_QWEN35_FP8_WEIGHT_SUFFIXES):
        return False
    if wire_format == MXFP8 and shape[1] % MXFP8_GROUP_SIZE != 0:
        return False
    return True


def batched_moe_expert_spec(name: str) -> Optional[MoeExpertSpec]:
    """Map a Megatron Bridge batched Qwen3.5 MoE tensor name onto projections."""

    if name.endswith(_QWEN35_MOE_GATE_UP_SUFFIX):
        return MoeExpertSpec(
            experts_base=name[: -len(".gate_up_proj")],
            projections=(_MOE_GATE, _MOE_UP),
            split_dim=1,
        )
    if name.endswith(_QWEN35_MOE_DOWN_SUFFIX):
        return MoeExpertSpec(experts_base=name[: -len(".down_proj")], projections=(_MOE_DOWN,))
    return None


QWEN35_FP8_SPEC = register_fp8_spec(
    ModelFp8Spec(
        name="qwen3.5",
        matches=is_qwen35_config,
        should_quantize=is_quantizable_weight_shape,
        ignored_layers=get_qwen35_fp8_ignored_layers,
        moe_expert_spec=batched_moe_expert_spec,
        moe_projections=(_MOE_GATE, _MOE_UP, _MOE_DOWN),
    )
)
