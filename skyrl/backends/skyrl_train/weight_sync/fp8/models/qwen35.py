"""Qwen3.5 ``ModelFp8Spec`` for serialized FP8 weight sync (blockwise and MXFP8)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from skyrl.backends.skyrl_train.distributed.megatron.quantization_utils import (
    resolve_text_config,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import (
    BLOCKWISE_FP8,
    MXFP8,
    ModelFp8Spec,
    MoeExpertSpec,
    MoeProjection,
    register_fp8_spec,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.quantize import MXFP8_GROUP_SIZE

# Linear modules synced as FP8, by the part of a decoder layer that holds them.
_QWEN35_FP8_ATTENTION_MODULES = {
    "full_attention": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"),
    # in_proj_b / in_proj_a stay BF16; see the base exclude list below.
    "linear_attention": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"),
}
_QWEN35_FP8_DENSE_MLP_MODULES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
# Shared-expert linears use FP8; router and shared-expert gates remain BF16.
_QWEN35_FP8_SHARED_EXPERT_MODULES = (
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
)
_QWEN35_FP8_WEIGHT_SUFFIXES = tuple(
    f".{module}.weight"
    for module in (
        *_QWEN35_FP8_ATTENTION_MODULES["full_attention"],
        *_QWEN35_FP8_ATTENTION_MODULES["linear_attention"],
        *_QWEN35_FP8_DENSE_MLP_MODULES,
        *_QWEN35_FP8_SHARED_EXPERT_MODULES,
    )
)
# Megatron Bridge exports routed experts in batched tensors. Keep the expert
# dimension intact on the wire so the receiver can use vLLM's fused MoE loader.
_QWEN35_MOE_EXPERTS_MODULE = "mlp.experts"
_QWEN35_MOE_GATE_UP_SUFFIX = f".{_QWEN35_MOE_EXPERTS_MODULE}.gate_up_proj"
_QWEN35_MOE_DOWN_SUFFIX = f".{_QWEN35_MOE_EXPERTS_MODULE}.down_proj"
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

    text_config = resolve_text_config(hf_config)
    model_type = str(getattr(text_config, "model_type", "") or getattr(hf_config, "model_type", ""))
    return model_type in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}


def get_qwen35_base_exclude_list(
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

    text_config = resolve_text_config(hf_config)
    if not is_qwen35_config(hf_config):
        return []

    layer_types = list(getattr(text_config, "layer_types", []) or [])
    ignored: list[str] = []
    for layer_idx, layer_type in enumerate(layer_types):
        if layer_type == "linear_attention":
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


def get_qwen35_fp8_modules(hf_config: Any) -> list[str]:
    """Return the HF module names Qwen3.5 syncs as FP8, layer by layer.

    Names follow Megatron-Bridge's export: unified VL checkpoints nest the
    language model under ``model.language_model``, text-only checkpoints keep
    it under ``model``. A MoE layer's routed experts are one ``mlp.experts``
    entry, matching the batched tensors the bridge exports for them.
    """

    if not is_qwen35_config(hf_config):
        return []
    text_config = resolve_text_config(hf_config)
    prefix = "model" if text_config is hf_config else "model.language_model"
    if "moe" in str(getattr(text_config, "model_type", "")):
        mlp_modules = (_QWEN35_MOE_EXPERTS_MODULE,)
        # vLLM builds the shared expert only when it has a width.
        if getattr(text_config, "shared_expert_intermediate_size", 0) > 0:
            mlp_modules += _QWEN35_FP8_SHARED_EXPERT_MODULES
    else:
        mlp_modules = _QWEN35_FP8_DENSE_MLP_MODULES

    modules = []
    for layer_idx, layer_type in enumerate(getattr(text_config, "layer_types", None) or []):
        for module in (*_QWEN35_FP8_ATTENTION_MODULES[layer_type], *mlp_modules):
            modules.append(f"{prefix}.layers.{layer_idx}.{module}")
    return modules


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
        base_exclude_list=get_qwen35_base_exclude_list,
        fp8_modules=get_qwen35_fp8_modules,
        moe_expert_spec=batched_moe_expert_spec,
        moe_projections=(_MOE_GATE, _MOE_UP, _MOE_DOWN),
    )
)
