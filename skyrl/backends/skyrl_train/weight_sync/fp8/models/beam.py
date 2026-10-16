"""Beam (``model_type="beam"``) ``ModelFp8Spec`` for serialized FP8 weight sync (blockwise and MXFP8).

The quantize / keep-BF16 split follows the released FP8 checkpoint's compressed-tensors config (its scheme is
per-channel FP8_DYNAMIC, but which layers are FP8 is the same question): every attention and MLP ``Linear`` is
FP8 -- ``q/k/v/o_proj``, the attention output gate ``gate_proj``, the dense MLP's ``gate/up/down_proj`` (the first
dense layers) and every routed expert's ``gate/up/down_proj``. BF16/FP32 stays for the embeddings, ``lm_head``,
all norms, the attention ``gate_bias`` and the router (``mlp.gate`` and its ``e_score_correction_bias``; FP32).
The checkpoint ignores ``lm_head``, ``model.embed_tokens`` and ``re:.*mlp\\.gate$``.

On the vLLM side (vLLM's Beam model), HF ``self_attn.gate_proj`` loads into ``output_gate_proj``, fused either into
``qkv_gate_proj`` (with ``q/k/v``) or kept beside ``qkv_proj``; every shard of a fused module is FP8 here, as vLLM
requires. The router is a ``GateLinear`` built without a quantization config, so it is never quantized; the
ignore pattern is kept anyway to mirror the released checkpoint. Embeddings, ``lm_head`` and norms are not
``LinearBase`` modules, so the FP8 config never reaches them.

Megatron-Bridge's Beam bridge exports routed experts per expert (``mlp.experts.<n>.{gate,up,down}_proj.weight``),
which the ordinary per-tensor path and vLLM's FusedMoE expert mapping handle, so there is no batched expert spec.

Every quantized linear has a reduction dim divisible by 32 and ``out_features >= 128`` after vLLM's TP-8 sharding
(narrowest: ``k/v_proj`` 1024 -> 128 rows, experts 2560 -> 320), as MXFP8 kernels require.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import (
    BLOCKWISE_FP8,
    MXFP8,
    ModelFp8Spec,
    MoeExpertSpec,
    register_fp8_spec,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.quantize import MXFP8_GROUP_SIZE

_BEAM_FP8_WEIGHT_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".self_attn.gate_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)
_BEAM_EXPERT_PROJ_SUFFIXES = (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")
# Mirrors the released FP8 checkpoint's ``ignore`` list.
_BEAM_FP8_IGNORED_LAYERS = ["lm_head", "model.embed_tokens", "re:.*mlp\\.gate$"]


def is_beam_config(hf_config: Any) -> bool:
    """Return whether an HF config is the Beam layout."""

    text_config = getattr(hf_config, "text_config", None) or hf_config
    model_type = str(getattr(text_config, "model_type", "") or getattr(hf_config, "model_type", ""))
    return model_type == "beam"


def get_beam_fp8_ignored_layers(hf_config: Any, wire_format: str = BLOCKWISE_FP8) -> list[str]:
    """vLLM modules that stay unquantized (same for both wires): see the module docstring."""

    return list(_BEAM_FP8_IGNORED_LAYERS)


def is_quantizable_weight_shape(name: str, shape: Sequence[int], wire_format: str = BLOCKWISE_FP8) -> bool:
    """Return whether an exported HF weight should be serialized as FP8.

    MXFP8 additionally requires the reduction dim to be a multiple of 32 (all of Beam's are).
    """

    if not name.endswith(".weight") or len(shape) != 2:
        return False
    # Routed experts: model.layers.<l>.mlp.experts.<n>.{gate,up,down}_proj.weight
    is_expert = ".mlp.experts." in name and name.endswith(_BEAM_EXPERT_PROJ_SUFFIXES)
    # ``.mlp.gate.weight`` (the router) matches neither list.
    if not (is_expert or name.endswith(_BEAM_FP8_WEIGHT_SUFFIXES)):
        return False
    if wire_format == MXFP8 and shape[1] % MXFP8_GROUP_SIZE != 0:
        return False
    return True


def batched_moe_expert_spec(name: str) -> Optional[MoeExpertSpec]:
    """The Beam bridge exports experts one tensor per expert; nothing is batched."""

    return None


BEAM_FP8_SPEC = register_fp8_spec(
    ModelFp8Spec(
        name="beam",
        matches=is_beam_config,
        should_quantize=is_quantizable_weight_shape,
        ignored_layers=get_beam_fp8_ignored_layers,
        moe_expert_spec=batched_moe_expert_spec,
    )
)
