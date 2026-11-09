"""Beam (``model_type="beam"``) ``ModelFp8Spec`` for serialized FP8 weight sync (blockwise and MXFP8).

The quantize / keep-BF16 split follows reflection/Beam-1.0-501B-A23B-FP8 (compressed-tensors, per-channel
FP8_DYNAMIC; which layers are FP8 is the same question for both wires): every attention and MLP ``Linear``
is FP8 -- ``q/k/v/o_proj``, the attention output gate ``gate_proj``, the dense MLP's ``gate/up/down_proj``
and every routed expert's ``gate/up/down_proj``. The embeddings, ``lm_head``, all norms, the attention
``gate_bias`` and the router (``mlp.gate`` and its ``e_score_correction_bias``, both fp32) stay unquantized.

On the vLLM side (``vllm.models.beam``), HF ``self_attn.gate_proj`` loads into ``output_gate_proj``,
fused into ``qkv_gate_proj`` with ``q/k/v`` when the KV heads cover the TP size, otherwise kept beside
``qkv_proj``; every shard of a fused module is FP8 here, as vLLM requires. The router is a ``GateLinear``
built without a quantization config, and the embeddings, ``lm_head`` and norms are not ``LinearBase``
modules, so the FP8 config never reaches them; the exclude list mirrors the released checkpoint's
``ignore`` list anyway.

Megatron-Bridge's ``BeamBridge`` exports routed experts per expert (``mlp.experts.<n>.{gate,up,down}_proj``),
which the per-tensor path and vLLM's FusedMoE expert mapping handle, so no tensor stacks experts. Every
quantized linear of the released model has a reduction dim divisible by 32 and ``out_features >= 128``
after vLLM's TP-8 sharding (narrowest: ``k/v_proj`` 1024 -> 128 rows, experts 2560 -> 320), as MXFP8
kernels require.
"""

from __future__ import annotations

from typing import Any, Sequence

from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import (
    BLOCKWISE_FP8,
    MXFP8,
    ModelFp8Spec,
    register_fp8_spec,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.quantize import MXFP8_GROUP_SIZE

_BEAM_FP8_ATTENTION_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.gate_proj",
    "self_attn.o_proj",
)
_BEAM_FP8_DENSE_MLP_MODULES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
_BEAM_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_BEAM_FP8_WEIGHT_SUFFIXES = tuple(
    f".{module}.weight" for module in (*_BEAM_FP8_ATTENTION_MODULES, *_BEAM_FP8_DENSE_MLP_MODULES)
)
_BEAM_EXPERT_PROJ_SUFFIXES = tuple(f".{projection}.weight" for projection in _BEAM_EXPERT_PROJECTIONS)
# The released FP8 checkpoint's ``ignore`` list.
_BEAM_BASE_EXCLUDE_LIST = ["lm_head", "model.embed_tokens", "re:.*mlp\\.gate$"]


def is_beam_config(hf_config: Any) -> bool:
    """Return whether an HF config is the Beam layout."""

    text_config = getattr(hf_config, "text_config", None) or hf_config
    model_type = str(getattr(text_config, "model_type", "") or getattr(hf_config, "model_type", ""))
    return model_type == "beam"


def get_beam_base_exclude_list(hf_config: Any, wire_format: str = BLOCKWISE_FP8) -> list[str]:
    """vLLM modules to build unquantized, on both wires: see the module docstring."""

    return list(_BEAM_BASE_EXCLUDE_LIST)


def is_quantizable_weight_shape(name: str, shape: Sequence[int], wire_format: str = BLOCKWISE_FP8) -> bool:
    """Return whether an exported HF weight should be serialized as FP8.

    MXFP8 additionally requires the reduction dim to be a multiple of 32 (all of Beam's are).
    """

    if not name.endswith(".weight") or len(shape) != 2:
        return False
    # Routed experts: model.layers.<l>.mlp.experts.<n>.{gate,up,down}_proj.weight. The router,
    # ``mlp.gate.weight``, matches neither this nor the dense ``mlp.gate_proj``.
    is_expert = ".mlp.experts." in name and name.endswith(_BEAM_EXPERT_PROJ_SUFFIXES)
    if not (is_expert or name.endswith(_BEAM_FP8_WEIGHT_SUFFIXES)):
        return False
    if wire_format == MXFP8 and shape[1] % MXFP8_GROUP_SIZE != 0:
        return False
    return True


def get_beam_fp8_modules(hf_config: Any, wire_format: str = BLOCKWISE_FP8) -> list[str]:
    """Return the tensors Beam syncs as FP8, layer by layer, minus ``.weight``.

    ``mlp_layer_types`` picks each layer's dense MLP or routed experts (every expert's ``gate_proj``,
    ``up_proj`` and ``down_proj``). The same on both wires.
    """

    text_config = getattr(hf_config, "text_config", None) or hf_config
    expert_modules = tuple(
        f"mlp.experts.{expert}.{projection}"
        for expert in range(text_config.num_experts)
        for projection in _BEAM_EXPERT_PROJECTIONS
    )
    modules = []
    for layer_idx, mlp_type in enumerate(text_config.mlp_layer_types):
        mlp_modules = _BEAM_FP8_DENSE_MLP_MODULES if mlp_type == "dense" else expert_modules
        for module in (*_BEAM_FP8_ATTENTION_MODULES, *mlp_modules):
            modules.append(f"model.layers.{layer_idx}.{module}")
    return modules


BEAM_FP8_SPEC = register_fp8_spec(
    ModelFp8Spec(
        name="beam",
        matches=is_beam_config,
        should_quantize=is_quantizable_weight_shape,
        base_exclude_list=get_beam_base_exclude_list,
        fp8_modules=get_beam_fp8_modules,
    )
)
