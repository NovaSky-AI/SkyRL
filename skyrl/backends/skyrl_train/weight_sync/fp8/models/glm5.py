"""GLM-5 / GLM-5.3 (``glm_moe_dsa``) ``ModelFp8Spec`` for serialized blockwise FP8 weight sync.

Follows the layout of zai-org/GLM-5-FP8 (128x128 blocks, e4m3): every attention and MLP
linear is FP8, including ``kv_b_proj`` and the routed / shared experts. BF16 stays for the
norms, the router (``mlp.gate`` and its ``e_score_correction_bias``), the embeddings and
``lm_head``, and the DSA indexer's ``wk`` and ``weights_proj``: vLLM fuses those two into
``indexer.wk_weights_proj``, a linear it builds without a quantization config, so both must
arrive unquantized (GLM-5-FP8 keeps only ``weights_proj`` unconverted and vLLM dequantizes
``wk`` on load). The indexer's ``wq_b`` is an ordinary FP8 linear.

Megatron-Bridge's GLM5Bridge exports routed experts per expert
(``mlp.experts.<n>.{gate,up,down}_proj.weight``), which the ordinary per-tensor path and vLLM's
FusedMoE expert mapping already handle, so no tensor stacks experts.
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

# Linear modules synced as FP8, by the part of a decoder layer that holds them.
_GLM5_FP8_ATTENTION_MODULES = (
    "self_attn.q_a_proj",
    "self_attn.q_b_proj",
    "self_attn.kv_a_proj_with_mqa",
    "self_attn.kv_b_proj",
    "self_attn.o_proj",
    "self_attn.indexer.wq_b",
)
# Layers below first_k_dense_replace have a dense MLP; the rest are MoE.
_GLM5_FP8_DENSE_MLP_MODULES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
_GLM5_FP8_SHARED_EXPERT_MODULES = (
    "mlp.shared_experts.gate_proj",
    "mlp.shared_experts.up_proj",
    "mlp.shared_experts.down_proj",
)
_GLM5_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_GLM5_FP8_WEIGHT_SUFFIXES = tuple(
    f".{module}.weight"
    for module in (*_GLM5_FP8_ATTENTION_MODULES, *_GLM5_FP8_DENSE_MLP_MODULES, *_GLM5_FP8_SHARED_EXPERT_MODULES)
)
_GLM5_EXPERT_PROJ_SUFFIXES = tuple(f".{projection}.weight" for projection in _GLM5_EXPERT_PROJECTIONS)


def is_glm5_config(hf_config: Any) -> bool:
    """Return whether an HF config is the GLM-5 DSA MoE layout."""

    text_config = getattr(hf_config, "text_config", None) or hf_config
    model_type = str(getattr(text_config, "model_type", "") or getattr(hf_config, "model_type", ""))
    return model_type == "glm_moe_dsa"


# MXFP8 only: MLA's kv_b_proj stays BF16. vLLM folds kv_b_proj into the absorbed W_UK_T / W_UV
# at load; with no direct MXFP8 dequant, it reconstructs the weight by running the MXFP8 GEMM on
# an identity matrix (``get_and_maybe_dequant_weights``), and after a level-2 sleep + reload that
# path produced wrong W_UK_T / W_UV in every layer (garbage rollouts; pre-train AIME -1.0). A BF16
# kv_b_proj takes vLLM's unquantized path. ~29 MB per layer.
_GLM5_MXFP8_BF16_SUFFIXES = (".self_attn.kv_b_proj.weight",)
_GLM5_MXFP8_BASE_EXCLUDE_LIST = ["re:.*self_attn\\.kv_b_proj"]


def get_glm5_base_exclude_list(hf_config: Any, wire_format: str = BLOCKWISE_FP8) -> list[str]:
    """vLLM modules to build unquantized: ``kv_b_proj`` on the MXFP8 wire, nothing on blockwise.

    Every other linear left in BF16 on the wire (router, indexer ``wk_weights_proj``) is built by
    vLLM without a quantization config, and the embeddings / ``lm_head`` / norms are not
    ``LinearBase`` modules, so the FP8 config never reaches them. Every quantized linear has a
    reduction dim divisible by 32 and ``out_features >= 128``, as MXFP8 kernels require.
    """

    return list(_GLM5_MXFP8_BASE_EXCLUDE_LIST) if wire_format == MXFP8 else []


def is_quantizable_weight_shape(name: str, shape: Sequence[int], wire_format: str = BLOCKWISE_FP8) -> bool:
    """Return whether an exported HF weight should be serialized as FP8.

    MXFP8 additionally requires the reduction dim to be a multiple of 32.
    """

    if not name.endswith(".weight") or len(shape) != 2:
        return False
    # Routed experts: model.layers.<l>.mlp.experts.<n>.{gate,up,down}_proj.weight
    is_expert = ".mlp.experts." in name and name.endswith(_GLM5_EXPERT_PROJ_SUFFIXES)
    if not (name.endswith(_GLM5_FP8_WEIGHT_SUFFIXES) or is_expert):
        return False
    if wire_format == MXFP8 and (shape[1] % MXFP8_GROUP_SIZE != 0 or name.endswith(_GLM5_MXFP8_BF16_SUFFIXES)):
        return False
    return True


def get_glm5_fp8_modules(hf_config: Any, wire_format: str = BLOCKWISE_FP8) -> list[str]:
    """Return the tensors GLM-5 syncs as FP8, layer by layer, minus ``.weight``.

    Names follow GLM5Bridge's export, which covers the decoder layers only (no MTP layer).
    A MoE layer's routed experts are every expert's ``gate_proj``, ``up_proj`` and
    ``down_proj``. The MXFP8 wire sends ``kv_b_proj`` BF16, so it is not listed there.
    """

    text_config = getattr(hf_config, "text_config", None) or hf_config
    attention_modules = _GLM5_FP8_ATTENTION_MODULES
    if wire_format == MXFP8:
        attention_modules = tuple(
            module for module in attention_modules if f".{module}.weight" not in _GLM5_MXFP8_BF16_SUFFIXES
        )
    moe_mlp_modules = (
        *(
            f"mlp.experts.{expert}.{projection}"
            for expert in range(text_config.n_routed_experts)
            for projection in _GLM5_EXPERT_PROJECTIONS
        ),
        *_GLM5_FP8_SHARED_EXPERT_MODULES,
    )
    modules = []
    for layer_idx in range(text_config.num_hidden_layers):
        is_dense = layer_idx < text_config.first_k_dense_replace
        for module in (*attention_modules, *(_GLM5_FP8_DENSE_MLP_MODULES if is_dense else moe_mlp_modules)):
            modules.append(f"model.layers.{layer_idx}.{module}")
    return modules


GLM5_FP8_SPEC = register_fp8_spec(
    ModelFp8Spec(
        name="glm5",
        matches=is_glm5_config,
        should_quantize=is_quantizable_weight_shape,
        base_exclude_list=get_glm5_base_exclude_list,
        fp8_modules=get_glm5_fp8_modules,
    )
)
