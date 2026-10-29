"""Hybrid stack spec for DeepSeek-V4.1: megatron-core's default hybrid stack with the C/H/W attention
layers swapped to the V4.1 attention (``attention.py``). Everything else (MoE layers, mHC wrapping, the
stack) is megatron-core's."""

import copy
from functools import partial

from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_stack_spec as default_hybrid_stack_spec,
)

from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.attention import (
    CompressedSparseAttention2,
    DSv41HybridSelfAttention,
)


def deepseek_v41_hybrid_stack_spec():
    """Named factory (bridge providers serialize it by import path)."""
    spec = copy.deepcopy(default_hybrid_stack_spec)
    for name in ("csa_layer", "csa_qk_layernorm_layer"):
        layer_spec = getattr(spec.submodules, name)
        attention = layer_spec.submodules.self_attention
        attention.module = DSv41HybridSelfAttention
        attention.submodules.core_attention = partial(CompressedSparseAttention2)
    return spec
