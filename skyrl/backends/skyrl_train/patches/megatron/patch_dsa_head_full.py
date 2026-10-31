"""Run absorbed-MLA sparse attention head-full: all heads for 1/TP of the queries per rank.

Under tensor parallelism every rank runs the sparse-attention kernel on its ``heads / TP`` heads
for the whole sequence. With GLM-5.3-Flash at TP8 that is 8 heads per rank. TileLang's SparseMLA
pads to a 16-head tile, and cuDNN/FlashMLA pads to 128 heads. Each query's top-k latent keys are
gathered once per rank, so with few heads the kernels are bound by those gathers and run at ~1% of
peak.

Absorbed MLA has a single latent KV head shared by every query head (``key`` is
``[sk, b, 1, d]``, identical on all TP ranks) and per-query top-k indices, so attention is
independent per query row. ``SKYRL_DSA_HEAD_FULL=1`` wraps the backend's
``run_fused_absorbed_sparse_attention`` hook in an all-to-all (Ulysses-style):
``query [sq, b, h/TP, d] -> [sq/TP, b, h, d]``, the kernel runs on the rank's contiguous query rows
with their slice of ``topk_indices``, and the output goes back ``[sq/TP, b, h, v] -> [sq, b, h/TP, v]``.
The key gradient on each rank is still a partial sum (over its query rows instead of its heads)
that the existing sequence-parallel reduce-scatter adds up, so the result is the same up to
summation order.

Applies to any ``dsa_kernel_backend``. It falls back to the unwrapped hook when TP is 1, ``sq`` is
not divisible by TP, or ``topk_length`` is set (cross-layer index sharing).
"""

import os

from loguru import logger

_APPLIED = False
_HOOK = "run_fused_absorbed_sparse_attention"


def _head_full(fn):
    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.mappings import (
        all_to_all_hp2sp,
        all_to_all_sp2hp,
    )

    def run(query, key, topk_indices, softmax_scale, v_channels, topk_length=None):
        group = mpu.get_tensor_model_parallel_group()
        tp = group.size()
        sq, b, h, d = query.shape
        if tp == 1 or sq % tp != 0 or topk_length is not None or topk_indices.size(1) != sq:
            return fn(query, key, topk_indices, softmax_scale, v_channels, topk_length)
        local_sq = sq // tp
        # [sq * b, h * d] -> [sq * b / TP, TP * h * d]: rank r keeps rows [r * sq/TP, (r+1) * sq/TP)
        # of every source rank, and source ranks' heads concatenate in global head order.
        q_local = all_to_all_hp2sp(query.reshape(sq * b, h * d), group=group).view(local_sq, b, tp * h, d)
        rank = group.rank()
        idx_local = topk_indices[:, rank * local_sq : (rank + 1) * local_sq]
        out = fn(q_local, key, idx_local, softmax_scale, v_channels, None)
        if out is None:
            return None
        v = out.size(-1)
        out = all_to_all_sp2hp(out.reshape(local_sq * b, tp * h * v), group=group)
        return out.view(sq, b, h, v)

    return run


def apply_dsa_head_full_patch() -> None:
    global _APPLIED
    if _APPLIED or os.environ.get("SKYRL_DSA_HEAD_FULL", "0").lower() not in ("1", "true"):
        return
    from megatron.core.transformer.experimental_attention_variant import dsa_kernels

    original = dsa_kernels._resolve_fused_hook
    wrapped = {}

    def _resolve_fused_hook(config, hook_name):
        fn = original(config, hook_name)
        if hook_name != _HOOK or fn is None:
            return fn
        if fn not in wrapped:
            wrapped[fn] = _head_full(fn)
        return wrapped[fn]

    dsa_kernels._resolve_fused_hook = _resolve_fused_hook
    _APPLIED = True
    logger.info("DSA: head-full sparse attention (all-to-all over the tensor-parallel group)")
