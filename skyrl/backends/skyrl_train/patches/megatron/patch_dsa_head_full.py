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

Megatron's ``all_to_all_hp2sp`` / ``all_to_all_sp2hp`` move the received / sent row blocks into
place with ``torch.cat`` of strided pieces (forward, and again in their autograd backward), which
runs at ~1/3 of copy bandwidth. ``SKYRL_DSA_HEAD_FULL_FAST_PERMUTE=1`` does the same all-to-alls
with one Triton block copy per direction instead; the values are identical.
"""

import os
from functools import partial

import torch
from loguru import logger

_APPLIED = False
_HOOK = "run_fused_absorbed_sparse_attention"
_FAST_PERMUTE = os.environ.get("SKYRL_DSA_HEAD_FULL_FAST_PERMUTE", "0").lower() in ("1", "true")


def _block_copy(src, dst, blocks, rows, cols, src_strides, dst_strides):
    """``dst[blk, row, :cols] = src[blk, row, :cols]`` with (block, row) strides; unit column stride."""
    import triton

    from skyrl.backends.skyrl_train.patches.megatron._block_copy_kernel import block_copy_kernel

    br, bc = 16, 512
    grid = (triton.cdiv(rows, br), triton.cdiv(cols, bc), blocks)
    block_copy_kernel[grid](src, dst, rows, cols, *src_strides, *dst_strides, BR=br, BC=bc)
    return dst


def _blocks_to_cols(x, tp):
    """``[tp * n, c]`` (row block r from source rank r) -> ``[n, tp * c]``."""
    n, c = x.size(0) // tp, x.size(1)
    return _block_copy(x, x.new_empty(n, tp * c), tp, n, c, (n * c, c), (c, tp * c))


def _cols_to_blocks(x, tp):
    """``[n, tp * c]`` -> ``[tp * n, c]`` (column block r becomes row block r)."""
    n, c = x.size(0), x.size(1) // tp
    return _block_copy(x, x.new_empty(tp * n, c), tp, n, c, (c, tp * c), (n * c, c))


def _all_to_all(x, group):
    out = torch.empty_like(x)
    torch.distributed.all_to_all_single(out, x, group=group)
    return out


class _HeadToSeq(torch.autograd.Function):
    """``[s, h/TP * d]`` -> ``[s/TP, h * d]``: ``all_to_all_hp2sp`` with a single block copy."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _blocks_to_cols(_all_to_all(x.contiguous(), group), group.size())

    @staticmethod
    def backward(ctx, grad):
        return _all_to_all(_cols_to_blocks(grad.contiguous(), ctx.group.size()), ctx.group), None


class _SeqToHead(torch.autograd.Function):
    """``[s/TP, h * d]`` -> ``[s, h/TP * d]``: ``all_to_all_sp2hp`` with a single block copy."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _all_to_all(_cols_to_blocks(x.contiguous(), group.size()), group)

    @staticmethod
    def backward(ctx, grad):
        return _blocks_to_cols(_all_to_all(grad.contiguous(), ctx.group), ctx.group.size()), None


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
        hp2sp = (lambda t: _HeadToSeq.apply(t, group)) if _FAST_PERMUTE else partial(all_to_all_hp2sp, group=group)
        sp2hp = (lambda t: _SeqToHead.apply(t, group)) if _FAST_PERMUTE else partial(all_to_all_sp2hp, group=group)
        q_local = hp2sp(query.reshape(sq * b, h * d)).view(local_sq, b, tp * h, d)
        rank = group.rank()
        idx_local = topk_indices[:, rank * local_sq : (rank + 1) * local_sq]
        out = fn(q_local, key, idx_local, softmax_scale, v_channels, None)
        if out is None:
            return None
        v = out.size(-1)
        out = sp2hp(out.reshape(local_sq * b, tp * h * v))
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
