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

The sparse-attention cost of a query row grows with its position (its top-k keys spread over a
longer prefix: ~75 vs ~103 ms backward for the first vs last quarter of 256k at TP4), so with
contiguous rows the last rank holds up the others at the next all-to-all.
``SKYRL_DSA_HEAD_FULL_ZIGZAG=1`` gives rank r row chunk r of the first half and chunk ``TP - 1 - r``
of the second half instead (ring-attention style balancing; needs ``sq`` divisible by ``2 * TP``).
"""

import os
from functools import partial

import torch
from loguru import logger

_APPLIED = False
_HOOK = "run_fused_absorbed_sparse_attention"
_FAST_PERMUTE = os.environ.get("SKYRL_DSA_HEAD_FULL_FAST_PERMUTE", "0").lower() in ("1", "true")
_ZIGZAG = os.environ.get("SKYRL_DSA_HEAD_FULL_ZIGZAG", "0").lower() in ("1", "true")
_WARNED_ZIGZAG_FALLBACK = False


def _block_copy(src, dst, blocks, rows, cols, src_strides, dst_strides):
    """``dst[blk, row, :cols] = src[blk, row, :cols]`` with (block, row) strides; unit column stride."""
    import triton

    from skyrl.backends.skyrl_train.patches.megatron._block_copy_kernel import (
        block_copy_kernel,
    )

    br, bc = 16, 512
    grid = (triton.cdiv(rows, br), triton.cdiv(cols, bc), blocks)
    block_copy_kernel[grid](src, dst, rows, cols, *src_strides, *dst_strides, BR=br, BC=bc)
    return dst


def _blocks_to_cols(x, tp, out=None):
    """``[tp * n, c]`` (row block r from source rank r) -> ``[n, tp * c]``."""
    n, c = x.size(0) // tp, x.size(1)
    out = x.new_empty(n, tp * c) if out is None else out
    return _block_copy(x, out, tp, n, c, (n * c, c), (c, tp * c))


def _cols_to_blocks(x, tp, out=None):
    """``[n, tp * c]`` -> ``[tp * n, c]`` (column block r becomes row block r)."""
    n, c = x.size(0), x.size(1) // tp
    out = x.new_empty(tp * n, c) if out is None else out
    return _block_copy(x, out, tp, n, c, (c, tp * c), (n * c, c))


def _all_to_all(x, group):
    out = torch.empty_like(x)
    torch.distributed.all_to_all_single(out, x, group=group)
    return out


def _second_half_chunks(x, tp):
    """Views of the second-half row chunks of ``x`` ``[2 * tp * ch, c]``, indexed by rank (reversed)."""
    half = x.size(0) // 2
    ch = half // tp
    return [x[half + (tp - 1 - r) * ch : half + (tp - r) * ch] for r in range(tp)]


def _scatter_rows(x, group, zigzag):
    """``[n, c]`` on every rank -> ``[n / TP, TP * c]``: this rank's query rows, all ranks' columns.

    Rank r gets rows ``[r * n/TP, (r+1) * n/TP)``, or with ``zigzag`` row chunk r of the first half
    and chunk ``TP - 1 - r`` of the second half (two all-to-alls over contiguous views).
    """
    tp = group.size()
    if not zigzag:
        return _blocks_to_cols(_all_to_all(x, group), tp)
    n, c = x.shape
    half, ch = n // 2, n // (2 * tp)
    recv = x.new_empty(2, half, c)
    torch.distributed.all_to_all_single(recv[0], x[:half], group=group)
    torch.distributed.all_to_all(list(recv[1].chunk(tp)), _second_half_chunks(x, tp), group=group)
    out = x.new_empty(2 * ch, tp * c)
    _blocks_to_cols(recv[0], tp, out[:ch])
    _blocks_to_cols(recv[1], tp, out[ch:])
    return out


def _gather_rows(y, group, zigzag):
    """Inverse of :func:`_scatter_rows`: ``[n / TP, TP * c]`` -> ``[n, c]``."""
    tp = group.size()
    if not zigzag:
        return _all_to_all(_cols_to_blocks(y, tp), group)
    ch, c = y.size(0) // 2, y.size(1) // tp
    half = tp * ch
    send = y.new_empty(2, half, c)
    _cols_to_blocks(y[:ch], tp, send[0])
    _cols_to_blocks(y[ch:], tp, send[1])
    out = y.new_empty(2 * half, c)
    torch.distributed.all_to_all_single(out[:half], send[0], group=group)
    torch.distributed.all_to_all(_second_half_chunks(out, tp), list(send[1].chunk(tp)), group=group)
    return out


class _HeadToSeq(torch.autograd.Function):
    """``[s, h/TP * d]`` -> ``[s/TP, h * d]``: ``all_to_all_hp2sp`` with a single block copy."""

    @staticmethod
    def forward(ctx, x, group, zigzag):
        ctx.group, ctx.zigzag = group, zigzag
        return _scatter_rows(x.contiguous(), group, zigzag)

    @staticmethod
    def backward(ctx, grad):
        return _gather_rows(grad.contiguous(), ctx.group, ctx.zigzag), None, None


class _SeqToHead(torch.autograd.Function):
    """``[s/TP, h * d]`` -> ``[s, h/TP * d]``: ``all_to_all_sp2hp`` with a single block copy."""

    @staticmethod
    def forward(ctx, x, group, zigzag):
        ctx.group, ctx.zigzag = group, zigzag
        return _gather_rows(x.contiguous(), group, zigzag)

    @staticmethod
    def backward(ctx, grad):
        return _scatter_rows(grad.contiguous(), ctx.group, ctx.zigzag), None, None


def _warn_zigzag_fallback(sq, tp):
    """Zigzag needs ``sq`` divisible by ``2 * TP``; say once that a call ran with contiguous rows."""
    global _WARNED_ZIGZAG_FALLBACK
    if not _WARNED_ZIGZAG_FALLBACK:
        _WARNED_ZIGZAG_FALLBACK = True
        logger.warning(
            f"SKYRL_DSA_HEAD_FULL_ZIGZAG=1, but sequence length {sq} is not divisible by 2 * TP ({2 * tp}): "
            "head-full sparse attention uses contiguous query rows for such calls (logged once)"
        )


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
        zigzag = _ZIGZAG and sq % (2 * tp) == 0
        if _ZIGZAG and not zigzag:
            _warn_zigzag_fallback(sq, tp)
        # [sq * b, h * d] -> [sq * b / TP, TP * h * d]: rank r keeps rows [r * sq/TP, (r+1) * sq/TP)
        # of every source rank (or its zigzag pair of chunks), and source ranks' heads concatenate
        # in global head order.
        if _FAST_PERMUTE or zigzag:
            hp2sp = lambda t: _HeadToSeq.apply(t, group, zigzag)  # noqa: E731
            sp2hp = lambda t: _SeqToHead.apply(t, group, zigzag)  # noqa: E731
        else:
            hp2sp = partial(all_to_all_hp2sp, group=group)
            sp2hp = partial(all_to_all_sp2hp, group=group)
        q_local = hp2sp(query.reshape(sq * b, h * d)).view(local_sq, b, tp * h, d)
        rank = group.rank()
        if zigzag:
            ch = sq // (2 * tp)
            second = sq // 2 + (tp - 1 - rank) * ch
            idx_local = torch.cat(
                [topk_indices[:, rank * ch : (rank + 1) * ch], topk_indices[:, second : second + ch]], dim=1
            )
        else:
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
