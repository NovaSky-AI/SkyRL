"""Block-tile-align the top-k axis of Megatron's cuDNN DSA backward.

``megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels`` runs the
DSA forward on FlashMLA and the backward on cudnn-frontend, and only the forward
normalizes the top-k axis:

* ``_dsa_fwd_flash_mla`` rounds the top-k width up to ``_get_topk_alignment()`` (512 on
  SM90+) before calling FlashMLA.
* ``_run_sparse_attention_backward`` pads the *head* axis (``_get_head_padding``, 8 -> 64
  on sm100) but forwards ``global_idxs`` at its native width. ``_get_topk_alignment`` has
  exactly two references in that module: its definition and the forward.

For a caller whose width already equals a 512-aligned ``index_topk``, the omission is
invisible. The kpool indexer path is not such a caller: ``fused_qk_topk_kpool`` returns
``[batch, queries, index_topk + pool_size - 1]`` because ``_append_tail_to_topk``
concatenates ``pool_size - 1`` tail columns, and the kpool branch of ``DSA.forward`` never
assigns ``topk_length``, so that ragged width is all the backward has to go on. With
``index_topk=2048`` and ``pool_size=4`` the width is 2051.

cudnn-frontend 1.26.0 made that width a kernel parameter (``max_topk =
topk_idxs.shape[1]``, and part of the compile-cache key). Its sm100 backward tiles the
axis as ``ceil_div(topk_length[row], block_tile)`` with ``block_tile = 64``, reads
``mTopkIdxs[idx]`` under an ``idx < max_topk`` guard, and then dereferences
``mKV[topk_idx]`` with no ``topk_idx >= 0`` check on non-final tiles (``_load_kv_rows``,
``is_first=False``). A width that is not a multiple of 64 is outside what that arithmetic
was written for, and the result is an illegal memory access during ``backward`` -- reported
asynchronously, so the traceback names a later collective (typically the optimizer's
grad-norm all-reduce) rather than the attention backward.

This module pads ``global_idxs`` along the top-k axis to a multiple of 64, mirroring what
the forward already does on its own axis. The pad value is ``0``, not ``-1``: the forward's
own comment notes that the backward needs only non-negative placeholders for ignored
slots, and it clamps sentinels to 0 for exactly this call. ``topk_length`` is untouched, so
the appended slots are never counted as valid and the padding is semantically inert.

``SKYRL_DSA_BACKWARD_CHECK=1`` additionally validates the contract the cuDNN kernel relies
on but never states: indices in ``[0, skv * b)``, ``topk_length`` in ``[0, width]``, int32,
and contiguous. That costs one device-to-host sync per DSA backward call, hence opt-in --
but it converts an asynchronous ``cudaErrorIllegalAddress`` blamed on an unrelated
collective into a Python exception naming the violated invariant.

Remove once megatron-core aligns the top-k axis in the backward as it does in the forward.
"""

from loguru import logger

from skyrl.env_vars import SKYRL_DSA_BACKWARD_CHECK

_APPLIED = False

# cudnn-frontend hardcodes this in both _interface_sm90.py and _interface_sm100.py.
_BLOCK_TILE = 64


def align_topk_axis(global_idxs):
    """Return ``global_idxs`` with its last axis padded up to a multiple of ``_BLOCK_TILE``.

    The tensor is returned unchanged when the width is already aligned. The pad value is
    ``0``, not ``-1``: the appended slots are never counted as valid because
    ``topk_length`` is left alone, and the kernel dereferences ``mKV[topk_idx]`` without a
    negativity check on non-final tiles.
    """
    import torch

    pad_width = -global_idxs.shape[-1] % _BLOCK_TILE
    if not pad_width:
        return global_idxs
    return torch.nn.functional.pad(global_idxs, (0, pad_width), value=0)


def patch_dsa_backward_topk_align() -> None:
    """Patch ``_run_sparse_attention_backward`` to pad the top-k axis to ``_BLOCK_TILE``."""
    global _APPLIED
    if _APPLIED:
        return

    import functools

    try:
        from megatron.core.transformer.experimental_attention_variant import (
            dsa_cudnn_kernels,
        )
    except ImportError:
        # A megatron-core rev without the cuDNN DSA kernels. Nothing to align.
        return

    original = getattr(dsa_cudnn_kernels, "_run_sparse_attention_backward", None)
    if original is None:
        return

    @functools.wraps(original)
    def _run_sparse_attention_backward(*, global_idxs, topk_length, skv, b, **kwargs):
        if SKYRL_DSA_BACKWARD_CHECK:
            _check_backward_contract(global_idxs, topk_length, skv, b, global_idxs.shape[-1])

        return original(global_idxs=align_topk_axis(global_idxs), topk_length=topk_length, skv=skv, b=b, **kwargs)

    dsa_cudnn_kernels._run_sparse_attention_backward = _run_sparse_attention_backward
    _APPLIED = True
    logger.info(
        f"Applied Megatron DSA backward top-k alignment to {_BLOCK_TILE}; contract checks "
        + ("ON" if SKYRL_DSA_BACKWARD_CHECK else "off (set SKYRL_DSA_BACKWARD_CHECK=1)")
    )


def _check_backward_contract(global_idxs, topk_length, skv: int, b: int, width: int) -> None:
    """Raise if the cuDNN sparse-attention backward would index out of bounds.

    Raises:
        RuntimeError: If an index or length is out of range, or if either tensor is not a
            contiguous int32 tensor.
    """
    import torch

    # One stacked reduction so the whole check costs a single sync, not four.
    bad = torch.stack(
        (
            (global_idxs < 0).any(),
            (global_idxs >= skv * b).any(),
            (topk_length < 0).any(),
            (topk_length > width).any(),
        )
    )
    if bool(bad.any()):
        negative_index, index_too_large, negative_length, length_too_large = (bool(flag) for flag in bad)
        raise RuntimeError(
            "cuDNN sparse_attention_backward would index out of bounds. "
            f"negative_index={negative_index} index_ge_skv_times_b={index_too_large} "
            f"negative_topk_length={negative_length} topk_length_gt_width={length_too_large}; "
            f"width={width} skv={skv} b={b} num_rows={global_idxs.shape[0]} "
            f"global_idxs={_value_range(global_idxs)} topk_length={_value_range(topk_length)}"
        )
    if global_idxs.dtype != torch.int32 or topk_length.dtype != torch.int32:
        raise RuntimeError(
            "cuDNN reinterprets these as int32 without checking: "
            f"global_idxs={global_idxs.dtype} topk_length={topk_length.dtype}"
        )
    if not global_idxs.is_contiguous() or not topk_length.is_contiguous():
        raise RuntimeError(
            "cuDNN takes strides from the tensors but the sm100 path assumes packed rows: "
            f"global_idxs.is_contiguous()={global_idxs.is_contiguous()} "
            f"topk_length.is_contiguous()={topk_length.is_contiguous()}"
        )


def _value_range(tensor) -> str:
    """``[min, max]`` for the error message; ``min()`` raises on an empty tensor."""
    if tensor.numel() == 0:
        return "empty"
    return f"[{int(tensor.min())}, {int(tensor.max())}]"
