"""Row kernels behind the fused once-per-node MoE dispatch (``_moe_twohop_fused``).

Two kernels cover every data move of the fused path:

- ``gather_rows``: ``dst[i, dst_col : dst_col + n] = src[idx[i], src_col : src_col + n]``. A pure
  copy, so it runs on 16-bit words and serves every activation dtype. It writes straight into a
  column window of a packed message buffer.
- ``gather_sum_rows``: ``dst[i, :n] = sum over k of src[idx[i, k], :n]`` for ``idx[i, k] >= 0``.
  Each output row has a small, fixed number of candidate inputs (the fan-in), so the sum is a
  gather. Every output element is written once: no atomics, and the result is deterministic. The
  sum is accumulated in FP32 and stored in the destination dtype.

Each kernel has a Triton implementation and a plain-tensor reference with the same semantics.
``SKYRL_MOE_NODE_DEDUP_KERNELS=torch`` selects the reference (the default on CPU tensors), which
lets the whole fused path run without a GPU and isolates a kernel fault on one.
"""

import os

import torch

_KERNELS_ENV = "SKYRL_MOE_NODE_DEDUP_KERNELS"
_BR, _BC = 4, 512  # rows x columns per Triton program

_triton_kernels = None


def _use_triton(t: torch.Tensor) -> bool:
    return t.is_cuda and os.environ.get(_KERNELS_ENV, "triton").lower() != "torch"


def _kernels():
    """Compile lazily: importing this module must not need Triton or a GPU."""
    global _triton_kernels
    if _triton_kernels is not None:
        return _triton_kernels
    import triton
    import triton.language as tl

    @triton.jit
    def gather_rows_kernel(
        src,
        dst,
        idx,
        n_rows,
        n_cols,
        src_stride,
        dst_stride,
        src_col,
        dst_col,
        BR: tl.constexpr,
        BC: tl.constexpr,
    ):
        r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
        c = (tl.program_id(1) * BC + tl.arange(0, BC)).to(tl.int64)
        rmask = r < n_rows
        mask = rmask[:, None] & (c[None, :] < n_cols)
        s = tl.load(idx + r, mask=rmask, other=0).to(tl.int64)
        v = tl.load(src + s[:, None] * src_stride + src_col + c[None, :], mask=mask)
        tl.store(dst + r[:, None] * dst_stride + dst_col + c[None, :], v, mask=mask)

    @triton.jit
    def gather_sum_rows_kernel(
        src,
        dst,
        idx,
        n_rows,
        n_cols,
        src_stride,
        dst_stride,
        FAN: tl.constexpr,
        BR: tl.constexpr,
        BC: tl.constexpr,
    ):
        r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
        c = (tl.program_id(1) * BC + tl.arange(0, BC)).to(tl.int64)
        rmask = r < n_rows
        mask = rmask[:, None] & (c[None, :] < n_cols)
        acc = tl.zeros((BR, BC), dtype=tl.float32)
        for k in tl.static_range(FAN):
            s = tl.load(idx + r * FAN + k, mask=rmask, other=-1).to(tl.int64)
            valid = mask & (s[:, None] >= 0)
            v = tl.load(src + s[:, None] * src_stride + c[None, :], mask=valid, other=0.0)
            acc += v.to(tl.float32)
        tl.store(dst + r[:, None] * dst_stride + c[None, :], acc.to(dst.dtype.element_ty), mask=mask)

    _triton_kernels = (triton, gather_rows_kernel, gather_sum_rows_kernel)
    return _triton_kernels


def gather_rows(
    src: torch.Tensor, idx: torch.Tensor, dst: torch.Tensor, n_cols: int, src_col: int = 0, dst_col: int = 0
) -> None:
    """``dst[i, dst_col:dst_col+n_cols] = src[idx[i], src_col:src_col+n_cols]`` for every row of ``dst``.

    ``src`` and ``dst`` are contiguous 2-D tensors of the same dtype; ``idx`` is int64.
    """
    n_rows = dst.size(0)
    if n_rows == 0 or n_cols == 0:
        return
    if not _use_triton(dst):
        dst[:, dst_col : dst_col + n_cols] = src[:, src_col : src_col + n_cols].index_select(0, idx)
        return
    triton, kernel, _ = _kernels()
    grid = (triton.cdiv(n_rows, _BR), triton.cdiv(n_cols, _BC))
    kernel[grid](src, dst, idx, n_rows, n_cols, src.size(1), dst.size(1), src_col, dst_col, BR=_BR, BC=_BC)


def gather_sum_rows(src: torch.Tensor, idx: torch.Tensor, dst: torch.Tensor, n_cols: int) -> None:
    """``dst[i, :n_cols] = sum_k src[idx[i, k], :n_cols]`` over the entries with ``idx[i, k] >= 0``.

    ``src`` and ``dst`` are contiguous 2-D float tensors of the same dtype; ``idx`` is a contiguous
    int64 ``[rows of dst, fan-in]`` table. Columns of ``dst`` past ``n_cols`` are left untouched.
    """
    n_rows = dst.size(0)
    if n_rows == 0 or n_cols == 0:
        return
    if not _use_triton(dst):
        acc = torch.zeros(n_rows, n_cols, dtype=torch.float32, device=dst.device)
        if src.size(0):
            for k in range(idx.size(1)):
                col = idx[:, k]
                rows = src[:, :n_cols].index_select(0, col.clamp_min(0)).float()
                acc += rows * (col >= 0).unsqueeze(1)
        dst[:, :n_cols] = acc.to(dst.dtype)
        return
    triton, _, kernel = _kernels()
    grid = (triton.cdiv(n_rows, _BR), triton.cdiv(n_cols, _BC))
    kernel[grid](src, dst, idx, n_rows, n_cols, src.size(1), dst.size(1), FAN=idx.size(1), BR=_BR, BC=_BC)
