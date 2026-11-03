"""Triton kernel behind ``patch_dsa_head_full``'s fast permute (kept apart so the patch imports lazily)."""

import triton
import triton.language as tl


@triton.jit
def block_copy_kernel(
    src,
    dst,
    rows,
    cols,
    src_block_stride,
    src_row_stride,
    dst_block_stride,
    dst_row_stride,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    pid_r, pid_c, blk = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    r = (pid_r * BR + tl.arange(0, BR)).to(tl.int64)
    c = pid_c * BC + tl.arange(0, BC)
    mask = (r[:, None] < rows) & (c[None, :] < cols)
    x = tl.load(src + blk * src_block_stride + r[:, None] * src_row_stride + c[None, :], mask=mask)
    tl.store(dst + blk * dst_block_stride + r[:, None] * dst_row_stride + c[None, :], x, mask=mask)
