"""Per-token NVFP4 activations for expert_nvfp4, as vLLM's W4A4 MoE quantizes both expert GEMM inputs.

The quantizer is FlashInfer's ``nvfp4_quantize(x, 1 / (448 * 6), layout_linear, per_token_activation=True)``
itself (vLLM's ``quantize_nvfp4_per_token_input``; absmax, or 4/6 with its 4over6 MSE fast-error path, set per
call). Everything else here is layout, done with built-ins:

- the bf16 round trip (``fake_quantize_per_token``) is the decode of FlashInfer's bytes (:mod:`.decode`); on sm_100
  it and ``pack_per_token`` run as one fused pass instead (:mod:`.fused_rt`, FlashInfer's quantizer arithmetic in
  registers: bit-identical bytes, round trip and write-back; FlashInfer stays the fallback);
- the tile-padded, 128x4-swizzled GEMM operand (``pack_per_token``) is a row copy of those bytes into the padded
  rows plus FlashInfer's ``block_scale_interleave``.
"""

from __future__ import annotations

import torch

from skyrl.backends.skyrl_train.kernels.nvfp4_quant import (
    flashinfer_block_scale_env,
    fused_rt,
)
from skyrl.backends.skyrl_train.kernels.nvfp4_quant.decode import (
    PER_TOKEN_SCALE_INV,
    dequantize_rows,
)


def flashinfer_per_token(x: torch.Tensor, four_over_six: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashInfer's per-token NVFP4 of bf16 ``x [M, K]``: ``(codes [M, K/2] uint8, E4M3 scales [M, K/16] uint8,
    decode scales [M] fp32)`` in the linear layout, absmax or 4/6 block scales."""
    from flashinfer import SfLayout, nvfp4_quantize

    m, k = x.shape
    if not x.is_cuda or x.dtype != torch.bfloat16 or k % 16:
        raise ValueError(f"per-token NVFP4 needs bf16 CUDA rows with K % 16 == 0, got {x.dtype} {tuple(x.shape)}")
    with flashinfer_block_scale_env(four_over_six):
        q, sf, t = nvfp4_quantize(
            x.contiguous(), PER_TOKEN_SCALE_INV, sfLayout=SfLayout.layout_linear, per_token_activation=True
        )
    return q.view(torch.uint8).reshape(m, k // 2), sf.view(torch.uint8).reshape(m, -1)[:, : k // 16], t.reshape(m)


def fake_quantize_per_token(x: torch.Tensor, inplace: bool = False, four_over_six: bool = False) -> torch.Tensor:
    """``dequant(quant(x))`` of bf16 ``x [M, K]`` on vLLM's per-token NVFP4 activation grid. ``inplace``
    overwrites ``x`` (contiguous) and returns it."""
    if x.dim() != 2:
        raise ValueError(f"per-token NVFP4 needs a 2D [M, K] input, got {tuple(x.shape)}")
    if inplace and not x.is_contiguous():
        raise ValueError("in-place per-token NVFP4 needs a contiguous input")
    y = x if inplace else torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if x.shape[0] and fused_rt.supports(x) and fused_rt.per_token_quantize(x, four_over_six, out=y):
        return y
    if x.shape[0]:
        q, sf, t = flashinfer_per_token(x, four_over_six)
        dequantize_rows(q, sf, t, out=y)
    return y


def pack_per_token(
    x: torch.Tensor, dst: torch.Tensor, padded_rows: int, four_over_six: bool = False, write_back: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token NVFP4 of bf16 ``x [M, K]`` packed for a block-scaled GEMM: row ``r`` goes to padded row ``dst[r]``
    of ``(codes [Mp, K/2] uint8, 128x4-swizzled E4M3 scales [Mp * K/16] uint8, decode scales [Mp] fp32)``.
    ``write_back`` also overwrites ``x`` with its round trip (:func:`fake_quantize_per_token`). Padded rows hold
    zero scales. ``K % 64 == 0`` and ``padded_rows % 128 == 0``."""
    from flashinfer import block_scale_interleave

    m, k = x.shape
    if k % 64 or padded_rows % 128 or x.stride(1) != 1:
        raise ValueError(f"packed per-token NVFP4 needs K % 64 == 0, Mp % 128 == 0, unit column stride; got {x.shape}")
    oq = torch.empty((padded_rows, k // 2), dtype=torch.uint8, device=x.device)
    opts = torch.empty((padded_rows,), dtype=torch.float32, device=x.device)
    if m == 0:  # FlashInfer's kernels reject empty launches
        return oq, torch.zeros((padded_rows * (k // 16),), dtype=torch.uint8, device=x.device), opts
    if fused_rt.supports(x) and fused_rt.per_token_fits(k):
        sw = torch.zeros((padded_rows * (k // 16),), dtype=torch.uint8, device=x.device)
        fused_rt.per_token_quantize(
            x, four_over_six, out=x if write_back else None, codes=oq, sf_swizzled=sw, t=opts, dst=dst
        )
        return oq, sw, opts
    lin = torch.zeros((padded_rows, k // 16), dtype=torch.uint8, device=x.device)
    q, sf, t = flashinfer_per_token(x, four_over_six)
    rows = dst.long()
    oq.view(torch.int64).index_copy_(0, rows, q.view(torch.int64))
    lin.index_copy_(0, rows, sf)
    opts.index_copy_(0, rows, t)
    if write_back:
        dequantize_rows(q, sf, t, out=x)
    return oq, block_scale_interleave(lin).view(torch.uint8).reshape(-1), opts
