"""NVFP4 decode to bf16, and the FlashInfer settings expert_nvfp4 quantizes with.

Every NVFP4 tensor expert_nvfp4 rounds is quantized by FlashInfer (``fp4_quantize`` for expert weights,
``nvfp4_quantize(per_token_activation=True)`` for activations, the rollouts' own quantizers). The bf16 round
trip a bf16 GEMM or wgrad needs is the decode of those bytes, :func:`dequantize_rows`::

    y = bf16((e2m1 * sf) * scale)        e2m1 * sf exact in fp32, one fp32 rounding, one bf16 rounding

with ``scale`` the row's decode scale (``1 / gs`` of its expert for weights, the per-token scale for
activations). That is FlashInfer's own dequant arithmetic (``nvfp4_kv_dequantize``), but FlashInfer has no
dequant with a scale per row, and its scalar one is ~8x slower than this kernel at GLM-5.3 shapes (one
launch per expert, or a second pass for the row scale), so this decode is the one custom NVFP4 kernel. A zero
code keeps its sign bit (FlashInfer does; -0.0 equals 0.0).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# FlashInfer's per-token NVFP4 decode scale is amax(row) * (1 / (448 * 6)) (vLLM's quantize_nvfp4_per_token_input).
PER_TOKEN_SCALE_INV = 1.0 / (448.0 * 6.0)
# 16-element blocks per program of the decode (one warp each).
_DECODE_BLOCKS = 64


@triton.jit
def _decode_kernel(
    q_ptr,  # (M, K/8) int32: packed E2M1 codes, 8 per word, element 2i in the low nibble
    s_ptr,  # (M, sf_stride) uint8: E4M3 block scales, linear layout
    t_ptr,  # (M,) fp32 decode scale of each row, stride t_stride
    y_ptr,  # (M, y_stride) bf16 out
    K,
    sf_stride,
    t_stride,
    y_stride,
    CB: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1) * CB + tl.arange(0, CB)
    bmask = blk < K // 16
    m = bmask[:, None]
    w = tl.load(q_ptr + row * (K // 8) + blk[:, None] * 2 + tl.arange(0, 2)[None, :], mask=m, other=0)
    code = tl.reshape((w[:, :, None] >> (tl.arange(0, 8) * 4)[None, None, :]) & 0xF, (CB, 16))
    mag = code & 7
    v = tl.where(mag == 7, 6.0, tl.where(mag == 6, 4.0, tl.where(mag == 5, 3.0, mag.to(tl.float32) * 0.5)))
    # The sign bit as fp32 bits, so a zero code with the sign set decodes to -0.0 as FlashInfer's does.
    v = (v.to(tl.int32, bitcast=True) | ((code & 8) << 28)).to(tl.float32, bitcast=True)
    b = tl.load(s_ptr + row * sf_stride + blk, mask=bmask, other=0).to(tl.int32)
    e = b >> 3
    sf = tl.where(
        e == 0, (b & 7).to(tl.float32) * 0.001953125, (((e + 120) << 23) | ((b & 7) << 20)).to(tl.float32, bitcast=True)
    )
    t = tl.load(t_ptr + row * t_stride)
    offs = blk[:, None] * 16 + tl.arange(0, 16)[None, :]
    tl.store(y_ptr + row * y_stride + offs, ((v * sf[:, None]) * t).to(tl.bfloat16), mask=m)


def dequantize_rows(
    q: torch.Tensor, sf: torch.Tensor, scale: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """bf16 ``[M, K]`` decode of NVFP4 rows: codes ``q [M, K/2]`` uint8, linear E4M3 block scales ``sf [M, >= K/16]``
    (uint8 or float8_e4m3fn, unit column stride), fp32 decode scales ``scale [M]`` (any stride); into ``out`` (unit
    column stride) if given."""
    m, k = q.shape[0], q.shape[1] * 2
    if out is None:
        out = torch.empty((m, k), dtype=torch.bfloat16, device=q.device)
    elif out.shape != (m, k) or out.dtype != torch.bfloat16 or out.stride(1) != 1:
        raise ValueError(f"out must be bf16 [{m}, {k}] with unit column stride, got {out.dtype} {tuple(out.shape)}")
    sf = sf.view(torch.uint8)
    if sf.shape[0] != m or sf.shape[1] < k // 16 or sf.stride(1) != 1 or scale.shape != (m,) or k % 16:
        raise ValueError(
            f"NVFP4 decode of [{m}, {k}]: got scales {tuple(sf.shape)} and decode scales {tuple(scale.shape)}"
        )
    if m:
        with torch.cuda.device(q.device):
            _decode_kernel[(m, triton.cdiv(k // 16, _DECODE_BLOCKS))](
                q.contiguous().view(torch.int32), sf, scale, out, k, sf.stride(0), scale.stride(0), out.stride(0),
                CB=_DECODE_BLOCKS, num_warps=1,
            )  # fmt: skip
    return out
