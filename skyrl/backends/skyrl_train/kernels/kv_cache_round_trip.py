"""One-pass dequantized vLLM DS-MLA KV cache rows (``nvfp4_ds_mla`` / ``fp8_ds_mla``) in megatron's absorbed-key order.

Imports vLLM (its jit helpers are globals here). vLLM writes these caches with ``fused_norm_rope``
(``vllm.models.deepseek_v32.common.kernels``, its MLA program) and
reads them back with ``cp_gather_and_upconvert_*_kv_cache``. This kernel reads each pre-norm ``[latent | k_pe]`` row
once and writes ``dequant(cache_row)`` once, in bf16:

- the RMSNorm and RoPE are vLLM's own ``@triton.jit`` helpers (``_rms_norm``, ``_get_cos_sin``), called with the
  same block shapes, one program per token and the same default launch configuration as vLLM's writer, so the fp32
  values the cache rounds from are vLLM's;
- the rounding is the writer's expressions, verbatim: ``nvfp4_ds_mla`` E2M1 values with E4M3 scales per 16
  (``sf = e4m3(max(amax * (1/6), 2^-9))``, ``q = rn_e2m1(x * (1/sf))``) and E4M3 RoPE; ``fp8_ds_mla`` E4M3 values
  with power-of-two fp32 scales per 128 and bf16 RoPE;
- the decode is exact in bf16 (``e2m1 * e4m3`` has at most 6 significant bits, ``e4m3 * 2^k`` at most 4), so it
  equals vLLM's reader;
- the RoPE half is stored deinterleaved (``[x[0::2] | x[1::2]]``), megatron's MLA RoPE layout; vLLM keeps it
  interleaved.

Bit-identical to vLLM's writer + reader (tested against both); it skips the cache bytes and the writer's query and
indexer programs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from vllm.models.deepseek_v32.common.kernels import _FP8_MAX, _get_cos_sin, _rms_norm

_KV_DIM = 512
_ROPE_DIM = 64


@triton.jit
def _signed_e2m1_rn(u):
    """``cvt.rn.satfinite.e2m1`` of fp32 ``u``, decoded: nearest E2M1 value (ties to the even code), saturated at
    6, signed as the hardware code: a negative value that rounds to zero is -0.0, an exact -0.0 is +0.0."""
    a = tl.minimum(tl.abs(u), 6.0)
    # Midpoints between the E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}; at a tie the even code (0, 1, 2, 4) wins.
    v = tl.where(a <= 0.25, 0.0, tl.where(a <= 0.75, tl.where(a == 0.75, 1.0, 0.5),
        tl.where(a <= 1.25, 1.0, tl.where(a < 1.75, 1.5, tl.where(a <= 2.5, 2.0,
        tl.where(a < 3.5, 3.0, tl.where(a <= 5.0, 4.0, 6.0)))))))  # fmt: skip
    # The sign bit is set for u < 0 (an exact -0.0 encodes as +0, as cvt does). Set by OR: Triton's float negation
    # is 0 - x, which turns -0.0 into +0.0.
    sign = tl.where(u < 0.0, -2147483648, 0)
    return (v.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)


@triton.jit
def _kv_cache_round_trip_kernel(
    kv_ptr,  # [T, 576] bf16 pre-norm latent | raw k_pe (row stride kv_stride)
    kv_stride,
    norm_w_ptr,  # [512] kv_a_layernorm weight
    eps,
    pos_ptr,  # [T] int64
    cos_sin_ptr,  # vLLM's bf16 cos/sin table [max_pos, 64]
    cos_sin_stride,
    out_ptr,  # [T, 576] bf16, megatron's absorbed-key order
    out_stride,
    KV_DIM: tl.constexpr,
    HALF_ROT_DIM: tl.constexpr,
    NVFP4: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    tok_idx = tl.program_id(0).to(tl.int64)
    kv_block = tl.arange(0, KV_DIM)
    kv_c = tl.load(kv_ptr + tok_idx * kv_stride + kv_block)
    kv_c_rms_w = tl.load(norm_w_ptr + kv_block)
    kv_c = _rms_norm(kv_c, kv_c_rms_w, eps, KV_DIM)

    pos = tl.load(pos_ptr + tok_idx)
    cos, sin = _get_cos_sin(cos_sin_ptr, cos_sin_stride, pos, HALF_ROT_DIM)
    dim_off = tl.arange(0, HALF_ROT_DIM)
    kpe_base = kv_ptr + tok_idx * kv_stride + KV_DIM
    x1 = tl.load(kpe_base + dim_off * 2).to(tl.float32)
    x2 = tl.load(kpe_base + dim_off * 2 + 1).to(tl.float32)
    r1 = x1 * cos - x2 * sin
    r2 = x2 * cos + x1 * sin

    out_row = out_ptr + tok_idx * out_stride
    if NVFP4:
        kv_2d = tl.reshape(kv_c.to(tl.float32), (KV_DIM // 16, 16))
        amax = tl.max(tl.abs(kv_2d), axis=1, keep_dims=True)
        sf8 = tl.maximum(amax * (1.0 / 6.0), 0.001953125).to(tl.float8e4nv)
        q = kv_2d * (1.0 / sf8.to(tl.float32))
        nope = _signed_e2m1_rn(q) * sf8.to(tl.float32)
        rope1 = r1.to(tl.float8e4nv).to(tl.float32)
        rope2 = r2.to(tl.float8e4nv).to(tl.float32)
    else:
        kv_2d = tl.reshape(kv_c, (KV_DIM // 128, 128))
        tile_amax = tl.max(tl.abs(kv_2d), axis=1, keep_dims=True)
        tile_scale = tl.maximum(tile_amax * (1.0 / FP8_MAX), 1e-4)
        tile_scale = tl.math.exp2(tl.math.ceil(tl.math.log2(tile_scale)))
        nope = (kv_2d / tile_scale).to(tl.float8e4nv).to(tl.float32) * tile_scale
        rope1 = r1.to(tl.bfloat16).to(tl.float32)
        rope2 = r2.to(tl.bfloat16).to(tl.float32)
    tl.store(out_row + kv_block, tl.reshape(nope, (KV_DIM,)).to(tl.bfloat16))
    tl.store(out_row + KV_DIM + dim_off, rope1.to(tl.bfloat16))
    tl.store(out_row + KV_DIM + HALF_ROT_DIM + dim_off, rope2.to(tl.bfloat16))


def kv_cache_round_trip(
    kv_combined: torch.Tensor,
    positions: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor,
    nvfp4: bool,
) -> torch.Tensor:
    """``[T, 576]`` bf16 dequantized cache rows (``nvfp4_ds_mla`` if ``nvfp4`` else ``fp8_ds_mla``) of pre-norm
    ``kv_combined [T, 576]`` (unit column stride) at ``positions [T]``, in megatron's absorbed-key column order."""
    rows = kv_combined.size(0)
    assert kv_combined.size(1) == _KV_DIM + _ROPE_DIM and kv_combined.stride(1) == 1
    out = torch.empty((rows, _KV_DIM + _ROPE_DIM), dtype=torch.bfloat16, device=kv_combined.device)
    if rows:
        with torch.cuda.device(kv_combined.device):
            _kv_cache_round_trip_kernel[(rows,)](
                kv_combined, kv_combined.stride(0), norm_weight, eps, positions, cos_sin_cache,
                cos_sin_cache.stride(0), out, out.stride(0),
                KV_DIM=_KV_DIM, HALF_ROT_DIM=_ROPE_DIM // 2, NVFP4=nvfp4, FP8_MAX=_FP8_MAX,
            )  # fmt: skip
    return out
