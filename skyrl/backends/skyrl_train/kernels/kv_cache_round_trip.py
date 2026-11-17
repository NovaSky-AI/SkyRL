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

Trainer-side variants (``kv_cache_qat_rounding``; the rollouts' cache stays as vLLM writes it):

- ``four_over_six`` (``nvfp4_ds_mla``): per 16-block the E4M3 scale ``amax / 6`` or ``amax / 4`` (both floored at
  2^-9), whichever round-to-nearest reconstructs with the lower squared error (amax/4 on a strict ``<``);
- ``stochastic``: every rounded value (E2M1 latent and E4M3 RoPE for ``nvfp4_ds_mla``, the E4M3 latent for
  ``fp8_ds_mla``) goes to one of its two neighbours, up with probability equal to its distance from the lower one, on
  the round-to-nearest scales (one E4M3 step larger where an NVFP4 block's largest value would saturate). On sm_100 /
  sm_103 the hardware rounds (``cvt.rs.satfinite.{e2m1,e4m3}x4.f32``, 8 random bits per value), elsewhere a uniform
  per value does. The random bits are Philox, seeded per call by the caller (from the CUDA generator).
"""

from __future__ import annotations

import contextlib
import os

import torch
import triton
import triton.language as tl
from vllm.models.deepseek_v32.common.kernels import _FP8_MAX, _get_cos_sin, _rms_norm

_KV_DIM = 512
_ROPE_DIM = 64
_ROW_DIM = _KV_DIM + _ROPE_DIM
# Devices whose PTX target has the stochastic cvt.rs conversions (arch-specific sm_100a / sm_103a).
_HW_SR_CAPABILITIES = ((10, 0), (10, 3))
_E4M3_MAX_BYTE = tl.constexpr(0x7E)


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
def _e2m1_sr_soft(u, r):
    """Stochastic E2M1 of fp32 ``u`` with uniforms ``r``, decoded and signed: up with probability (|u| - lo) / step."""
    a = tl.minimum(tl.abs(u), 6.0)
    step = tl.where(a < 2.0, 0.5, tl.where(a < 4.0, 1.0, 2.0))
    lo = tl.floor(a / step) * step
    v = tl.minimum(tl.where(r < (a - lo) / step, lo + step, lo), 6.0)
    sign = tl.where(u < 0.0, -2147483648, 0)
    return (v.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)


@triton.jit
def _e4m3_sr_soft(u, r):
    """Stochastic E4M3 (satfinite at 448) of fp32 ``u`` with uniforms ``r``, decoded and signed."""
    a = tl.minimum(tl.abs(u), 448.0)
    e = tl.maximum(((a.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127, -6)
    step = ((e + 124) << 23).to(tl.float32, bitcast=True)  # 2^(e - 3): 3 mantissa bits, subnormals below 2^-6
    lo = tl.floor(a / step) * step
    v = tl.minimum(tl.where(r < (a - lo) / step, lo + step, lo), 448.0)
    sign = tl.where(u < 0.0, -2147483648, 0)
    return (v.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)


# cvt.rs.satfinite.<fmt>x4.f32 of elements 0..3 ($4..$7; the instruction packs its first operand highest, so element 0
# lands in the lowest field) with the random bits $8, decoded back to fp32 through f16 (exact) into $0..$3.
_SR_E2M1_ASM = tl.constexpr(
    """{
.reg .b16 c, l0, l1, l2, l3;
.reg .b8 b0, b1;
.reg .b32 h0, h1;
cvt.rs.satfinite.e2m1x4.f32 c, {$7, $6, $5, $4}, $8;
mov.b16 {b0, b1}, c;
cvt.rn.f16x2.e2m1x2 h0, b0;
cvt.rn.f16x2.e2m1x2 h1, b1;
mov.b32 {l0, l1}, h0;
mov.b32 {l2, l3}, h1;
cvt.f32.f16 $0, l0;
cvt.f32.f16 $1, l1;
cvt.f32.f16 $2, l2;
cvt.f32.f16 $3, l3;
}"""
)
_SR_E4M3_ASM = tl.constexpr(
    """{
.reg .b32 w, h0, h1;
.reg .b16 s0, s1, l0, l1, l2, l3;
cvt.rs.satfinite.e4m3x4.f32 w, {$7, $6, $5, $4}, $8;
mov.b32 {s0, s1}, w;
cvt.rn.f16x2.e4m3x2 h0, s0;
cvt.rn.f16x2.e4m3x2 h1, s1;
mov.b32 {l0, l1}, h0;
mov.b32 {l2, l3}, h1;
cvt.f32.f16 $0, l0;
cvt.f32.f16 $1, l1;
cvt.f32.f16 $2, l2;
cvt.f32.f16 $3, l3;
}"""
)


@triton.jit
def _sr_hw(u, bits, N4: tl.constexpr, E2M1: tl.constexpr):
    """Hardware stochastic rounding of fp32 ``u`` (``4 * N4`` values, any shape) with one int32 of random bits per 4
    consecutive values (``bits [N4]``), decoded; E2M1 (saturating at 6) or E4M3 (at 448)."""
    pairs = tl.reshape(u, (N4, 2, 2))
    even, odd = tl.split(pairs)  # even[n, i] = element 4n + 2i, odd[n, i] = 4n + 2i + 1
    e0, e2 = tl.split(even)
    e1, e3 = tl.split(odd)
    o0, o1, o2, o3 = tl.inline_asm_elementwise(
        _SR_E2M1_ASM if E2M1 else _SR_E4M3_ASM,
        "=f,=f,=f,=f,f,f,f,f,r",
        [e0, e1, e2, e3, bits],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )
    return tl.reshape(tl.join(tl.join(o0, o2), tl.join(o1, o3)), u.shape)


@triton.jit
def _sr(u, seed, offs, N4: tl.constexpr, E2M1: tl.constexpr, HW_SR: tl.constexpr):
    """Stochastic E2M1 / E4M3 of fp32 ``u`` (``4 * N4`` values), decoded. ``offs``: the Philox counters, one per value
    (software) or per 4 consecutive values (hardware; ``N4`` of them)."""
    # Explicit branches: Triton compiles what follows a constexpr-guarded return.
    if HW_SR:
        out = _sr_hw(u, tl.randint(seed, offs).to(tl.int32, bitcast=True), N4, E2M1)
    else:
        r = tl.reshape(tl.rand(seed, offs), u.shape)
        if E2M1:
            out = _e2m1_sr_soft(u, r)
        else:
            out = _e4m3_sr_soft(u, r)
    return out


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
    seed_ptr,  # [1] int64 Philox seed (STOCHASTIC)
    KV_DIM: tl.constexpr,
    HALF_ROT_DIM: tl.constexpr,
    NVFP4: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FOUR_OVER_SIX: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    HW_SR: tl.constexpr,
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
    if STOCHASTIC:
        seed = tl.load(seed_ptr)
        # Philox counters, unique per row: values (software), or groups of 4 values (hardware).
        base = tok_idx.to(tl.int32) * (KV_DIM + 2 * HALF_ROT_DIM)
        nope_offs = base + tl.arange(0, KV_DIM // 4 if HW_SR else KV_DIM)
        rope_offs = base + KV_DIM + tl.arange(0, HALF_ROT_DIM // 4 if HW_SR else HALF_ROT_DIM)
    if NVFP4:
        kv_2d = tl.reshape(kv_c.to(tl.float32), (KV_DIM // 16, 16))
        amax = tl.max(tl.abs(kv_2d), axis=1, keep_dims=True)
        sf8 = tl.maximum(amax * (1.0 / 6.0), 0.001953125).to(tl.float8e4nv)
        if FOUR_OVER_SIX:
            sf4 = tl.maximum(amax * 0.25, 0.001953125).to(tl.float8e4nv)
            d6 = _signed_e2m1_rn(kv_2d * (1.0 / sf8.to(tl.float32))) * sf8.to(tl.float32) - kv_2d
            d4 = _signed_e2m1_rn(kv_2d * (1.0 / sf4.to(tl.float32))) * sf4.to(tl.float32) - kv_2d
            pick4 = tl.sum(d4 * d4, axis=1, keep_dims=True) < tl.sum(d6 * d6, axis=1, keep_dims=True)
            sf8 = tl.where(pick4, sf4, sf8)
        if STOCHASTIC:
            byte = sf8.to(tl.uint8, bitcast=True).to(tl.int32)
            clips = (amax * (1.0 / sf8.to(tl.float32)) > 6.0) & (byte < _E4M3_MAX_BYTE)
            sf8 = tl.where(clips, byte + 1, byte).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
            q = kv_2d * (1.0 / sf8.to(tl.float32))
            nope = _sr(q, seed, nope_offs, KV_DIM // 4, True, HW_SR) * sf8.to(tl.float32)
            rope1 = _sr(r1, seed, rope_offs, HALF_ROT_DIM // 4, False, HW_SR)
            rope2 = _sr(r2, seed + 1, rope_offs, HALF_ROT_DIM // 4, False, HW_SR)
        else:
            q = kv_2d * (1.0 / sf8.to(tl.float32))
            nope = _signed_e2m1_rn(q) * sf8.to(tl.float32)
            rope1 = r1.to(tl.float8e4nv).to(tl.float32)
            rope2 = r2.to(tl.float8e4nv).to(tl.float32)
    else:
        kv_2d = tl.reshape(kv_c, (KV_DIM // 128, 128))
        tile_amax = tl.max(tl.abs(kv_2d), axis=1, keep_dims=True)
        tile_scale = tl.maximum(tile_amax * (1.0 / FP8_MAX), 1e-4)
        tile_scale = tl.math.exp2(tl.math.ceil(tl.math.log2(tile_scale)))
        if STOCHASTIC:
            nope = _sr(kv_2d / tile_scale, seed, nope_offs, KV_DIM // 4, False, HW_SR) * tile_scale
        else:
            nope = (kv_2d / tile_scale).to(tl.float8e4nv).to(tl.float32) * tile_scale
        rope1 = r1.to(tl.bfloat16).to(tl.float32)
        rope2 = r2.to(tl.bfloat16).to(tl.float32)
    tl.store(out_row + kv_block, tl.reshape(nope, (KV_DIM,)).to(tl.bfloat16))
    tl.store(out_row + KV_DIM + dim_off, rope1.to(tl.bfloat16))
    tl.store(out_row + KV_DIM + HALF_ROT_DIM + dim_off, rope2.to(tl.bfloat16))


def hardware_sr_available(device: torch.device) -> bool:
    """Whether ``device``'s Triton target has the stochastic ``cvt.rs`` conversions (not in the interpreter)."""
    return (
        os.environ.get("TRITON_INTERPRET", "0") != "1"
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device) in _HW_SR_CAPABILITIES
    )


def kv_cache_round_trip(
    kv_combined: torch.Tensor,
    positions: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor,
    nvfp4: bool,
    four_over_six: bool = False,
    seed: torch.Tensor | None = None,
    hw_sr: bool | None = None,
) -> torch.Tensor:
    """``[T, 576]`` bf16 dequantized cache rows (``nvfp4_ds_mla`` if ``nvfp4`` else ``fp8_ds_mla``) of pre-norm
    ``kv_combined [T, 576]`` (unit column stride) at ``positions [T]``, in megatron's absorbed-key column order.
    ``four_over_six`` (``nvfp4`` only) and ``seed`` (an int64 ``[1]`` device tensor: round stochastically) select the
    trainer-side variants (module docstring); ``hw_sr`` forces the hardware / software stochastic rounding (default:
    hardware where available)."""
    rows = kv_combined.size(0)
    assert kv_combined.size(1) == _ROW_DIM and kv_combined.stride(1) == 1
    if four_over_six and not nvfp4:
        raise ValueError("4/6 block scales apply to nvfp4_ds_mla only")
    device = kv_combined.device
    if seed is not None and (seed.shape != (1,) or seed.dtype != torch.int64 or seed.device != device):
        raise ValueError(f"seed must be an int64 [1] tensor on {device}, got {seed.dtype} {tuple(seed.shape)}")
    hw = hardware_sr_available(device) if hw_sr is None else hw_sr
    out = torch.empty((rows, _ROW_DIM), dtype=torch.bfloat16, device=device)
    if rows:
        # CPU only under the Triton interpreter (tests).
        with torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext():
            _kv_cache_round_trip_kernel[(rows,)](
                kv_combined, kv_combined.stride(0), norm_weight, eps, positions, cos_sin_cache,
                cos_sin_cache.stride(0), out, out.stride(0), seed if seed is not None else out,
                KV_DIM=_KV_DIM, HALF_ROT_DIM=_ROPE_DIM // 2, NVFP4=nvfp4, FP8_MAX=_FP8_MAX,
                FOUR_OVER_SIX=four_over_six, STOCHASTIC=seed is not None, HW_SR=hw and seed is not None,
            )  # fmt: skip
    return out
