"""Stochastic rounding (SR) of NVFP4 values for expert_nvfp4's trainer-side rounding (the rollouts round to nearest).

SR keeps each 16-block's E4M3 scale from the round-to-nearest quantizer (absmax or its 4/6 choice), raised by one E4M3
step where the block's largest value would otherwise exceed 6 (E2M1 saturates there, which would bias the block down),
and rounds every scaled value ``v`` to one of its two E2M1 neighbours ``lo <= |v| <= hi``, up with probability
``(|v| - lo) / (hi - lo)``: unbiased, ``E[q] = v``. Only the value rounding is random; scales and the decode are the
round-to-nearest grid's.

Draws:

- weights (:class:`WeightDraw`): one per weight version (optimizer step) and expert, so every fprop, dgrad and
  recompute of a version, and every rank holding the expert (expert data parallelism), rounds it alike;
- activations: a fresh draw per call from the current CUDA generator, whose state Megatron's activation checkpointing
  saves and restores, so a recompute redraws exactly the forward's rounding.

This module is the torch reference: FlashInfer's round-to-nearest bytes, then the scaled values re-rounded with
``torch.rand`` uniforms. The fused sm_100 kernels round with the hardware's ``cvt.rs`` instead (8 random bits per
value, so probabilities are multiples of 1/256); the two agree in distribution, not bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0
# Largest finite E4M3 byte (448); a scale at it cannot be raised.
E4M3_MAX_BYTE = 0x7E
_MASK64 = (1 << 64) - 1


def splitmix64(x: int) -> int:
    """SplitMix64 finalizer of a 64-bit integer (a bijective mix, for deriving independent seeds)."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def weight_seed(base_seed: int, version: int) -> int:
    """The 64-bit seed of weight version ``version`` (optimizer steps since start) under ``base_seed``."""
    return splitmix64(splitmix64(base_seed & _MASK64) ^ (version & _MASK64))


def expert_stream(layer_number: int, fc2: bool, expert: int) -> int:
    """The 64-bit stream id of one routed expert projection: global layer number, fc1 / fc2, global expert index."""
    return splitmix64((layer_number << 24) | (int(fc2) << 23) | expert)


@dataclass(frozen=True)
class WeightDraw:
    """The SR draw of one grouped GEMM's expert weights: ``seed`` (:func:`weight_seed`) and one
    :func:`expert_stream` per expert, in expert order."""

    seed: int
    streams: tuple[int, ...]

    def in_memory_order(self, reversed_: bool) -> tuple[int, ...]:
        """``streams`` in the order of an ``ExpertStack``'s ``data`` (``reversed_``: highest expert first)."""
        return self.streams[::-1] if reversed_ else self.streams


def expert_uniforms(shape: tuple[int, int, int], draw: WeightDraw, reversed_: bool, device: torch.device):
    """fp32 uniforms ``[0, 1)`` for ``(E, N, K)`` expert weights in memory order: expert ``i``'s from a generator
    seeded by ``draw.seed ^ stream``, so each expert's draw depends only on the version and its stream."""
    e, n, k = shape
    streams = draw.in_memory_order(reversed_)
    if len(streams) != e:
        raise ValueError(f"WeightDraw has {len(streams)} streams for {e} experts")
    u = torch.empty(shape, dtype=torch.float32, device=device)
    gen = torch.Generator(device=device)
    for i, s in enumerate(streams):
        gen.manual_seed((draw.seed ^ s) & ((1 << 63) - 1))
        torch.rand((n, k), generator=gen, device=device, out=u[i])
    return u


def _grid(device: torch.device) -> torch.Tensor:
    return torch.tensor(E2M1_GRID, dtype=torch.float32, device=device)


def stochastic_e2m1(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """E2M1 codes (``uint8``, sign in bit 3) of fp32 ``v`` rounded stochastically with uniforms ``u``; magnitudes
    above 6 saturate."""
    grid = _grid(v.device)
    a = v.abs().clamp(max=E2M1_MAX)
    lo_idx = (torch.bucketize(a, grid, right=True) - 1).clamp(0, len(E2M1_GRID) - 1)
    hi_idx = (lo_idx + 1).clamp(max=len(E2M1_GRID) - 1)
    lo, hi = grid[lo_idx], grid[hi_idx]
    p = torch.where(hi > lo, (a - lo) / (hi - lo), torch.zeros_like(a))
    idx = lo_idx + (u < p).to(lo_idx.dtype)
    return (idx | torch.where(v < 0, 8, 0)).to(torch.uint8)


def decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """fp32 values of E2M1 codes (sign in bit 3)."""
    mag = _grid(codes.device)[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -mag, mag)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """``[..., K]`` codes -> ``[..., K/2]`` bytes, element ``2i`` in the low nibble."""
    c = codes.view(*codes.shape[:-1], -1, 2)
    return (c[..., 0] | (c[..., 1] << 4)).to(torch.uint8)


def e4m3_bytes_to_float(sf: torch.Tensor) -> torch.Tensor:
    return sf.view(torch.float8_e4m3fn).to(torch.float32)


def stochastic_requantize(
    x: torch.Tensor, enc: torch.Tensor, sf: torch.Tensor, u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-round NVFP4 rows stochastically on their round-to-nearest scales: ``x [M, K]`` (bf16 or fp32), the rows'
    encode scales ``enc [M]`` (``gs`` for weights, ``1 / t`` per token), their E4M3 block scales ``sf [M, K/16]``
    (uint8) and uniforms ``u [M, K]``. Returns ``(codes [M, K/2] uint8, sf [M, K/16] uint8)``, the scales raised one
    E4M3 step in blocks whose largest value would clip."""
    m, k = x.shape
    xb = x.to(torch.float32).view(m, k // 16, 16)
    sf = sf.view(torch.uint8).reshape(m, k // 16).clone()
    enc = enc.to(torch.float32).view(m, 1)

    def scaled(s: torch.Tensor) -> torch.Tensor:
        val = e4m3_bytes_to_float(s)
        os_ = torch.where(val > 0, enc / val.clamp_min(torch.finfo(torch.float32).tiny), torch.zeros_like(val))
        return xb * os_.unsqueeze(-1)

    clip = (scaled(sf).abs().amax(-1) > E2M1_MAX) & (sf > 0) & (sf < E4M3_MAX_BYTE)
    sf = torch.where(clip, sf + 1, sf)
    codes = stochastic_e2m1(scaled(sf), u.view(m, k // 16, 16))
    return pack_nibbles(codes.view(m, k)), sf
