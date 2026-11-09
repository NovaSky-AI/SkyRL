"""NVFP4 quantization of routed-expert weights on vLLM's grid, through FlashInfer's ``fp4_quantize``.

One fp32 global scale per expert projection, computed as vLLM's online quantizer does (``6 * 448 / amax``; gate and
up share one), E2M1 values with an E4M3 scale per 16 along K, round-to-nearest, and absmax or 4/6 block scales.
absmax equals vLLM's own ``scaled_fp4_quant`` byte for byte. ``expert_nvfp4``'s ModelOpt weight sync and the
trainer's round trips both call :func:`quantize_experts`, so vLLM serves exactly the grid the trainer rounds to.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator

import torch

NVFP4_GROUP_SIZE = 16
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)

# ``trainer.policy.megatron_config.expert_nvfp4_block_scale`` values.
BLOCK_SCALE_ABSMAX = "absmax"
BLOCK_SCALE_FOUR_OVER_SIX = "four_over_six"
BLOCK_SCALES = (BLOCK_SCALE_ABSMAX, BLOCK_SCALE_FOUR_OVER_SIX)


def check_block_scale(block_scale: str) -> str:
    if block_scale not in BLOCK_SCALES:
        raise ValueError(f"NVFP4 block_scale must be one of {list(BLOCK_SCALES)}, got {block_scale!r}")
    return block_scale


def flashinfer_nvfp4_env(four_over_six: bool) -> dict[str, str]:
    """Every env var FlashInfer's NVFP4 quantizers read (on each launch), set explicitly: 4/6 block scales
    (amax/6 or amax/4 per block, whichever has the lower squared error, on FlashInfer's fast error path) or
    absmax, the E4M3 bound 448, and FlashInfer's default fast scale math. FlashInfer treats only ``"1"`` as
    true; setting all of them keeps a stray value from the shell or the job env (e.g.
    ``*_DISABLE_FP4_QUANT_FAST_MATH=1``) from changing the grid. Exported to the vLLM engines for their per-token
    activations, and set around each trainer-side quantization by :func:`flashinfer_block_scale_env`."""
    return {
        "FLASHINFER_NVFP4_4OVER6": "1" if four_over_six else "0",
        "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
        "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
        "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "0",
        "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "0",
        "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": "0",
    }


@contextlib.contextmanager
def flashinfer_block_scale_env(four_over_six: bool) -> Iterator[None]:
    """Set FlashInfer's NVFP4 env (:func:`flashinfer_nvfp4_env`) for the enclosed launches, then restore it."""
    env = flashinfer_nvfp4_env(four_over_six)
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def nvfp4_amax(*weights: torch.Tensor) -> torch.Tensor:
    """fp32 ``max|W|`` over all ``weights`` jointly, as vLLM's ``weight_amax``."""
    amax = None
    for w in weights:
        lo, hi = w.aminmax()
        a = torch.maximum(lo.abs(), hi.abs()).to(torch.float32)
        amax = a if amax is None else torch.maximum(amax, a)
    assert amax is not None, "nvfp4_amax needs at least one tensor"
    return amax.clamp_min(1e-8)


def nvfp4_global_scale(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(global_scale, weight_scale_2)`` for an fp32 ``amax``: vLLM's encode scale and its decode scale."""
    global_scale = (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX) / amax
    weight_scale_2 = (1.0 / global_scale).to(torch.float32)
    return global_scale, weight_scale_2


def quantize_experts(w: torch.Tensor, gs: torch.Tensor, block_scale: str) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer ``fp4_quantize`` of CUDA ``w (E, N, K)`` with per-expert fp32 global scales ``gs (E,)``,
    in vLLM's online layout: ``(uint8 (E, N, K/2), float8_e4m3fn (E, N, K/16))``, scales unswizzled.

    One launch for all experts, with the global scale repeated per row (bit-identical to one launch per
    expert)."""
    import flashinfer

    E, N, K = w.shape
    with flashinfer_block_scale_env(check_block_scale(block_scale) == BLOCK_SCALE_FOUR_OVER_SIX):
        q, sf = flashinfer.fp4_quantize(
            w.contiguous().view(E * N, K),
            gs.to(torch.float32).repeat_interleave(N).contiguous(),
            sf_vec_size=NVFP4_GROUP_SIZE,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )
    q = q.view(torch.uint8).view(E, N, K // 2)
    sf = sf.view(torch.uint8).reshape(E, N, K // NVFP4_GROUP_SIZE)
    # A block whose E4M3 scale underflows to 0 dequantizes to 0 either way, but FlashInfer saturates its
    # values (+-6) where vLLM's scaled_fp4_quant writes +-0; keep only the sign bits so the bytes match.
    keep = torch.where(sf == 0, 0x88, 0xFF).to(torch.uint8)
    q.view(E, N, K // NVFP4_GROUP_SIZE, 8).bitwise_and_(keep.unsqueeze(-1))
    return q, sf.view(torch.float8_e4m3fn)
