"""Fake-INT4 quantization-aware training for Megatron MoE experts.

When vLLM serves the experts as real ``compressed-tensors`` INT4 but the trainer
holds BF16 masters, the two disagree (a train/infer log-prob gap). This fake-
quantizes the frozen fused expert GEMMs (``TEGroupedLinear``) onto the same
group-symmetric INT4 grid inside the forward pass with a straight-through-
estimator backward, so gradients still reach the BF16 masters (LoRA adapters stay
BF16, matching "INT4 base + BF16 adapter" at inference).

The grid is computed with the *same arithmetic* the serving artifact was produced
with, so the fake-quantized weights are bit-exact to what the inference engine
dequantizes (verified element-for-element against real checkpoints):

    amax  = max|w| per group (exact)
    scale = rn_dtype(amax / scale_divisor)      # single fp32->bf16 rounding,
                                                # equals the stored ``weight_scale``
    q     = clamp(round(w / scale), q_min, 7)   # divide+round in the weight dtype,
                                                # matches compressed-tensors quantize()
    deq   = q * scale                           # bf16 multiply, matches dequantize()

All-zero groups quantize to zero (guarded division; no eps clamp -- an eps floor
would distort near-denormal groups that real checkpoints do contain).

Two conventions, selected by ``(scale_divisor, q_min)``:

- ``(7.5, -8)``: llm-compressor / compressed-tensors RTN. Verified bit-exact
  against ``Qwen3.6-35B-A3B-INT4-RTN`` (requires the *original* BF16 weights as
  masters; a dequantized INT4 checkpoint does NOT reproduce a /7.5 grid).
- ``(7.0, -7)``: Kimi K2-Thinking / K2.6 / Miles QAT. Verified bit-exact against
  ``moonshotai/Kimi-K2.6`` with masters dequantized from the INT4 release (the
  max-|w| element of every group codes to +-7, which makes the recomputed grid
  reproduce the stored one exactly).

Enabled and parameterised entirely by ``trainer.policy.model.fake_int4_qat``.
"""

from __future__ import annotations

import torch
from loguru import logger

# Symmetric signed-INT4 upper bound; shared by both conventions. The convention
# knobs (scale_divisor, q_min) come from ``trainer.policy.model.fake_int4_qat``.
_Q_MAX = 7.0


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class _FakeInt4QuantizeSTE(torch.autograd.Function):
    """Group-symmetric INT4 fake-quantize with a straight-through backward.

    The forward reproduces the compressed-tensors quantize->dequantize pipeline
    bit-exactly in the weight dtype (see module docstring); the backward is the
    identity, so gradients pass straight through to the BF16 master weight.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group_size: int, scale_div: float, q_min: float) -> torch.Tensor:  # noqa: D401
        out_features, in_features = x.shape

        # Pad the input dim up to a whole number of groups. Real MoE dims divide
        # evenly (2048 / 512 by 32), but stay defensive so odd shapes don't crash.
        n_padded = _ceil_div(in_features, group_size) * group_size
        if n_padded != in_features:
            x_p = x.new_zeros((out_features, n_padded))
            x_p[:, :in_features] = x
        else:
            x_p = x

        # reshape (not view): free for the always-contiguous TE weights, and
        # copy-on-noncontiguous keeps the public helper safe for sliced inputs.
        g = x_p.reshape(out_features, n_padded // group_size, group_size)
        # amax is exact in the weight dtype; the fp32 divide + cast back applies
        # exactly one rounding, matching compressed-tensors' calculate_qparams
        # (the stored ``weight_scale``). Grid arithmetic below stays in the
        # weight dtype so q and deq match quantize()/dequantize() bit-for-bit.
        amax = g.abs().amax(dim=-1, keepdim=True).to(torch.float32)
        scale = (amax / scale_div).to(x.dtype)
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        q = torch.clamp(torch.round(g / safe_scale), q_min, _Q_MAX)
        deq = (q * scale).reshape(out_features, n_padded)
        out = deq[:, :in_features].contiguous()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: identity gradient to the BF16 master weight.
        return grad_output, None, None, None


def fake_int4_quantize_ste(
    x: torch.Tensor,
    group_size: int,
    scale_div: float,
    q_min: float,
) -> torch.Tensor:
    """Apply the fake-INT4 STE to a 2D ``[out, in]`` weight, preserving Megatron's
    ``main_grad`` bookkeeping so the fused optimizer still finds its grad buffer.

    ``(scale_div, q_min)`` selects the convention: ``(7.5, -8)`` llm-compressor
    RTN, ``(7.0, -7)`` Kimi/Miles."""
    out = _FakeInt4QuantizeSTE.apply(x, group_size, scale_div, q_min)
    if hasattr(x, "main_grad"):
        out.main_grad = x.main_grad
    return out


_installed = False


def install_fake_int4_qat(
    group_size: int,
    scale_divisor: float,
    q_min: float,
) -> None:
    """Monkeypatch ``TEGroupedLinear._get_weight_tensors`` to fake-quantize the
    fused MoE expert weights. Call once per worker when
    ``trainer.policy.model.fake_int4_qat.enabled`` is set (the config also supplies
    ``group_size``, ``scale_divisor`` and ``q_min``)."""
    global _installed
    if _installed:
        return

    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    original = TEGroupedLinear._get_weight_tensors

    def _patched(self):
        return [
            (
                fake_int4_quantize_ste(w, group_size, scale_divisor, q_min)
                if isinstance(w, torch.Tensor) and w.dim() == 2
                else w
            )
            for w in original(self)
        ]

    TEGroupedLinear._get_weight_tensors = _patched
    _installed = True
    logger.info(
        f"fake-INT4 QAT: patched TEGroupedLinear MoE experts "
        f"(group_size={group_size}, scale_divisor={scale_divisor}, q_min={q_min}, STE backward)."
    )
