"""Fake-INT4 quantization-aware training for Megatron MoE experts.

When vLLM serves the experts as real ``compressed-tensors`` INT4 but the trainer
holds BF16 masters, the two disagree (a train/infer log-prob gap). This fake-
quantizes the frozen fused expert GEMMs (``TEGroupedLinear``) onto the same
group-symmetric INT4 grid inside the forward pass with a straight-through-
estimator backward, so gradients still reach the BF16 masters (LoRA adapters stay
BF16, matching "INT4 base + BF16 adapter" at inference). The grid is
``scale = amax(group) / scale_divisor``, ``q = clamp(round(w / scale), -8, 7)``,
grouped along the input dim: ``scale_divisor=7.5`` matches llm-compressor /
compressed-tensors RTN (verified bit-exact) and ``7.0`` matches the Kimi/Miles
scheme (values land in ``[-7, 7]``, so the ``-8`` clamp is a no-op). Enabled and
parameterised entirely by ``trainer.policy.model.fake_int4_qat``.
"""

from __future__ import annotations

import torch
from loguru import logger

# int4 signed range (clamp is a no-op for the [-7, 7] Kimi convention).
_Q_MIN = -8.0
_Q_MAX = 7.0
# scale_divisor: 7.5 = llm-compressor/compressed-tensors RTN; 7.0 = Kimi/Miles.
SCALE_DIV_LLMCOMPRESSOR = 7.5
SCALE_DIV_KIMI = 7.0


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class _FakeInt4QuantizeSTE(torch.autograd.Function):
    """Group-symmetric INT4 fake-quantize with a straight-through backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group_size: int, scale_div: float) -> torch.Tensor:  # noqa: D401
        out_features, in_features = x.shape

        # Pad the input dim up to a whole number of groups. Real MoE dims divide
        # evenly (2048 / 512 by 32), but stay defensive so odd shapes don't crash.
        n_padded = _ceil_div(in_features, group_size) * group_size
        if n_padded != in_features:
            x_p = x.new_zeros((out_features, n_padded))
            x_p[:, :in_features] = x
        else:
            x_p = x

        g = x_p.view(out_features, n_padded // group_size, group_size).to(torch.float32)
        amax = g.abs().amax(dim=-1, keepdim=True)
        scale = (amax / scale_div).clamp(min=torch.finfo(torch.float32).eps)

        q = torch.clamp(torch.round(g / scale), _Q_MIN, _Q_MAX)
        deq = (q * scale).view(out_features, n_padded)
        out = deq[:, :in_features].contiguous().to(x.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: identity gradient to the BF16 master weight.
        return grad_output, None, None


def fake_int4_quantize_ste(
    x: torch.Tensor, group_size: int, scale_div: float = SCALE_DIV_LLMCOMPRESSOR
) -> torch.Tensor:
    """Apply the fake-INT4 STE to a 2D ``[out, in]`` weight, preserving Megatron's
    ``main_grad`` bookkeeping so the fused optimizer still finds its grad buffer.

    ``scale_div`` selects the convention: ``7.5`` (llm-compressor RTN) or ``7.0``
    (Kimi/Miles)."""
    out = _FakeInt4QuantizeSTE.apply(x, group_size, scale_div)
    if hasattr(x, "main_grad"):
        out.main_grad = x.main_grad
    return out


_installed = False


def install_fake_int4_qat(group_size: int = 32, scale_divisor: float = SCALE_DIV_LLMCOMPRESSOR) -> None:
    """Monkeypatch ``TEGroupedLinear._get_weight_tensors`` to fake-quantize the
    fused MoE expert weights. Call once per worker when
    ``trainer.policy.model.fake_int4_qat.enabled`` is set (the config also supplies
    ``group_size`` and ``scale_divisor``)."""
    global _installed
    if _installed:
        return

    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    original = TEGroupedLinear._get_weight_tensors

    def _patched(self):
        return [
            fake_int4_quantize_ste(w, group_size, scale_divisor)
            if isinstance(w, torch.Tensor) and w.dim() == 2
            else w
            for w in original(self)
        ]

    TEGroupedLinear._get_weight_tensors = _patched
    _installed = True
    logger.info(
        f"fake-INT4 QAT: patched TEGroupedLinear MoE experts "
        f"(group_size={group_size}, scale_divisor={scale_divisor}, STE backward)."
    )
