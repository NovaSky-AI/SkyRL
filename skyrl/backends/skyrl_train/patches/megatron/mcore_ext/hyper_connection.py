"""Standard-RMSNorm input normalization for megatron-core's mHC module.

``HyperConnectionModule`` normalizes the flattened residual streams as ``x / (rms(x) + eps)``
with ``eps`` hard-coded to 1e-6. GLM-5.3-Flash instead uses a standard RMSNorm,
``x * rsqrt(mean(x^2) + rms_norm_eps)``. The two agree for O(1) activations but not for small
residual streams -- this model's embeddings have a per-token rms below ``sqrt(1e-5)``, where the
placement of the epsilon changes the mixing weights materially.

DELETE THIS MODULE once ``TransformerConfig`` carries the input-norm knobs upstream
(``mhc_norm_eps`` / ``mhc_norm_eps_inside_sqrt``, read by ``HyperConnectionModule`` itself).
"""

import math
from typing import Tuple

import torch
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor


class RMSNormInputHyperConnectionModule(HyperConnectionModule):
    """mHC module whose input normalization is a standard RMSNorm.

    Reads ``mhc_norm_eps`` from the config, falling back to ``layernorm_epsilon``.
    """

    def __init__(self, config: TransformerConfig, layer_number: int):
        super().__init__(config, layer_number)
        if config.use_fused_mhc:
            raise NotImplementedError(
                "The fused mHC kernels implement the 1/(rms+eps) input normalization only; "
                "use_fused_mhc is not compatible with mhc_norm_eps_inside_sqrt=True."
            )
        self.norm_eps = getattr(config, "mhc_norm_eps", None) or config.layernorm_epsilon

    def _projection_and_get_norm(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Projection + standard RMS normalization.

        Args:
            x: [s, b, n*C] - n-stream hidden states
        """
        s, b, nC = x.shape
        # The mHC mapping runs in FP32 (the parameters are kept in FP32 and the activations are
        # upcast here); compute_mappings casts the bounded mixing weights back down.
        proj, r = proj_rms_saving_input_dtype(
            x.reshape(s * b, nC), self.mapping_proj.weight, self.norm_eps, eps_inside_sqrt=True
        )
        return proj.view(s, b, -1), r.view(s, b, 1)


def _proj_rms_fp32(x: Tensor, weight: Tensor, eps: float, eps_inside_sqrt: bool) -> Tuple[Tensor, Tensor]:
    """NVIDIA/Megatron-LM#7521's ``native_proj_rms``: projection + RMS normalization, on FP32 inputs."""
    proj = torch.matmul(x, weight.t())
    if eps_inside_sqrt:
        return proj, torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    norm = x.norm(dim=-1, keepdim=True)
    K = x.shape[-1]
    v = norm / math.sqrt(K) + eps
    r = 1.0 / v
    return proj, r


def proj_rms_saving_input_dtype(
    x: Tensor, weight: Tensor, eps: float = 1e-6, eps_inside_sqrt: bool = False
) -> Tuple[Tensor, Tensor]:
    """``native_proj_rms(x.float(), weight.float(), eps, eps_inside_sqrt)`` without keeping ``x.float()``.

    Same signature as #7521's ``native_proj_rms`` (its ``_proj_rms_op``), but takes ``x`` in the
    activation dtype and upcasts inside. Plain autograd keeps the ``[tokens, n * hidden]`` FP32
    upcast of every mHC site's input for the layer's lifetime (2 GiB per site at 32k tokens per
    rank for GLM-5.3-Flash); this saves only ``x`` itself -- a view of the residual stream, alive
    anyway -- and redoes the FP32 math under autograd in backward, so outputs and gradients are
    bitwise those of the plain version, for one extra (n^2 + 2n)-column projection per site.
    """
    return _ProjectionAndRMSNorm.apply(x, weight, eps, eps_inside_sqrt)


class _ProjectionAndRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float, eps_inside_sqrt: bool) -> Tuple[Tensor, Tensor]:
        ctx.save_for_backward(x, weight)
        ctx.eps, ctx.eps_inside_sqrt = eps, eps_inside_sqrt
        ctx.set_materialize_grads(False)  # an unused output's grad stays None, as in plain autograd
        return _proj_rms_fp32(x.to(torch.float32), weight.to(torch.float32), eps, eps_inside_sqrt)

    @staticmethod
    def backward(ctx, grad_proj: Tensor, grad_r: Tensor):
        x, weight = ctx.saved_tensors
        need_x, need_w = ctx.needs_input_grad[:2]
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(need_x)
            w_in = weight.detach().requires_grad_(need_w)
            outs = _proj_rms_fp32(x_in.to(torch.float32), w_in.to(torch.float32), ctx.eps, ctx.eps_inside_sqrt)
            pairs = [(o, g) for o, g in zip(outs, (grad_proj, grad_r)) if g is not None]
            inputs = [t for t in (x_in, w_in) if t.requires_grad]
            grads = (
                torch.autograd.grad([o for o, _ in pairs], inputs, [g for _, g in pairs], allow_unused=True)
                if pairs and inputs
                else [None] * len(inputs)
            )
        grads = iter(grads)
        grad_x = next(grads) if need_x else None
        grad_w = next(grads) if need_w else None
        return grad_x, grad_w, None, None
