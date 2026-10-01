"""Standard-RMSNorm input normalization for megatron-core's mHC module.

``HyperConnectionModule`` normalizes the flattened residual streams as ``x / (rms(x) + eps)``
with ``eps`` hard-coded to 1e-6. GLM-5.3-Flash instead uses a standard RMSNorm,
``x * rsqrt(mean(x^2) + rms_norm_eps)``. The two agree for O(1) activations but not for small
residual streams -- this model's embeddings have a per-token rms below ``sqrt(1e-5)``, where the
placement of the epsilon changes the mixing weights materially.

DELETE THIS MODULE once ``TransformerConfig`` carries the input-norm knobs upstream
(``mhc_norm_eps`` / ``mhc_norm_eps_inside_sqrt``, read by ``HyperConnectionModule`` itself).
"""

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
        proj, r = _ProjectionAndRMSNorm.apply(x.reshape(s * b, nC), self.mapping_proj.weight, self.norm_eps)
        return proj.view(s, b, -1), r.view(s, b, 1)


class _ProjectionAndRMSNorm(torch.autograd.Function):
    """``proj = x32 @ w32^T`` and ``r = rsqrt(mean(x32^2) + eps)``, with ``x32 = x.float()``.

    Same FP32 math as the plain autograd version, but saves the activation-dtype ``x`` (a view of
    the residual stream, which is alive anyway) instead of its FP32 upcast, and redoes the upcast
    in backward. Autograd would keep a ``[tokens, n * hidden]`` FP32 copy per mHC site for the
    layer's lifetime: 2 GiB per site at 32k tokens per rank for GLM-5.3-Flash.
    """

    @staticmethod
    def forward(ctx, x_2d: Tensor, weight: Tensor, eps: float) -> Tuple[Tensor, Tensor]:
        x32 = x_2d.to(torch.float32)
        w32 = weight.to(torch.float32)
        proj = torch.matmul(x32, w32.t())
        r = torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps)
        ctx.save_for_backward(x_2d, weight, r)
        return proj, r

    @staticmethod
    def backward(ctx, grad_proj: Tensor, grad_r: Tensor):
        x_2d, weight, r = ctx.saved_tensors
        x32 = x_2d.to(torch.float32)
        w32 = weight.to(torch.float32)
        grad_x = grad_w = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.zeros_like(x32)
            if grad_proj is not None:
                grad_x = torch.matmul(grad_proj, w32)
            if grad_r is not None:
                # r = (m + eps)^(-1/2), m = mean(x^2)  =>  dr/dx = -r^3 * x / K
                grad_x = grad_x - (grad_r * r.pow(3) / x32.shape[-1]) * x32
            grad_x = grad_x.to(x_2d.dtype)
        if ctx.needs_input_grad[1] and grad_proj is not None:
            grad_w = torch.matmul(grad_proj.t(), x32).to(weight.dtype)
        return grad_x, grad_w, None
