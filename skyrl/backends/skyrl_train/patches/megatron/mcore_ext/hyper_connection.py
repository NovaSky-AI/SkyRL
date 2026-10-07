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
from torch.utils.checkpoint import checkpoint


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


def _proj_rms_from_input_dtype(x: Tensor, weight: Tensor, eps: float, eps_inside_sqrt: bool) -> Tuple[Tensor, Tensor]:
    return _proj_rms_fp32(x.to(torch.float32), weight.to(torch.float32), eps, eps_inside_sqrt)


def proj_rms_saving_input_dtype(
    x: Tensor, weight: Tensor, eps: float = 1e-6, eps_inside_sqrt: bool = False
) -> Tuple[Tensor, Tensor]:
    """``native_proj_rms(x.float(), weight.float(), eps, eps_inside_sqrt)`` without keeping ``x.float()``.

    Same signature as #7521's ``native_proj_rms`` (its ``_proj_rms_op``), but takes ``x`` in the
    activation dtype. Plain autograd keeps the ``[tokens, n * hidden]`` FP32 upcast of every mHC
    site's input for backward (2 GiB per site at 32k tokens per rank for GLM-5.3-Flash).
    Checkpointing the upcast together with the math saves only ``x`` itself -- a view of the
    residual stream, alive anyway -- and reruns both in backward, so outputs and gradients are
    bitwise those of the plain version, for one extra (n^2 + 2n)-column projection per site.
    """
    return checkpoint(
        _proj_rms_from_input_dtype, x, weight, eps, eps_inside_sqrt, use_reentrant=False, preserve_rng_state=False
    )
