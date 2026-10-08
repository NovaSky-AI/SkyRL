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
        # upcast in _proj_rms); compute_mappings casts the bounded mixing weights back down.
        # Checkpointed so backward keeps only the activation-dtype input, not its FP32 upcast
        # (2 GiB per mHC site at 32k tokens per rank); the upcast and the math rerun in backward.
        proj, r = checkpoint(
            _proj_rms,
            x.reshape(s * b, nC),
            self.mapping_proj.weight,
            self.norm_eps,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        return proj.view(s, b, -1), r.view(s, b, 1)


def _proj_rms(x: Tensor, weight: Tensor, eps: float) -> Tuple[Tensor, Tensor]:
    """FP32 projection and standard RMS factor ``rsqrt(mean(x^2) + eps)`` of the activation-dtype ``x``."""
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    proj = torch.matmul(x, weight.t())
    r = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return proj, r
