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
        x_2d = x.reshape(s * b, nC).to(torch.float32)
        weight = self.mapping_proj.weight.to(torch.float32)
        proj = torch.matmul(x_2d, weight.t())
        r = torch.rsqrt(x_2d.square().mean(dim=-1, keepdim=True) + self.norm_eps)
        return proj.view(s, b, -1), r.view(s, b, 1)
