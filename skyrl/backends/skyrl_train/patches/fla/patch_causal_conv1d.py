"""Use a safe static FLA causal-convolution launch on B300.

FLA autotunes causal_conv1d over configurations that include 32-warp launches.
On sm103, benchmarking that search at a 262,144-token sequence raises an
illegal-memory-access error before the model performs useful work.  Pinning the
same Triton kernels to a conservative launch avoids the faulty candidate; the
kernel math is unchanged.

Remove this patch after FLA's causal-convolution autotuner supports sm103.
"""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_B300_COMPUTE_CAPABILITY = (10, 3)
_PATCHED_FLAG = "_skyrl_b300_static_config"


def _set_static_config(kernel: Any, config: Any) -> bool:
    """Replace the config search on a Triton heuristic-wrapped autotuner."""
    autotuner = getattr(kernel, "fn", None)
    if autotuner is None or not hasattr(autotuner, "configs"):
        return False
    if getattr(autotuner, _PATCHED_FLAG, False):
        return True

    autotuner.configs = [config]
    setattr(autotuner, _PATCHED_FLAG, True)
    return True


def patch_b300_causal_conv1d_autotune() -> bool:
    """Pin FLA's forward and backward causal-convolution kernels on sm103."""
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != _B300_COMPUTE_CAPABILITY
    ):
        return False

    import triton
    from fla.modules.conv.triton import kernels

    config = triton.Config({"BD": 64}, num_warps=8)
    patched = all(
        _set_static_config(getattr(kernels, name, None), config)
        for name in ("causal_conv1d_fwd_kernel", "causal_conv1d_bwd_kernel")
    )
    if not patched:
        logger.warning(
            "FLA causal_conv1d kernels do not expose the expected autotuner; skipping B300 patch"
        )
        return False

    logger.info("Pinned FLA causal_conv1d Triton kernels to BD=64,num_warps=8 on sm103")
    return True
