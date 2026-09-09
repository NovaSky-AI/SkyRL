"""Restrict FlashAttention 4 to the architectures whose kernels actually run.

TE 2.16.0 gates FA4 on compute capability with a single lower bound::

    # FA4 supports SM80, SM90, SM100, SM120
    if device_compute_capability < (8, 0):
        use_flash_attention_4 = False

The comment describes an allowlist, but the check only excludes anything below
sm80, so every sm8x part -- sm86 (A10/A40), sm87, sm89 (L4, L40S, RTX 4090) --
passes it and TE selects FA4. FA4 then dispatches on ``arch // 10``, so those
land in its ``== 8`` branch, where the CuTe JIT produces a kernel the device
cannot launch::

    cutlass.cutlass_dsl.tvm_ffi_provider.CUDADialectError:
    error: cudaErrorInvalidValue (error code: 1)
    - Target SM ARCH: unknown (unspecified)
    - Architecture: Ada (sm_89)

This surfaced on the L4 GPU CI runners as soon as SkyRL began shipping FA4 (it
is bundled into the combined FA2+FA4 `flash-attn` wheel, so it is present on
every Linux x86_64/aarch64 node, not just Hopper/Blackwell ones).

Rather than recompile TE's ~1000-line ``get_attention_backend`` the way the FA2
head_dim patch has to, this flips ``FlashAttentionUtils.v4_is_installed``, which
every FA4 branch in that function already consults -- including the final
``if use_flash_attention_4 and not FlashAttentionUtils.v4_is_installed`` -- so
TE falls back to FA2 or cuDNN fused attention through its own logic.

The allowlist below is the set of architectures FA4 4.0.0b28 ships dedicated
kernels for (``flash_fwd_sm90``/``sm100``/``sm120``) and that SkyRL has run on.
sm80 (A100) is deliberately excluded as *untested*, not known-broken: FA4 was
not installed at all before the combined wheel, so nothing regresses by leaving
it off. Add it here once someone verifies FA4 on an A100.

DELETE THIS PATCH once the transformer-engine pin moves to a release whose FA4
gate matches FA4's real architecture support.
"""

import torch
from loguru import logger

# Compute capabilities with dedicated FA4 kernels that SkyRL has exercised.
_FA4_SUPPORTED_COMPUTE_CAPABILITIES = ((9, 0), (10, 0), (12, 0))


def patch_fa4_sm_arch() -> bool:
    """Disable FA4 in TE on architectures where its kernels cannot launch.

    Returns True if FA4 was disabled, False if it was left alone (already
    unavailable, supported architecture, or no TE/CUDA to inspect).
    """
    try:
        from transformer_engine.pytorch.attention.dot_product_attention import (
            dot_product_attention as dpa,
        )
        from transformer_engine.pytorch.attention.dot_product_attention import (
            utils as dpa_utils,
        )
    except ImportError:
        return False

    fa_utils = getattr(dpa_utils, "FlashAttentionUtils", None)
    if fa_utils is None or not getattr(fa_utils, "v4_is_installed", False):
        return False

    if not torch.cuda.is_available():
        return False

    capability = torch.cuda.get_device_capability()
    if capability in _FA4_SUPPORTED_COMPUTE_CAPABILITIES:
        return False

    fa_utils.v4_is_installed = False

    # Backend selection is memoized per attention config; drop any entry that
    # was chosen while FA4 still looked available.
    backends = getattr(dpa, "_attention_backends", None)
    if isinstance(backends, dict):
        backends["attention_params"] = None
        backends["backend_selection_requires_update"] = True

    logger.info(
        "Disabled FlashAttention 4 in TransformerEngine for sm{}{}: FA4 ships kernels only for {}. "
        "Falling back to FA2 / cuDNN fused attention.".format(
            capability[0],
            capability[1],
            ", ".join(f"sm{major}{minor}" for major, minor in _FA4_SUPPORTED_COMPUTE_CAPABILITIES),
        )
    )
    return True
