"""Runtime opt-out for FlashAttention 4 in Transformer Engine.

SkyRL ships FA4 inside the combined ``flash-attn`` wheel, but only installs the
metadata-only ``flash-attn-4`` companion under the ``fa4`` extra, so FA4 is off
unless a user opts in (see the ``fa4`` extra in ``pyproject.toml``). This module
is the second half of that switch: a way to turn FA4 back off *without*
rebuilding the environment.

It flips ``FlashAttentionUtils.v4_is_installed``, which every FA4 branch in TE's
``get_attention_backend`` already consults -- including the final
``if use_flash_attention_4 and not FlashAttentionUtils.v4_is_installed`` -- so TE
falls back to FA2 (or cuDNN fused attention) through its own selection logic
rather than anything bolted on here.

Two uses:

* A/B-ing FA2 against FA4 on identical environments, where re-resolving the venv
  between arms would confound the comparison.
* An escape hatch if an FA4 kernel misbehaves on shapes or an architecture SkyRL
  exercises -- e.g. TE 2.16's arch gate only excludes ``< sm80``, so the sm8x
  parts that are not sm80 proper (sm86/sm87/sm89: A10, L4, L40S, RTX 4090) pass
  it and then fail in the CuTe JIT with
  ``cudaErrorInvalidValue ... Target SM ARCH: unknown (unspecified)``.
"""

import torch
from loguru import logger

from skyrl.env_vars import SKYRL_DISABLE_FA4


def disable_fa4_if_requested() -> bool:
    """Disable FA4 in TE when ``SKYRL_DISABLE_FA4`` is set.

    Returns True if FA4 was disabled, False if it was left alone (not requested,
    FA4 not installed, or no TE/CUDA to inspect).
    """
    if not SKYRL_DISABLE_FA4:
        return False

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

    fa_utils.v4_is_installed = False

    # Backend selection is memoized per attention config; drop any entry that was
    # chosen while FA4 still looked available.
    backends = getattr(dpa, "_attention_backends", None)
    if isinstance(backends, dict):
        backends["attention_params"] = None
        backends["backend_selection_requires_update"] = True

    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    where = f" on sm{capability[0]}{capability[1]}" if capability else ""
    logger.info(
        f"SKYRL_DISABLE_FA4 is set: disabled FlashAttention 4 in TransformerEngine{where}. "
        "Falling back to FA2 / cuDNN fused attention."
    )
    return True
