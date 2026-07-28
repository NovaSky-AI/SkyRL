"""Spec-decode (MTP / Eagle) drafter weight-sync helper.

vLLM's weight reload (``GPUModelRunner.reload_weights`` and direct
``model_runner.model.load_weights``) only updates the **main** model. For models
with native MTP heads (Qwen3.5, DeepSeek-V3, GLM-4.x, ...) the speculative-decoding
**drafter is a SEPARATE module** (``model_runner.drafter.model``, e.g.
``Qwen3_5MTP``) that the main-model load never touches. In colocated training the
inference engine is also slept with ``level=2`` (which discards weights), so after
wake the drafter is left uninitialized/garbage and MTP speculative decoding drafts
with a broken head -> ~0 draft-acceptance from the first generate.

This helper re-loads the drafter from the same synced weights right after the main
model load. The drafter's ``load_weights`` filters to the names it consumes (e.g.
Qwen3.5: ``mtp.*`` plus ``embed_tokens``/``lm_head``) and ignores the rest, so the
full weight list is safe to pass. No-op when speculative decoding is disabled (no
drafter) or the drafter has no loadable model (e.g. ngram).
"""

import logging
from typing import Iterable, List, Tuple

import torch

logger = logging.getLogger(__name__)

# One-shot flags so the per-sync outcome is logged once, not every chunk of every sync.
_logged_reload = False
_logged_noop = False


def _reload_spec_decode_drafter(model_runner, weight_list: List[Tuple[str, torch.Tensor]]) -> bool:
    """Re-load the spec-decode drafter model from ``weight_list`` (HF-named tensors).

    Args:
        model_runner: the vLLM ``GPUModelRunner`` (``self.model_runner`` on the worker).
        weight_list: the list of ``(name, tensor)`` pairs already received for the
            main-model sync. A list (not a one-shot iterator) so it can be re-iterated.

    Returns:
        True if a drafter model was reloaded, False if there was nothing to reload.
    """
    global _logged_reload, _logged_noop
    drafter = getattr(model_runner, "drafter", None)
    drafter_model = getattr(drafter, "model", None)
    if drafter_model is None or not hasattr(drafter_model, "load_weights"):
        spec_cfg = getattr(model_runner, "speculative_config", None)
        if spec_cfg is not None and drafter is None:
            # vLLM sets model_runner.drafter whenever speculative decoding is on (last PP
            # rank). Reaching here means the attribute chain broke (vLLM rename?) -- the
            # drafter would silently draft with stale weights, so warn EVERY sync.
            logger.warning(
                "Speculative decoding is enabled (%s) but model_runner.drafter was not found; "
                "the drafter is NOT being weight-synced and will go stale. vLLM's drafter "
                "attribute layout likely changed -- update spec_decode_utils.py.",
                getattr(spec_cfg, "method", spec_cfg),
            )
        elif not _logged_noop:
            reason = (
                "speculative decoding disabled"
                if spec_cfg is None
                else f"proposer ({type(drafter).__name__}) has no weight-loadable model"
            )
            logger.info("Spec-decode drafter reload: nothing to reload (%s).", reason)
            _logged_noop = True
        return False
    weights: Iterable[Tuple[str, torch.Tensor]] = iter(weight_list)
    drafter_model.load_weights(weights)
    if not _logged_reload:
        logger.info("Spec-decode drafter (%s) reloaded from synced weights.", type(drafter_model).__name__)
        _logged_reload = True
    return True
