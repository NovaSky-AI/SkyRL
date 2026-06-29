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

from typing import Iterable, List, Tuple

import torch


def _reload_spec_decode_drafter(model_runner, weight_list: List[Tuple[str, torch.Tensor]]) -> bool:
    """Re-load the spec-decode drafter model from ``weight_list`` (HF-named tensors).

    Args:
        model_runner: the vLLM ``GPUModelRunner`` (``self.model_runner`` on the worker).
        weight_list: the list of ``(name, tensor)`` pairs already received for the
            main-model sync. A list (not a one-shot iterator) so it can be re-iterated.

    Returns:
        True if a drafter model was reloaded, False if there was nothing to reload.
    """
    drafter = getattr(model_runner, "drafter", None)
    drafter_model = getattr(drafter, "model", None)
    if drafter_model is None or not hasattr(drafter_model, "load_weights"):
        # No spec decoding, or a drafter without a weight-loadable model (e.g. ngram).
        return False
    weights: Iterable[Tuple[str, torch.Tensor]] = iter(weight_list)
    drafter_model.load_weights(weights)
    return True
