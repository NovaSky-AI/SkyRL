"""Drop the MoE token dispatcher's reference to the router probabilities once the layer is done.

``MoEAlltoAllTokenDispatcher.dispatch_preprocess`` stores ``self.probs`` (the router output, with
its ``grad_fn``); the same forward's ``combine_preprocess`` reads it, and nothing after that.
Left in place, it pins that forward's autograd graph until the next forward through the layer
overwrites it. Under full activation recompute the forward re-runs inside each layer's backward,
so every MoE layer keeps its recomputed graph -- back to the checkpoint's detached input and that
input's ``.grad`` -- for the rest of backward: ~21 GiB/GPU at 64k tokens on GLM-5.3-Flash
(42 MoE layers, TP8). Not specific to GLM or mHC: any MoE model under full recompute pays it.

The release wraps ``MoELayer.postprocess``, the last step of every MoE forward -- the plain
``MoELayer.forward`` and the fine-grained overlap callables both end there -- rather than a
dispatcher method, so a patch that replaces dispatcher methods (e.g. a custom all-to-all) can't
drop it. Upstream, the same one line could live at the end of ``combine_postprocess`` or of
``MoELayer.postprocess``.
"""

import functools

from loguru import logger

_APPLIED = False


def patch_moe_release_dispatcher_probs() -> bool:
    """Wrap ``MoELayer.postprocess`` to clear ``token_dispatcher.probs``. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    from megatron.core.transformer.moe.moe_layer import MoELayer

    inner = MoELayer.postprocess

    @functools.wraps(inner)
    def postprocess(self, *args, **kwargs):
        output = inner(self, *args, **kwargs)
        dispatcher = getattr(self, "token_dispatcher", None)
        if getattr(dispatcher, "probs", None) is not None:
            dispatcher.probs = None
        return output

    MoELayer.postprocess = postprocess
    _APPLIED = True
    logger.info("MoE: token dispatcher drops its router-probs reference after each MoE forward")
    return True
