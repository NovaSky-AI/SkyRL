"""Drop a hybrid mHC attention layer's ``attn_norm_manager`` once the layer is done.

``TransformerLayer._forward_self_attention_output_with_bias`` stores the fine-grained offloading
interface for the input layernorm on ``self.attn_norm_manager``; that interface keeps the layer's
input (``self.tensor``) even with offloading off. ``TransformerLayer``'s own forward clears it after
the attention BDA, but ``HyperConnectionHybridLayer._call_inner_transformer_layer_without_local_bda``
(the fast path every split attention-only hybrid layer takes, e.g. DeepSeek-V4 / V4.1) uses it and
never clears it -- its MLP branch does clear ``mlp_norm_manager``. So every attention sublayer pins
its last forward's mHC-aggregated input, with its autograd graph, until its next forward:

- under full activation recompute the forward re-runs inside each layer's backward, so each finished
  attention layer keeps its recomputed graph, back to the checkpoint's detached input and that
  input's ``.grad`` (two ``[s, b, n*C]`` tensors), for the rest of backward: +4.3 GiB per V4.1 layer
  pair at 128k tokens, ~60 GiB/GPU at 256k on an 8-layer slice;
- otherwise one ``[s, b, C]`` tensor per attention layer outlives the step (2.7 GiB per layer at 256k).

Same class as ``patch_moe_release_dispatcher_probs.py``. Upstream, the one line
``layer.attn_norm_manager = None`` belongs after the attention branch's
``_group_offload_output_with_bias`` call, as the MLP branch does (megatron-core main and dev at
2026-11-04 still lack it).
"""

import functools

from loguru import logger

_APPLIED = False


def patch_hybrid_mhc_release_attn_norm_manager() -> bool:
    """Wrap ``HyperConnectionHybridLayer._call_inner_transformer_layer_without_local_bda`` to clear the inner
    layer's ``attn_norm_manager``. Idempotent; ``False`` when megatron-core has no such layer."""
    global _APPLIED
    if _APPLIED:
        return True
    try:
        from megatron.core.models.hybrid.layers.hybrid_hyper_connection import (
            HyperConnectionHybridLayer,
        )
    except ImportError:
        return False
    inner = getattr(HyperConnectionHybridLayer, "_call_inner_transformer_layer_without_local_bda", None)
    if inner is None:
        return False

    @functools.wraps(inner)
    def call_inner(self, *args, **kwargs):
        try:
            return inner(self, *args, **kwargs)
        finally:
            layer = getattr(self, "inner_layer", None)
            if getattr(layer, "attn_norm_manager", None) is not None:
                layer.attn_norm_manager = None

    HyperConnectionHybridLayer._call_inner_transformer_layer_without_local_bda = call_inner
    _APPLIED = True
    logger.info("Hybrid mHC: attention layers drop their attn_norm_manager (and its input) after each forward")
    return True
