"""Recompute the mHC mappings too under ``recompute_modules=[..., "mhc"]``.

megatron-core's ``HyperConnectionModule._forward_with_checkpoint`` (the ``mhc`` selective-recompute path) checkpoints
only ``aggregate`` and calls ``compute_mappings`` directly. ``compute_mappings`` upcasts the n-stream hidden states
(``[s, b, n * hidden]``) to fp32 for the mapping projection, and autograd keeps that fp32 copy for the matmul
backward: twice the bf16 hidden states, in every attention and MoE sub-layer. On DeepSeek-V4.1-Flash (n=4,
hidden 4096, 80 sub-layers) that is 17.8 GiB of a 136 GiB peak at ~3.5K-token rows.

This runs ``compute_mappings`` under non-reentrant ``torch.utils.checkpoint``, which saves only its bf16 input (a
view of the hidden states the residual path keeps anyway) and recomputes the small mapping projection in backward.
The mapping math is deterministic, so gradients are unchanged.
"""

from loguru import logger

_APPLIED = False


def apply_mhc_recompute_mappings_patch() -> bool:
    """Wrap ``HyperConnectionModule._forward_with_checkpoint``. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    try:
        from megatron.core.transformer.hyper_connection import HyperConnectionModule
        from torch.utils.checkpoint import checkpoint
    except ImportError:
        return False

    original = HyperConnectionModule._forward_with_checkpoint

    def _forward_with_checkpoint(self, hidden_states, manager):
        compute_mappings = self.compute_mappings
        self.compute_mappings = lambda x: checkpoint(compute_mappings, x, use_reentrant=False)
        try:
            return original(self, hidden_states, manager)
        finally:
            del self.compute_mappings

    HyperConnectionModule._forward_with_checkpoint = _forward_with_checkpoint
    _APPLIED = True
    logger.info("Installed mHC recompute patch: compute_mappings is checkpointed under recompute_modules=[mhc]")
    return True
