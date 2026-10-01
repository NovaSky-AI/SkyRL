"""Skip the FP32 upcast before the MoE combine's 2-rank expert-TP reduce-scatter.

``MoEAlltoAllTokenDispatcher.combine_preprocess`` reduce-scatters the expert outputs across the
expert-tensor-parallel group as ``reduce_scatter(hidden.to(self.probs.dtype)).to(hidden.dtype)``.
With an FP32 router (``moe_router_dtype="float32"``, as GLM-5.3-Flash uses) that materializes an
FP32 copy of every token-expert row this rank received: ~19 GiB/GPU at 576k tokens on
GLM-5.3-Flash (TP8, EP32 x ETP2), the allocation that ran that length out of memory.

With exactly two ranks the reduction is a single addition per element, and NCCL's bf16 sum (FP32
accumulate, one rounding) equals the FP32 sum rounded back to bf16 -- checked bit-for-bit on
64M elements spanning five orders of magnitude. So for a 2-rank expert-TP group the upcast buys
nothing, and this patch reduces in the activation dtype instead. Larger groups (several roundings
in bf16) are left untouched.

``combine_preprocess`` reads ``self.probs`` only for its dtype there, so the wrapper swaps in an
empty placeholder of the activation dtype for the duration of the call.
"""

import functools

from loguru import logger

_APPLIED = False


def patch_moe_combine_bf16_reduce() -> bool:
    """Wrap ``MoEAlltoAllTokenDispatcher.combine_preprocess``. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
    )

    inner = MoEAlltoAllTokenDispatcher.combine_preprocess

    @functools.wraps(inner)
    def combine_preprocess(self, hidden_states, *args, **kwargs):
        probs = getattr(self, "probs", None)
        if self.tp_size != 2 or probs is None or probs.dtype == hidden_states.dtype:
            return inner(self, hidden_states, *args, **kwargs)
        self.probs = probs.new_empty(0, dtype=hidden_states.dtype)
        try:
            return inner(self, hidden_states, *args, **kwargs)
        finally:
            self.probs = probs

    MoEAlltoAllTokenDispatcher.combine_preprocess = combine_preprocess
    _APPLIED = True
    logger.info("MoE combine: 2-rank expert-TP reduce-scatter in the activation dtype (no FP32 upcast)")
    return True
