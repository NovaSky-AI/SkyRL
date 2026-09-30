"""Keep full-recompute checkpoint inputs in pinned host memory (opt-in).

Under ``recompute_granularity="full"`` each layer's forward runs under ``no_grad`` inside
megatron-core's ``checkpointed_forward``, and the only tensors autograd keeps are each checkpoint's
inputs -- one hidden-state tensor per layer, alive until backward reaches that layer. For
GLM-5.3-Flash those are the mHC n-stream residuals (``[s / TP, b, 4 * hidden]``): 45 of them,
~1.4 MB per token per GPU at TP8, the largest term in long-context memory after the recompute
working set.

``torch.autograd.graph.save_on_cpu`` around the block-level ``checkpointed_forward`` moves exactly
those to pinned host memory and brings each back when backward unpacks it. The recompute inside
backward runs outside the context, so its activations stay on the GPU. Host cost: the same bytes
in pinned RAM per GPU (e.g. ~45 GiB/GPU at 256k tokens on 64 GPUs); PCIe cost: one D2H + one H2D
copy per layer per step.

Enable with ``SKYRL_OFFLOAD_CHECKPOINT_INPUTS=1``.
"""

import functools

import torch
from loguru import logger

_APPLIED = False


def patch_offload_checkpoint_inputs() -> bool:
    """Wrap ``transformer_block.checkpointed_forward`` in ``save_on_cpu``. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    from megatron.core.transformer import transformer_block

    inner = transformer_block.checkpointed_forward

    @functools.wraps(inner)
    def checkpointed_forward(*args, **kwargs):
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            return inner(*args, **kwargs)

    transformer_block.checkpointed_forward = checkpointed_forward
    _APPLIED = True
    logger.info("Full-recompute checkpoint inputs are kept in pinned host memory (SKYRL_OFFLOAD_CHECKPOINT_INPUTS=1)")
    return True
