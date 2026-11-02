"""Let megatron-core's output-discarding checkpoint recompute when none of its inputs require grad.

``CheckpointWithoutOutput.checkpoint`` (used by the ``mhc`` / ``layernorm`` / ``moe_act`` selective-recompute
modules) saves its inputs with ``ctx.save_for_backward`` inside ``CheckpointWithoutOutputFunction``, discards its
outputs, and recomputes them from a grad hook on a downstream tensor. When no tensor input requires grad, autograd
does not connect the function and ``save_for_backward`` keeps nothing, but the hook still fires as soon as a later
consumer is trainable -- under LoRA the embeddings and early base weights are frozen while the adapters after the
first mHC / layernorm blocks are not. ``_load_args_from_ctx`` then runs off the end of ``ctx.saved_tensors``:
``RuntimeError: coroutine raised StopIteration`` in backward (DeepSeek-V4 LoRA with ``recompute_modules=[...,
mhc, layernorm]``). Full finetuning is unaffected because every input requires grad.

When no tensor input requires grad, keep references to the inputs on the ctx -- the same memory the normal path
pins through ``save_for_backward`` -- and let ``_load_args_from_ctx`` use them.

Remove once megatron-core keeps (or recomputes from) inputs that do not require grad.
"""

import functools

from loguru import logger

_APPLIED = False
_FALLBACK_ATTR = "_skyrl_inputs_without_grad"


def apply_checkpoint_without_output_frozen_inputs_patch() -> bool:
    """Patch ``CheckpointWithoutOutput.checkpoint`` and ``_load_args_from_ctx``. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    try:
        import torch
        from megatron.core.tensor_parallel import random as mcore_random
    except ImportError:
        return False

    checkpoint_cls = mcore_random.CheckpointWithoutOutput
    inner_checkpoint = checkpoint_cls.checkpoint
    inner_load = mcore_random._load_args_from_ctx

    @functools.wraps(inner_checkpoint)
    def checkpoint(self, run_function, *args):
        outputs = inner_checkpoint(self, run_function, *args)
        ctx = getattr(self, "ctx", None)
        tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
        if ctx is not None and tensor_args and not any(a.requires_grad for a in tensor_args):
            setattr(ctx, _FALLBACK_ATTR, args)
        return outputs

    @functools.wraps(inner_load)
    def _load_args_from_ctx(ctx):
        saved = getattr(ctx, _FALLBACK_ATTR, None)
        if saved is None:
            return inner_load(ctx)
        return tuple(a.detach() if isinstance(a, torch.Tensor) else a for a in saved)

    checkpoint_cls.checkpoint = checkpoint
    mcore_random._load_args_from_ctx = _load_args_from_ctx
    _APPLIED = True
    logger.info("Patched CheckpointWithoutOutput to recompute from inputs that do not require grad")
    return True
