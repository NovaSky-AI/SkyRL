"""Gradient synchronization at SkyRL's optimizer-step boundary."""

from contextlib import ExitStack, contextmanager


@contextmanager
def defer_grad_sync(model_chunks):
    """Accumulate locally, including across multiple pipeline schedule calls.

    Megatron's schedules normally enable DDP synchronization for their last
    microbatch. SkyRL's optimizer window can contain several schedule calls,
    so only ``optim_step`` knows when the accumulated gradients are complete.
    Clear the schedule callbacks while owning the DDP no-sync contexts: nesting
    the schedule's own DDP no-sync would re-enable hooks when its context exits.
    """
    with ExitStack() as stack:
        for chunk in model_chunks:
            config = chunk.config
            for name in ("no_sync_func", "grad_sync_func"):
                stack.callback(setattr, config, name, getattr(config, name))
                setattr(config, name, None)
            stack.enter_context(chunk.no_sync())
        yield


def start_deferred_grad_sync(model_chunks):
    """Dispatch async reductions once the whole optimizer window is complete.

    Non-overlap DDP dispatches synchronously from ``finalize_model_grads``.
    Overlap DDP normally dispatches from backward hooks, which were suppressed
    during accumulation, so it needs an explicit start before finalization.
    """
    for chunk in model_chunks:
        if chunk.ddp_config.overlap_grad_reduce:
            chunk.start_grad_sync()
