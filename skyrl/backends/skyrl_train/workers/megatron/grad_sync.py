"""Connect Megatron schedules to DDP gradient accumulation contexts."""

import contextlib


def configure_no_sync(model_chunks, config) -> bool:
    """Let schedules dispatch overlapped reductions only on the last microbatch.

    Core learns per-parameter readiness counts from the first batch. Without
    its no_sync context, those counts include every microbatch and become
    invalid when packing changes the number of microbatches in a later batch.

    Installs one callable covering every chunk: only the interleaved schedule
    unwraps a list, and the others call ``config.no_sync_func`` directly.

    Returns whether overlapped reduction is wired. No-op unless every chunk is
    DDP-wrapped with ``overlap_grad_reduce``, and if a schedule context is
    already set.
    """
    if config.no_sync_func is not None:
        return False
    if not model_chunks or not all(
        hasattr(chunk, "no_sync") and getattr(getattr(chunk, "ddp_config", None), "overlap_grad_reduce", False)
        for chunk in model_chunks
    ):
        return False
    chunks = list(model_chunks)

    @contextlib.contextmanager
    def no_sync_all_chunks():
        with contextlib.ExitStack() as stack:
            for chunk in chunks:
                stack.enter_context(chunk.no_sync())
            yield

    config.no_sync_func = no_sync_all_chunks
    return True
