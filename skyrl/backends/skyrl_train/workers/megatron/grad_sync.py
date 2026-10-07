"""Connect Megatron schedules to DDP gradient accumulation contexts."""


def configure_no_sync(model_chunks, config):
    """Let schedules dispatch overlapped reductions only on the last microbatch.

    Core learns per-parameter readiness counts from the first batch. Without
    its no_sync context, those counts include every microbatch and become
    invalid when packing changes the number of microbatches in a later batch.
    """
    if config.no_sync_func is not None:
        return
    if not model_chunks or not all(
        getattr(getattr(chunk, "ddp_config", None), "overlap_grad_reduce", False) for chunk in model_chunks
    ):
        return
    callbacks = [chunk.no_sync for chunk in model_chunks]
    config.no_sync_func = callbacks[0] if len(callbacks) == 1 else callbacks
