"""Complete deferred DDP parameter publication before out-of-forward reads."""


def sync_params_for_export(model_chunks, optimizer=None):
    """Gather updated optimizer shards before transfer/checkpoint/export on all ranks.

    With parameter-gather overlap, an optimizer step updates only local shards;
    the next forward pre-hooks normally publish the complete model. Exporters
    read tensors directly and cannot rely on those hooks. Use Core's explicit
    synchronization API, including optimizer staging for reused FP8 buffers.

    Non-overlap and unwrapped inference models already contain complete weights.
    Call only at a completed optimizer boundary, with model parameters on GPU.
    """
    chunks = [
        chunk for chunk in model_chunks if getattr(getattr(chunk, "ddp_config", None), "overlap_param_gather", False)
    ]
    if not chunks:
        return

    import torch

    with torch.no_grad():
        if optimizer is not None:
            # Older Core releases have no FP8 shared-buffer staging API.
            prepare = getattr(optimizer, "prepare_model_params_for_param_sync", None)
            if prepare is not None:
                prepare()
        for chunk in chunks:
            chunk.start_param_sync(force_sync=True)
