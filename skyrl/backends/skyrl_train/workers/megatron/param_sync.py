"""Complete deferred DDP parameter publication before out-of-forward reads."""


def _stages_into_grad_buffer(optimizer):
    # Core's staging only does work when the param all-gather reuses the grad
    # buffer (MXFP8); it then zeroes that buffer and copies FP32 masters into it.
    config = getattr(optimizer, "config", None)
    return bool(
        getattr(config, "reuse_grad_buf_for_mxfp8_param_ag", False) and getattr(config, "overlap_param_gather", False)
    )


def _grad_buffers_freed(chunks):
    # Offloading the optimizer frees the DDP grad buffers with storage().resize_(0).
    for chunk in chunks:
        for buffer in list(getattr(chunk, "buffers", [])) + list(getattr(chunk, "expert_parallel_buffers", [])):
            grad_data = buffer.grad_data
            if grad_data is not None and grad_data.numel() > 0 and grad_data.untyped_storage().size() == 0:
                return True
    return False


def sync_params_for_export(model_chunks, optimizer=None):
    """Gather updated optimizer shards before transfer/checkpoint/export on all ranks.

    With parameter-gather overlap, an optimizer step updates only local shards;
    the next forward pre-hooks normally publish the complete model. Exporters
    read tensors directly and cannot rely on those hooks. Use Core's explicit
    synchronization API, including optimizer staging for reused FP8 buffers.

    Non-overlap and unwrapped inference models already contain complete weights.
    Call only at a completed optimizer boundary, with model parameters on GPU.
    """
    if model_chunks is None:
        return
    chunks = [
        chunk for chunk in model_chunks if getattr(getattr(chunk, "ddp_config", None), "overlap_param_gather", False)
    ]
    if not chunks:
        return

    if optimizer is not None and _stages_into_grad_buffer(optimizer) and _grad_buffers_freed(chunks):
        # Skipping the staging would all-gather grad data into the parameters.
        raise RuntimeError(
            "Cannot publish parameters with reuse_grad_buf_for_mxfp8_param_ag and overlap_param_gather "
            "while the optimizer is offloaded: staging writes into the freed DDP grad buffers. "
            "Keep the optimizer on GPU for this export."
        )

    import torch

    with torch.no_grad():
        if optimizer is not None:
            # Older Core releases have no FP8 shared-buffer staging API. A
            # ChainedOptimizer implements it itself, zeroing each shared model
            # chunk's buffer once before its sub-optimizers stage into it, so do
            # not unwrap it here.
            prepare = getattr(optimizer, "prepare_model_params_for_param_sync", None)
            if prepare is not None:
                prepare()
        for chunk in chunks:
            chunk.start_param_sync(force_sync=True)
