"""Keep Megatron's fused L2-norm kernel inputs homogeneous without casting gradients.

TE/apex multi-tensor kernels dispatch once on the first tensor's dtype. Megatron's
precision-aware optimizer can supply both BF16 and FP32 gradients, so a mixed list
can be misread. Group at the kernel boundary, combine the local norms in FP32, and
leave Megatron's existing distributed reduction and non-L2 paths unchanged.

Remove when the pinned Megatron version handles mixed-dtype norm inputs itself.
"""

from loguru import logger


def make_dtype_grouped_applier(original, l2_norm_impl):
    """Wrap only L2 norm calls; preserve the applier interface for every other kernel."""
    import torch

    def multi_tensor_applier(op, overflow_buf, tensor_lists, *args, **kwargs):
        if op is not l2_norm_impl or len(tensor_lists) != 1 or not tensor_lists[0]:
            return original(op, overflow_buf, tensor_lists, *args, **kwargs)

        grads = tensor_lists[0]
        if all(g.dtype == grads[0].dtype and g.is_contiguous() for g in grads):
            return original(op, overflow_buf, tensor_lists, *args, **kwargs)

        # Megatron supplies local tensors on one device here, after DTensor unwrapping.
        # Keep original values/dtypes; only strided views need an allocation.
        groups = {}
        for index, grad in enumerate(grads):
            if grad.device != grads[0].device:
                raise ValueError("Gradient norm tensors must be on the same device")
            groups.setdefault(grad.dtype, []).append((index, grad.contiguous()))

        per_tensor = args[0] if args else kwargs.get("per_tensor", False)
        norms = []
        individual = [None] * len(grads) if per_tensor else None
        unused_per_tensor = None
        for entries in groups.values():
            norm, per_norm = original(op, overflow_buf, [[g for _, g in entries]], *args, **kwargs)
            norms.append(norm.float())
            unused_per_tensor = per_norm
            if per_tensor:
                for (index, _), value in zip(entries, per_norm, strict=True):
                    individual[index] = value

        # Only scalar norms are promoted. No gradient buffer is cast, and the original
        # get_grad_norm_fp32 performs its existing all-reduces exactly once afterwards.
        total = torch.linalg.vector_norm(torch.stack(norms), dim=0)
        return total, torch.stack(individual) if per_tensor else unused_per_tensor

    multi_tensor_applier._skyrl_dtype_grouped_norm = True
    return multi_tensor_applier


def patch_grad_norm_mixed_dtype() -> None:
    """Patch the module-global applier used even by previously imported norm functions."""
    from megatron.core.optimizer import clip_grads

    if getattr(clip_grads.multi_tensor_applier, "_skyrl_dtype_grouped_norm", False):
        return
    clip_grads.multi_tensor_applier = make_dtype_grouped_applier(
        clip_grads.multi_tensor_applier, clip_grads.l2_norm_impl
    )
    logger.info("Applied dtype-grouped Megatron L2 norm without casting gradients")
