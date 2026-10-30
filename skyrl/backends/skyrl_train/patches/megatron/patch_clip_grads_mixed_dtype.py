"""Run megatron-core's grad-norm and grad-clip multi-tensor kernels once per dtype.

``clip_grads.get_grad_norm_fp32`` and ``clip_grad_by_total_norm_fp32`` hand every grad to one
``multi_tensor_applier`` call, and TransformerEngine's (and apex's) multi-tensor kernels dispatch on the
first tensor's dtype. Without the precision-aware optimizer those grads are the fp32 main grads, so the
list is uniform. With it (``use_precision_aware_optimizer``, which full-finetune CPU optimizer offload needs
to avoid materializing fp32 copies of every grad on GPU) they are the ``decoupled_grad`` views of the model
grad buffers, which keep each param's dtype. Models that train some params in fp32 under bf16 -- the mHC
params of DeepSeek-V4/V4.1 and GLM-5.3 -- then pass a mixed list, the kernel reads bf16 tensors as fp32,
and the first optimizer step dies with "CUDA error: an illegal memory access" in ``multi_tensor_apply``.

TODO: remove once the pinned megatron-core groups grads by dtype itself: NVIDIA/Megatron-LM#7279 (open; issue
NVIDIA/Megatron-LM#7338) loops ``get_grad_norm_fp32`` and ``clip_grad_by_total_norm_fp32`` over
``_group_grads_by_dtype`` and sums the squared norms, the same math as this wrapper. Check for
``_group_grads_by_dtype`` in the pinned ``clip_grads.py`` before removing; see ``README.md``.
"""

from loguru import logger

_APPLIED = False


def split_by_dtype(tensor_lists):
    """``[[a0, a1, ...], [b0, b1, ...]]`` -> ``{dtype: [[a_i ...], [b_i ...]]}``, keyed by the first list's dtypes."""
    groups = {}
    for index, tensor in enumerate(tensor_lists[0]):
        groups.setdefault(tensor.dtype, []).append(index)
    return {dtype: [[tensors[i] for i in indices] for tensors in tensor_lists] for dtype, indices in groups.items()}


def apply_clip_grads_mixed_dtype_patch() -> bool:
    """Wrap ``clip_grads.multi_tensor_applier`` to split mixed-dtype tensor lists. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    try:
        import torch
        from megatron.core.optimizer import clip_grads
    except ImportError:
        return False

    inner = clip_grads.multi_tensor_applier
    l2_norm_impl = getattr(clip_grads, "l2_norm_impl", None)
    warned = []

    def multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
        groups = split_by_dtype(tensor_lists) if tensor_lists and tensor_lists[0] else {}
        if len(groups) <= 1:
            return inner(op, noop_flag_buffer, tensor_lists, *args)
        if not warned:
            warned.append(True)
            logger.info(f"clip_grads: splitting a mixed-dtype grad list by dtype ({sorted(map(str, groups))})")
        results = [inner(op, noop_flag_buffer, lists, *args) for lists in groups.values()]
        if op is l2_norm_impl:
            # Each call returns (norm, per-tensor norms); per_tensor is False at both call sites.
            total = torch.sqrt(sum(norm.float().square() for norm, _ in results))
            return total, None
        return results[0]

    clip_grads.multi_tensor_applier = multi_tensor_applier
    _APPLIED = True
    return True
