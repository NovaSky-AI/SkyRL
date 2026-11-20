"""A torch fallback for apex's ``fused_weight_gradient_mlp_cuda`` so ``gradient_accumulation_fusion`` works without apex.

With ``gradient_accumulation_fusion`` every weight-gradient GEMM accumulates straight into the
parameter's ``main_grad`` buffer (no separate ``param.grad`` tensor, no ``main_grad.add_`` in the DDP
backward hook). TransformerEngine's linear and grouped-linear modules do this themselves
(``fuse_wgrad_accumulation``). megatron-core's own ``ColumnParallelLinear`` / ``RowParallelLinear``
(the LM head, for example) call apex's ``wgrad_gemm_accum_fp32`` / ``wgrad_gemm_accum_fp16`` and
refuse to build when that extension is not installed.

The fallback computes the same thing, ``main_grad += grad_output^T @ input``, with one cuBLAS call:
``addmm_`` with beta=1 for a bf16/fp16 ``main_grad``, and an fp32-output GEMM added in place for an
fp32 ``main_grad``. It is installed only when the real extension cannot be imported.
"""

from loguru import logger

_APPLIED = False


def _flat(x):
    return x.reshape(-1, x.shape[-1])


def wgrad_gemm_accum_fp32(total_input, grad_output, main_grad):
    """``main_grad`` (fp32, [out, in]) += ``grad_output^T`` ([out, N]) @ ``total_input`` ([N, in])."""
    import torch

    inp, grad = _flat(total_input), _flat(grad_output)
    if main_grad.is_cuda:
        main_grad.add_(torch.mm(grad.t(), inp, out_dtype=torch.float32))
    else:  # torch.mm's out_dtype is CUDA-only
        main_grad.addmm_(grad.t().float(), inp.float())


def wgrad_gemm_accum_fp16(total_input, grad_output, main_grad):
    """``main_grad`` (bf16 / fp16, [out, in]) += ``grad_output^T @ total_input``, one GEMM with beta=1."""
    inp, grad = _flat(total_input), _flat(grad_output)
    main_grad.addmm_(grad.t().to(main_grad.dtype), inp.to(main_grad.dtype))


def apply_grad_accum_fusion_fallback_patch() -> bool:
    """Make megatron-core's ``gradient_accumulation_fusion`` usable without apex. Idempotent.

    Returns True when the fallback is installed, False when apex's extension is present (nothing to
    do) or megatron-core is not importable.
    """
    global _APPLIED
    if _APPLIED:
        return True
    try:
        import fused_weight_gradient_mlp_cuda  # noqa: F401

        return False
    except ImportError:
        pass
    try:
        import types

        from megatron.core.tensor_parallel import layers
    except ImportError:
        return False
    shim = types.SimpleNamespace(
        wgrad_gemm_accum_fp32=wgrad_gemm_accum_fp32, wgrad_gemm_accum_fp16=wgrad_gemm_accum_fp16
    )
    layers.fused_weight_gradient_mlp_cuda = shim
    layers._grad_accum_fusion_available = True
    _APPLIED = True
    logger.info("megatron-core gradient_accumulation_fusion: using the torch fallback for apex's wgrad GEMM.")
    return True
