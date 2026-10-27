"""Restore FlashInfer CUTLASS MoE per-expert constants after a weight sync.

``FlashInferExperts.__init__`` (``vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py``)
builds per-expert constant tensors with ``torch.full`` from scalar quant-config values:
``gemm1_clamp_limit`` (clamped SwiGLU, e.g. DeepSeek-V4's ``swiglu_limit``), ``gemm1_alpha``,
``gemm1_beta`` and the MXFP4xMXFP8 ``fake_input_scale``. They are plain attributes of the kernel
object -- not parameters or buffers -- allocated in the weights memory pool, so a level-2 sleep zeroes
them and neither the worker's buffer save/restore nor a weight reload rebuilds them. A zero clamp limit
clamps every routed expert's SwiGLU input to 0.

``restore_moe_kernel_constants`` refills them in place (CUDA graphs keep their addresses) from the
kernel's own quant config. Run after every full weight sync; it is a no-op where the values are intact.

Remove once vLLM registers these as buffers or rebuilds them on wake.
"""

from typing import Any

from loguru import logger


def restore_moe_kernel_constants(model: Any) -> int:
    """Refill FlashInferExperts' per-expert constants under ``model``; returns tensors refilled."""
    try:
        import torch
        from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
            FlashInferExperts,
        )
    except ImportError:
        return 0
    modules = getattr(model, "modules", None)
    if modules is None:
        return 0

    refilled = 0
    with torch.no_grad():
        for module in modules():
            quant_method = getattr(module, "quant_method", None)
            impl = getattr(getattr(quant_method, "moe_kernel", None), "impl", None)
            experts = getattr(impl, "fused_experts", None)
            if not isinstance(experts, FlashInferExperts):
                continue
            quant_config = experts.quant_config
            for attr, value in (
                ("gemm1_clamp_limit", quant_config.gemm1_clamp_limit),
                ("gemm1_alpha", quant_config.gemm1_alpha),
                ("gemm1_beta", quant_config.gemm1_beta),
                ("fake_input_scale", 1.0),
            ):
                tensor = getattr(experts, attr, None)
                if isinstance(tensor, torch.Tensor) and value is not None:
                    tensor.fill_(float(value))
                    refilled += 1
    if refilled:
        logger.info(f"Restored {refilled} FlashInfer CUTLASS MoE per-expert constant tensors after weight sync")
    return refilled
