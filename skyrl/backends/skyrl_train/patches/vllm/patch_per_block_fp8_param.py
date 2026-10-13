"""Runtime patch: hand ``per_block_cast_to_fp8`` a plain tensor, not a vLLM parameter.

``vllm.utils.deep_gemm.per_block_cast_to_fp8`` is ``torch.compile``d. Online
``fp8_per_block`` quantization calls it with ``layer.weight``, which is a vLLM
parameter subclass (``BasevLLMParameter`` defines ``__torch_function__``). Under
torch 2.13, dynamo recurses through that ``__torch_function__`` while tracing
``m, n = x.shape`` and raises ``InternalTorchDynamoError: RecursionError`` on the first
weight sync. A plain tensor (``.data``) or a plain ``nn.Parameter`` compiles fine and
produces the same values, so unwrap before the call.

Remove once vLLM passes ``layer.weight.data`` itself.
"""

from loguru import logger

_PATCHED = False


def apply_per_block_fp8_param_patch() -> bool:
    """Install the patch once per process. Returns True if installed."""
    global _PATCHED
    if _PATCHED:
        return False
    try:
        import torch
        from vllm.model_executor.layers.quantization.online import fp8 as online_fp8
        from vllm.utils import deep_gemm
    except ImportError:
        return False

    compiled = getattr(deep_gemm, "per_block_cast_to_fp8", None)
    if compiled is None:
        return False

    def per_block_cast_to_fp8(x, *args, **kwargs):
        if isinstance(x, torch.Tensor) and type(x) is not torch.Tensor:
            x = x.data
        return compiled(x, *args, **kwargs)

    # online/fp8 imported the name at module load; fp8_utils re-imports it from deep_gemm
    # at call time, so both bindings need the wrapper.
    deep_gemm.per_block_cast_to_fp8 = per_block_cast_to_fp8
    online_fp8.per_block_cast_to_fp8 = per_block_cast_to_fp8
    _PATCHED = True
    logger.info("Installed per_block_cast_to_fp8 plain-tensor patch")
    return True
