"""Runtime patch: per-token NVFP4 MoE activations with the 4/6 E4M3 bound 256 on vLLM's GEMM1 input.

vLLM 0.32 quantizes the routed-expert GEMM1 input with
``flashinfer_utils.quantize_nvfp4_per_token_input``, which calls
``flashinfer.nvfp4_quantize(h, NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE, per_token_activation=True)`` with the module
constant ``NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE = 1 / (448 * 6)``. FlashInfer's kernel takes its 4/6 candidate bound
from ``FLASHINFER_NVFP4_4OVER6_E4M3_USE_256`` but its per-token base scale from that argument, so with the 256
bound the two disagree (blocks are encoded against 448 while the candidates assume 256), and the result no longer
matches TE's row-scaled 4/6 quantization with ``nvfp4_e4m3_max=256``. FlashInfer's own GEMM2-input quantization
inside the fused TRT-LLM MoE already switches its base scale to 1/(256*6) from the same env.

When ``FLASHINFER_NVFP4_4OVER6=1`` and ``FLASHINFER_NVFP4_4OVER6_E4M3_USE_256=1`` (set by ``nvfp4_runtime_env``
when the trainer's TE recipe uses 256 on activations), this sets the constant to 1/(256*6), plus any module that
imported it by name (``fused_moe/experts/flashinfer_cutedsl_moe.py``) if already imported; modules imported later
read the patched value. ``quantize_nvfp4_per_token_input`` reads the module global at call time. No-op otherwise.
Raises if the env asks for 256 but the vLLM layout lacks the constant, rather than serving mismatched numerics.

Applied in every vLLM worker process from ``inference_servers/new_inference_worker_wrap.py`` (loaded before model
init). Each application logs one line starting ``[skyrl nvfp4]`` with the pid and the env it saw.
"""

from __future__ import annotations

import os
import sys

from loguru import logger

CONSTANT = "NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE"
TARGET_MODULE = "vllm.model_executor.layers.quantization.utils.flashinfer_utils"
SCALE_448 = 1.0 / (448.0 * 6.0)
SCALE_256 = 1.0 / (256.0 * 6.0)

_APPLIED = False


def wants_e4m3_256(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return (
        env.get("FLASHINFER_NVFP4_4OVER6", "0") == "1" and env.get("FLASHINFER_NVFP4_4OVER6_E4M3_USE_256", "0") == "1"
    )


def apply_nvfp4_per_token_e4m3_256_patch() -> bool:
    """Patch the GEMM1 per-token base scale to 1/(256*6) if the env asks for it. Returns whether it applied."""
    global _APPLIED
    if not wants_e4m3_256():
        return False
    if _APPLIED:
        return True
    import importlib

    try:
        if TARGET_MODULE not in sys.modules:
            # flashinfer_utils and fused_moe import each other; entering through fused_moe resolves the cycle
            # (importing flashinfer_utils first fails with a partially-initialized-module ImportError).
            try:
                importlib.import_module("vllm.model_executor.layers.fused_moe")
            except ModuleNotFoundError as exc:
                if exc.name == "vllm":
                    raise
        module = importlib.import_module(TARGET_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == "vllm":  # no vLLM in this process at all: nothing to patch
            return False
        raise RuntimeError(
            f"FLASHINFER_NVFP4_4OVER6_E4M3_USE_256=1 needs {TARGET_MODULE}, which this vLLM lacks ({exc})"
        ) from exc
    current = getattr(module, CONSTANT, None)
    if current is None:
        raise RuntimeError(
            f"FLASHINFER_NVFP4_4OVER6_E4M3_USE_256=1 needs {TARGET_MODULE}.{CONSTANT}, which this vLLM lacks; "
            "the GEMM1 per-token activations would be quantized against 448 instead of 256."
        )
    if current not in (SCALE_448, SCALE_256):
        raise RuntimeError(f"Unexpected {TARGET_MODULE}.{CONSTANT}={current!r}; refusing to patch")
    setattr(module, CONSTANT, SCALE_256)
    rebound = []
    for name, mod in list(sys.modules.items()):  # modules that already copied the constant by name
        if mod is None or mod is module or not name.startswith("vllm."):
            continue
        if getattr(mod, CONSTANT, None) == SCALE_448:
            setattr(mod, CONSTANT, SCALE_256)
            rebound.append(name)
    _APPLIED = True
    logger.warning(
        "[skyrl nvfp4] per-token activation base global scale = 1/(256*6) (4/6 E4M3 256) pid={} "
        "FLASHINFER_NVFP4_4OVER6={} FLASHINFER_NVFP4_4OVER6_E4M3_USE_256={} FLASHINFER_NVFP4_4OVER6_ERR_MODE={} "
        "rebound={}",
        os.getpid(),
        os.environ.get("FLASHINFER_NVFP4_4OVER6"),
        os.environ.get("FLASHINFER_NVFP4_4OVER6_E4M3_USE_256"),
        os.environ.get("FLASHINFER_NVFP4_4OVER6_ERR_MODE"),
        rebound,
    )
    return True
