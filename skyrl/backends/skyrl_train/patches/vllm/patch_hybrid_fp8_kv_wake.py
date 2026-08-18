"""Backport vllm-project/vllm#41602 for vLLM 0.26.0.

vLLM 0.26.0 assumes each quantized KV-cache entry is one tensor when it
reinitializes the cache after sleep. Hybrid models such as Qwen3.6 and Qwen3.8
also store recurrent state as lists of tensors, so ``wake_up(tags=["kv_cache"])``
fails with ``AttributeError: 'list' object has no attribute 'zero_'``.

The upstream fix handles both entry shapes. Remove this backport when the vLLM
pin includes https://github.com/vllm-project/vllm/pull/41602.
"""

import inspect
import textwrap

from loguru import logger

_OLD_LOOP = """    kv_caches = getattr(self, \"kv_caches\", [])
    for cache_tensor in kv_caches:
        if cache_tensor is not None:
            cache_tensor.zero_()
"""

_UPSTREAM_LOOP = """    kv_caches = getattr(self, \"kv_caches\", [])
    for cache_entry in kv_caches:
        if cache_entry is None:
            continue
        # Hybrid models (Mamba, DeltaNet) store per-layer state as a
        # list of tensors rather than a single tensor.
        if isinstance(cache_entry, list):
            for tensor in cache_entry:
                tensor.zero_()
        else:
            cache_entry.zero_()
"""

_PATCHED_FLAG = "_skyrl_hybrid_fp8_kv_wake_patched"


def patch_hybrid_fp8_kv_wake() -> bool:
    """Teach the pinned vLLM wake path to zero hybrid KV-cache entries."""
    from vllm.v1.worker import gpu_model_runner

    runner_cls = gpu_model_runner.GPUModelRunner
    target = runner_cls.init_fp8_kv_scales
    if getattr(target, _PATCHED_FLAG, False):
        return True

    try:
        source = textwrap.dedent(inspect.getsource(target))
    except (OSError, TypeError):
        logger.warning(
            "Cannot read vLLM init_fp8_kv_scales source; skipping hybrid KV wake patch"
        )
        return False

    if _OLD_LOOP not in source:
        logger.info(
            "vLLM init_fp8_kv_scales no longer matches 0.26.0; skipping hybrid KV wake patch. "
            "If the vLLM pin includes vllm-project/vllm#41602, delete this patch module."
        )
        return False

    patched_source = source.replace(_OLD_LOOP, _UPSTREAM_LOOP)
    namespace = gpu_model_runner.__dict__
    exec(compile(patched_source, gpu_model_runner.__file__, "exec"), namespace)  # noqa: S102
    patched = namespace["init_fp8_kv_scales"]
    setattr(patched, _PATCHED_FLAG, True)
    runner_cls.init_fp8_kv_scales = patched

    logger.info(
        "Applied hybrid FP8 KV-cache wake patch (vllm-project/vllm#41602 backport)"
    )
    return True
