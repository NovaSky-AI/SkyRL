"""Small compatibility patches for the installed vLLM runtime."""

import logging
from collections.abc import Iterator, Mapping
from functools import wraps
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


def _iter_kv_cache_tensors(value: object) -> Iterator[torch.Tensor]:
    """Yield tensor leaves from vLLM's attention and Mamba KV-cache layout."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_kv_cache_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_kv_cache_tensors(child)


def _has_nested_kv_cache_entries(kv_caches: object) -> bool:
    if isinstance(kv_caches, Mapping):
        return True
    if not isinstance(kv_caches, (list, tuple)):
        return False
    return any(isinstance(cache, (Mapping, list, tuple)) for cache in kv_caches)


def patch_vllm_fp8_kv_cache_sleep_wake(runner_cls: type[Any] | None = None) -> bool:
    """Handle nested Mamba cache tensors in vLLM 0.23 FP8 sleep/wake.

    vLLM's ``GPUModelRunner.init_fp8_kv_scales`` resets FP8 caches after a
    ``wake_up(tags=["kv_cache"])`` call.  Its current implementation assumes
    every ``self.kv_caches`` entry is a tensor.  Hybrid attention/Mamba models
    such as Qwen3.5 include a list of Mamba state tensors in that list, so the
    upstream reset fails before the first rollout.

    Keep the upstream reset implementation, including its scale restoration,
    but present a temporary flat list of tensor leaves for the duration of the
    call.  Flat attention-only layouts continue through the upstream method
    unchanged.
    """
    if runner_cls is None:
        try:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        except ImportError:
            return False
        runner_cls = GPUModelRunner

    original: Callable[..., Any] | None = getattr(runner_cls, "init_fp8_kv_scales", None)
    if not callable(original):
        return False
    if getattr(original, "_skyrl_handles_nested_kv_caches", False):
        return False

    @wraps(original)
    def _patched_init_fp8_kv_scales(self: Any, *args: Any, **kwargs: Any) -> Any:
        kv_caches = getattr(self, "kv_caches", None)
        if not _has_nested_kv_cache_entries(kv_caches):
            return original(self, *args, **kwargs)

        tensor_leaves = list(_iter_kv_cache_tensors(kv_caches))
        self.kv_caches = tensor_leaves
        try:
            return original(self, *args, **kwargs)
        finally:
            self.kv_caches = kv_caches

    setattr(_patched_init_fp8_kv_scales, "_skyrl_handles_nested_kv_caches", True)
    runner_cls.init_fp8_kv_scales = _patched_init_fp8_kv_scales
    logger.info("Patched vLLM FP8 KV-cache sleep/wake reset for nested cache entries")
    return True
