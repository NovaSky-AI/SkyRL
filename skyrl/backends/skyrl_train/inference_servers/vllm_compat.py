"""Compatibility fixes for vLLM inference workers."""

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
    """Patch vLLM 0.23 FP8 cache reset for nested hybrid-model caches.

    Qwen3.5 Mamba cache entries are nested, while upstream expects tensors.
    Flatten them only during reset; flat layouts remain on the upstream path.
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


def normalize_serialized_fp8_kv_scales(model_runner: Any) -> int:
    """Force every attention layer's Q/K/V/prob scales — tensors, float mirrors,
    and CPU copies — to 1.0.

    Serialized FP8 weight sync ships no KV/attention scale calibration, so 1.0
    is the wire's contract. vLLM 0.26 breaks it twice on this path:

    * ``CompressedTensorsKVCacheMethod.process_weights_after_loading`` copies the
      boot placeholders verbatim (no "<0 => default 1.0" branch like the base
      class), so ``load_format=dummy`` random init becomes the live scales and
      ``*_float`` mirrors from the first token.
    * After a level-2 sleep, ``GPUModelRunner.init_fp8_kv_scales`` resets only
      the ``_k_scale``/``_v_scale`` tensors; ``_q_scale`` wakes as 0.0 and every
      float mirror keeps its garbage. FlashInfer feeds
      ``bmm1_scale *= _q_scale_float * _k_scale_float`` from exactly those.

    Measured on Qwen3.5-0.8B (fp8_e4m3 KV): quantized-Q path generates NaN
    (``_q_scale`` = 0), bf16-Q path generates finite garbage (mean logprob diff
    1.33 vs the 0.05 bar) from ``_k_scale_float`` = -0.000745. Returns the
    number of layers normalized.
    """
    context = getattr(getattr(model_runner, "compilation_config", None), "static_forward_context", None)
    if not context:
        return 0
    normalized = 0
    for module in context.values():
        normalized += _normalize_layer_fp8_scales(module)
    return normalized


def _normalize_layer_fp8_scales(module: Any) -> int:
    """Set one attention layer's scale tensors, float mirrors, and CPU copies to 1.0."""
    k_scale = getattr(module, "_k_scale", None)
    if not isinstance(k_scale, torch.Tensor):
        return 0
    for name in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale", "_k_scale_cpu", "_v_scale_cpu"):
        tensor = getattr(module, name, None)
        if isinstance(tensor, torch.Tensor):
            # .data: some scale tensors are Parameters (requires_grad varies
            # by load path) and in-place ops on grad-requiring leaves throw.
            tensor.data.fill_(1.0)
    for name in ("_k_scale_float", "_v_scale_float", "_q_scale_float"):
        if hasattr(module, name):
            setattr(module, name, 1.0)
    return 1


def patch_vllm_fp8_kv_scale_completion(runner_cls: type[Any] | None = None) -> bool:
    """Complete vLLM's post-wake FP8 KV scale reset for serialized-FP8 engines.

    Wraps ``GPUModelRunner.post_kv_cache_wake_up`` to run
    :func:`normalize_serialized_fp8_kv_scales` after the upstream reset. Only
    engines booted by SkyRL's serialized FP8 weight sync import this module, and
    that wire never ships scale calibration, so 1.0 is always correct here.
    """
    if runner_cls is None:
        try:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        except ImportError:
            return False
        runner_cls = GPUModelRunner

    original: Callable[..., Any] | None = getattr(runner_cls, "post_kv_cache_wake_up", None)
    if not callable(original):
        return False
    if getattr(original, "_skyrl_completes_fp8_kv_scales", False):
        return False

    @wraps(original)
    def _patched_post_kv_cache_wake_up(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        count = normalize_serialized_fp8_kv_scales(self)
        if count:
            logger.info("Normalized FP8 KV/attention scales to 1.0 on %d layers after wake", count)
        return result

    setattr(_patched_post_kv_cache_wake_up, "_skyrl_completes_fp8_kv_scales", True)
    runner_cls.post_kv_cache_wake_up = _patched_post_kv_cache_wake_up
    logger.info("Patched vLLM post-wake FP8 KV scale reset (q/prob scales + float mirrors)")
    return True


def patch_vllm_fp8_kv_scale_boot_normalization() -> bool:
    """Normalize KV/attention scales at boot, before CUDA-graph capture.

    Under ``load_format=dummy`` the scale placeholders are random.
    ``CompressedTensorsKVCacheMethod.process_weights_after_loading`` copies them
    verbatim into the live tensors and ``*_float`` mirrors (the base class only
    normalizes the never-loaded sentinel case), and FlashInfer bakes the floats
    into captured attention plans (``bmm1_scale *= _q_scale_float *
    _k_scale_float``) — after which no runtime reset can reach them. Measured:
    with runtime-only normalization the NaN goes away but generation stays
    wrong (mean logprob diff 1.24 vs 0.05); with ``enforce_eager`` the same
    stack passes, pinning the baked-at-capture mechanism. Wrap both KV-cache
    methods so every layer leaves weight processing with scales == 1.0 — the
    serialized-FP8 wire's contract, since it ships no scale calibration.
    """
    try:
        from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
    except ImportError:
        return False
    targets = [BaseKVCacheMethod]
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsKVCacheMethod,
        )
    except ImportError:
        pass
    else:
        # Subclass overrides the method, so it needs its own wrap.
        if "process_weights_after_loading" in vars(CompressedTensorsKVCacheMethod):
            targets.append(CompressedTensorsKVCacheMethod)

    patched_any = False
    for cls in targets:
        original = vars(cls).get("process_weights_after_loading")
        if original is None or getattr(original, "_skyrl_normalizes_fp8_scales", False):
            continue

        def _make(original: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(original)
            def _patched(self: Any, layer: Any, *args: Any, **kwargs: Any) -> Any:
                result = original(self, layer, *args, **kwargs)
                _normalize_layer_fp8_scales(layer)
                return result

            setattr(_patched, "_skyrl_normalizes_fp8_scales", True)
            return _patched

        setattr(cls, "process_weights_after_loading", _make(original))
        patched_any = True
    if patched_any:
        logger.info("Patched vLLM KV-cache methods to normalize FP8 scales at boot (pre-capture)")
    return patched_any
