"""Compatibility fixes for vLLM inference workers."""

import logging
from functools import wraps
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


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


# Per-worker-process latch, set by the model-loader patch below. ``True`` only
# once this process has loaded a model with ``load_format="dummy"``.
_BOOTED_WITHOUT_CHECKPOINT_WEIGHTS = False


def booted_without_checkpoint_weights() -> bool:
    """Return whether this worker's weights came from ``load_format="dummy"``.

    Gate for every KV/attention scale normalization below. Forcing those scales
    to 1.0 is only correct when *nothing* calibrated them: SkyRL's serialized
    FP8 wire ships no scale calibration, and the engines that receive it boot
    with ``load_format="dummy"`` (see
    ``inference_servers/utils._apply_serialized_fp8_weight_sync_defaults``).

    This module is imported by *every* SkyRL vLLM engine, because
    ``new_inference_worker_wrap`` is the worker-extension class for all of them
    — including one serving a real FP8 checkpoint whose ``k_scale``/``v_scale``
    were calibrated offline. Overwriting those with 1.0 silently drifts FP8-KV
    generation, so the normalization must not fire there. A dummy-weight boot is
    the exact discriminator: it means no checkpoint scale ever reached the
    layers, so 1.0 is the only value they can legitimately hold.
    """
    return _BOOTED_WITHOUT_CHECKPOINT_WEIGHTS


def _load_format_is_dummy(vllm_config: Any) -> bool:
    load_format = getattr(getattr(vllm_config, "load_config", None), "load_format", None)
    if load_format is None:
        return False
    # vLLM has spelled this as both a plain str and a ``LoadFormat`` enum.
    return str(getattr(load_format, "value", load_format)).lower() == "dummy"


def patch_vllm_dummy_weight_boot_detection(loader_cls: type[Any] | None = None) -> bool:
    """Latch whether this worker process booted its weights from dummy values.

    ``BaseModelLoader.load_model`` is the one place that sees the resolved
    ``LoadConfig`` in every worker process, whatever the executor backend, and
    it runs before ``process_weights_after_loading``, so the latch is live by
    the time the boot normalization below needs it. Reading an env var instead
    would not survive the Ray executor, which starts workers from the raylet
    rather than from the server actor that set it.
    """
    if loader_cls is None:
        try:
            from vllm.model_executor.model_loader.base_loader import BaseModelLoader
        except ImportError:
            return False
        loader_cls = BaseModelLoader

    original: Callable[..., Any] | None = getattr(loader_cls, "load_model", None)
    if not callable(original):
        return False
    if getattr(original, "_skyrl_latches_dummy_boot", False):
        return False

    @wraps(original)
    def _patched_load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _BOOTED_WITHOUT_CHECKPOINT_WEIGHTS
        # GPUModelRunner passes vllm_config by keyword; accept the positional
        # form too. Anything else resolves to None, which latches False — an
        # upstream signature change must degrade to "leave the scales alone",
        # never to a reset that overwrites calibrated ones.
        vllm_config = kwargs.get("vllm_config", args[0] if args else None)
        _BOOTED_WITHOUT_CHECKPOINT_WEIGHTS = _load_format_is_dummy(vllm_config)
        if _BOOTED_WITHOUT_CHECKPOINT_WEIGHTS:
            logger.info("Dummy-weight boot detected: FP8 KV/attention scales will be normalized to 1.0")
        return original(self, *args, **kwargs)

    setattr(_patched_load_model, "_skyrl_latches_dummy_boot", True)
    loader_cls.load_model = _patched_load_model
    return True


def patch_vllm_fp8_kv_scale_completion(runner_cls: type[Any] | None = None) -> bool:
    """Complete vLLM's post-wake FP8 KV scale reset for serialized-FP8 engines.

    Wraps ``GPUModelRunner.post_kv_cache_wake_up`` to run
    :func:`normalize_serialized_fp8_kv_scales` after the upstream reset, but
    only on a dummy-weight boot — see :func:`booted_without_checkpoint_weights`
    for why an engine serving a calibrated FP8 checkpoint must be left alone.
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
        if not booted_without_checkpoint_weights():
            return result
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

    Fires only on a dummy-weight boot (see
    :func:`booted_without_checkpoint_weights`); a real FP8 checkpoint's
    calibrated scales are left exactly as loaded.
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
                if booted_without_checkpoint_weights():
                    _normalize_layer_fp8_scales(layer)
                return result

            setattr(_patched, "_skyrl_normalizes_fp8_scales", True)
            return _patched

        setattr(cls, "process_weights_after_loading", _make(original))
        patched_any = True
    if patched_any:
        logger.info("Patched vLLM KV-cache methods to normalize FP8 scales at boot (pre-capture)")
    return patched_any
