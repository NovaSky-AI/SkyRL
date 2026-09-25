"""SkyRL's receive-side weight-transfer engines (the inference-worker half).

vLLM's NCCL and IPC engines, subclassed to add one thing: reloading the
speculative-decoding drafter. The drafter (``model_runner.drafter.model``) is a
separate module that the main model's ``load_weights`` never touches, and vLLM's
engines call ``self.model.load_weights(...)`` directly with no callback, so the
only injection point is the ``self.model`` handle they read.

Registered (in ``weight_sync/register.py``) under ``skyrl_nccl`` / ``skyrl_ipc``
rather than shadowing vLLM's
``nccl`` / ``ipc``: ``register_engine`` raises on a duplicate name, and
``WeightTransferConfig.backend`` is typed ``Literal[...] | str`` and validated
against the registry, so a new name is all that is needed.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

import torch

logger = logging.getLogger(__name__)

SKYRL_NCCL_BACKEND = "skyrl_nccl"
SKYRL_IPC_BACKEND = "skyrl_ipc"


def empty_cuda_cache_rocm() -> None:
    """Release unused ROCm cached blocks after a full-weight sync.

    ROCm's allocator does not return the reload's transient blocks on its own.
    """
    if torch.version.hip is None or not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


class _LoadWeightsProxy:
    """Wraps a model, overriding only ``load_weights``.

    Every other attribute access falls through to the real model.
    """

    def __init__(self, model: Any, load_weights: Any) -> None:
        self._model = model
        self.load_weights = load_weights

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not set on the proxy itself.
        return getattr(self._model, name)


class SkyrlDrafterReloadMixin:
    """Handle SkyRL-specific reloads around the engine's model load.

    The wrapper splits the compact batched-MoE FP8 wire tensors from ordinary
    checkpoint weights, then reloads a spec-decode drafter when one is present.
    """

    @contextmanager
    def skyrl_drafter_reload(self) -> Iterator[None]:
        """Install the FP8-aware, drafter-reloading proxy over ``self.model`."""
        from skyrl.backends.skyrl_train.patches.vllm.patch_model_runner_registry import (
            current_model_runner,
        )

        model_runner = current_model_runner()
        drafter = getattr(model_runner, "drafter", None) if model_runner is not None else None
        from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
            _load_checkpoint_weights,
        )

        model = self.model

        def load_weights(weights: Any, **kwargs: Any) -> Any:
            # The engines hand us a one-shot generator. The compact FP8 loader
            # and a speculative drafter both need a complete view of the stream.
            weight_list = list(weights)
            loaded = _load_checkpoint_weights(model, weight_list, **kwargs)
            if drafter is not None and getattr(drafter, "model", None) is not None:
                from skyrl.backends.skyrl_train.inference_servers.spec_decode_utils import (
                    _reload_spec_decode_drafter,
                )

                _reload_spec_decode_drafter(model_runner, weight_list)
            return loaded

        # The proxy scopes the override to the `WeightTransferEngine` context
        # instead of mutating `load_weights` on the model object itself.
        self.model = _LoadWeightsProxy(model, load_weights)
        try:
            yield
        finally:
            # Restore the exact object we found, so this composes with the
            # worker's set_weight_update_target / reset_weight_update_target.
            self.model = model


def skyrl_before_weight_update() -> None:
    """Serialized-FP8 hooks that must be in place before the reload runs.

    MXFP8 TRT-LLM MoE prepare re-derives a fixed per-layer weight/scale
    relocation on every sync. Replace it with a learned, bitwise-validated
    permutation cache; it falls back to the original on any mismatch, and is a
    no-op for non-MXFP8 wires. Installed here rather than at worker-extension
    import so the wrap lands in the process that owns the engine, and only once
    a sync actually happens.
    """
    from skyrl.backends.skyrl_train.inference_servers.trtllm_moe_prepare_cache import (
        install as install_trtllm_moe_prepare_cache,
    )

    install_trtllm_moe_prepare_cache()


def skyrl_after_weight_update() -> None:
    """Re-assert the serialized-FP8 wire's KV/attention scale contract.

    The wire ships no KV/attention scale calibration, so those scales are 1.0 by
    contract. vLLM corrupts them at boot (compressed-tensors copies the
    dummy-load placeholders verbatim) and after a level-2 wake
    (``init_fp8_kv_scales`` resets only the k/v tensors — q wakes as 0.0 and the
    float mirrors keep garbage), which serves NaN on the quantized-Q path or
    silently wrong logprobs on the bf16-Q path under ``kv_cache_dtype=fp8_*``.
    Both of those are patched in ``vllm_compat``; a reload re-runs
    ``process_weights_after_loading``, so re-assert it here too.

    Gated on a dummy-weight boot: these engines also serve models started from a
    real FP8 checkpoint, whose calibrated k/v scales must survive a sync that
    carries no replacement for them.
    """
    from skyrl.backends.skyrl_train.inference_servers.vllm_compat import (
        booted_without_checkpoint_weights,
        normalize_serialized_fp8_kv_scales,
    )

    if not booted_without_checkpoint_weights():
        return
    from skyrl.backends.skyrl_train.patches.vllm.patch_model_runner_registry import (
        current_model_runner,
    )

    model_runner = current_model_runner()
    if model_runner is None:
        return
    count = normalize_serialized_fp8_kv_scales(model_runner)
    if count:
        logger.info("Normalized FP8 KV/attention scales to 1.0 on %d layers after weight sync", count)


# Each engine brackets its lifecycle in `torch.device(self.device)`. vLLM's own
# path passes `device=` where it matters instead; SkyRL's loaders rely on it
# being the default device that weight loading sees.


def _build_skyrl_nccl_engine() -> type:
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

    class SkyrlNCCLWeightTransferEngine(SkyrlDrafterReloadMixin, NCCLWeightTransferEngine):
        """vLLM's dense NCCL receive engine plus the drafter reload."""

        def start_weight_update(self) -> None:
            skyrl_before_weight_update()
            with torch.device(self.device):
                super().start_weight_update()

        def receive_weights(self, update_info: Any) -> None:
            with torch.device(self.device), self.skyrl_drafter_reload():
                super().receive_weights(update_info)

        def finish_weight_update(self) -> None:
            with torch.device(self.device):
                super().finish_weight_update()
            skyrl_after_weight_update()
            empty_cuda_cache_rocm()

    return SkyrlNCCLWeightTransferEngine


def _build_skyrl_ipc_engine() -> type:
    from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine

    class SkyrlIPCWeightTransferEngine(SkyrlDrafterReloadMixin, IPCWeightTransferEngine):
        """vLLM's CUDA IPC receive engine plus the drafter reload."""

        def start_weight_update(self) -> None:
            skyrl_before_weight_update()
            with torch.device(self.device):
                super().start_weight_update()

        def receive_weights(self, update_info: Any) -> None:
            with torch.device(self.device), self.skyrl_drafter_reload():
                super().receive_weights(update_info)

        def finish_weight_update(self) -> None:
            with torch.device(self.device):
                super().finish_weight_update()
            skyrl_after_weight_update()
            empty_cuda_cache_rocm()

    return SkyrlIPCWeightTransferEngine


# Built lazily and cached: the classes subclass vLLM's engines, and this module
# must stay importable without the wheel.
_ENGINE_CACHE: dict[str, type] = {}


def get_skyrl_nccl_engine() -> type:
    if SKYRL_NCCL_BACKEND not in _ENGINE_CACHE:
        _ENGINE_CACHE[SKYRL_NCCL_BACKEND] = _build_skyrl_nccl_engine()
    return _ENGINE_CACHE[SKYRL_NCCL_BACKEND]


def get_skyrl_ipc_engine() -> type:
    if SKYRL_IPC_BACKEND not in _ENGINE_CACHE:
        _ENGINE_CACHE[SKYRL_IPC_BACKEND] = _build_skyrl_ipc_engine()
    return _ENGINE_CACHE[SKYRL_IPC_BACKEND]
