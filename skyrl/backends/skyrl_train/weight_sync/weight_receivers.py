"""SkyRL's receive-side weight-transfer engines (the inference-worker half).

vLLM's NCCL and IPC engines, subclassed to add one thing: loading the compact
batched-MoE FP8 wire tensors. vLLM's engines call ``self.model.load_weights(...)``
directly with no callback, so the only injection point is the ``self.model``
handle they read. That handle is whichever model the session targets -- the main
model, or the spec-decode drafter under ``/start_draft_weight_update``.

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


class SkyrlCheckpointLoadMixin:
    """Split the compact batched-MoE FP8 wire tensors from ordinary checkpoint weights."""

    @contextmanager
    def skyrl_checkpoint_load(self) -> Iterator[None]:
        """Install the FP8-aware loader proxy over ``self.model``."""
        from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
            _load_checkpoint_weights,
        )

        model = self.model

        def load_weights(weights: Any, **kwargs: Any) -> Any:
            return _load_checkpoint_weights(model, weights, **kwargs)

        # The proxy scopes the override to the `WeightTransferEngine` context
        # instead of mutating `load_weights` on the model object itself.
        self.model = _LoadWeightsProxy(model, load_weights)
        try:
            yield
        finally:
            # Restore the exact object we found, so this composes with the
            # worker's set_weight_update_target / reset_weight_update_target.
            self.model = model


# Each engine brackets its lifecycle in `torch.device(self.device)`. vLLM's own
# path passes `device=` where it matters instead; SkyRL's loaders rely on it
# being the default device that weight loading sees.


def _build_skyrl_nccl_engine() -> type:
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

    class SkyrlNCCLWeightTransferEngine(SkyrlCheckpointLoadMixin, NCCLWeightTransferEngine):
        """vLLM's dense NCCL receive engine plus the FP8-aware loader."""

        def start_weight_update(self) -> None:
            with torch.device(self.device):
                super().start_weight_update()

        def receive_weights(self, update_info: Any) -> None:
            with torch.device(self.device), self.skyrl_checkpoint_load():
                super().receive_weights(update_info)

        def finish_weight_update(self) -> None:
            with torch.device(self.device):
                super().finish_weight_update()
            empty_cuda_cache_rocm()

    return SkyrlNCCLWeightTransferEngine


def _build_skyrl_ipc_engine() -> type:
    from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine

    class SkyrlIPCWeightTransferEngine(SkyrlCheckpointLoadMixin, IPCWeightTransferEngine):
        """vLLM's CUDA IPC receive engine plus the FP8-aware loader."""

        def start_weight_update(self) -> None:
            with torch.device(self.device):
                super().start_weight_update()

        def receive_weights(self, update_info: Any) -> None:
            with torch.device(self.device), self.skyrl_checkpoint_load():
                super().receive_weights(update_info)

        def finish_weight_update(self) -> None:
            with torch.device(self.device):
                super().finish_weight_update()
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
