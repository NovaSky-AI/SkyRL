"""SkyRL's receive-side weight-transfer engines (the inference-worker half).

vLLM's NCCL and IPC engines, subclassed to add one thing: reloading the
speculative-decoding drafter. The drafter (``model_runner.drafter.model``) is a
separate module that the main model's ``load_weights`` never touches, and vLLM's
engines call ``self.model.load_weights(...)`` directly with no callback, so the
only injection point is the ``self.model`` handle they read.

Registered under ``skyrl_nccl`` / ``skyrl_ipc`` rather than shadowing vLLM's
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

_REGISTERED = False


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
    """Reload the spec-decode drafter from the same weights the main model got.

    Wrap the engine's own load in :meth:`skyrl_drafter_reload`. Costs nothing
    when this process has no drafter: the proxy is not installed at all.
    """

    @contextmanager
    def skyrl_drafter_reload(self) -> Iterator[None]:
        """Install the drafter-reloading proxy over ``self.model``, if needed."""
        from skyrl.backends.skyrl_train.patches.vllm.patch_model_runner_registry import (
            current_model_runner,
        )

        model_runner = current_model_runner()
        drafter = getattr(model_runner, "drafter", None) if model_runner is not None else None
        if drafter is None or getattr(drafter, "model", None) is None:
            # No speculative decoding, or a proposer with no loadable model
            # (ngram): nothing to interpose.
            yield
            return

        from skyrl.backends.skyrl_train.inference_servers.spec_decode_utils import (
            _reload_spec_decode_drafter,
        )

        model = self.model

        def load_weights(weights: Any, **kwargs: Any) -> Any:
            # The engines hand us a one-shot generator, and the drafter needs its
            # own pass to filter for the names it consumes.
            weight_list = list(weights)
            loaded = model.load_weights(weights=weight_list, **kwargs)
            _reload_spec_decode_drafter(model_runner, weight_list)
            return loaded

        self.model = _LoadWeightsProxy(model, load_weights)
        try:
            yield
        finally:
            # Restore the exact object we found, so this composes with the
            # worker's set_weight_update_target / reset_weight_update_target.
            self.model = model


# Each engine brackets its lifecycle in `torch.device(self.device)`. vLLM's own
# path does not (it passes `device=` where it matters), but SkyRL's loaders have
# always run under it, so keep it as the default device weight loading sees.


def _build_skyrl_nccl_engine() -> type:
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

    class SkyrlNCCLWeightTransferEngine(SkyrlDrafterReloadMixin, NCCLWeightTransferEngine):
        """vLLM's dense NCCL receive engine plus the drafter reload."""

        def start_weight_update(self) -> None:
            with torch.device(self.device):
                super().start_weight_update()

        def receive_weights(self, update_info: Any) -> None:
            with torch.device(self.device), self.skyrl_drafter_reload():
                super().receive_weights(update_info)

        def finish_weight_update(self) -> None:
            with torch.device(self.device):
                super().finish_weight_update()
            empty_cuda_cache_rocm()

    return SkyrlNCCLWeightTransferEngine


def _build_skyrl_ipc_engine() -> type:
    from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine

    class SkyrlIPCWeightTransferEngine(SkyrlDrafterReloadMixin, IPCWeightTransferEngine):
        """vLLM's CUDA IPC receive engine plus the drafter reload."""

        def start_weight_update(self) -> None:
            with torch.device(self.device):
                super().start_weight_update()

        def receive_weights(self, update_info: Any) -> None:
            with torch.device(self.device), self.skyrl_drafter_reload():
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


def register_skyrl_engines() -> None:
    """Register ``skyrl_nccl`` / ``skyrl_ipc`` in vLLM's factory (idempotent).

    Must run in every vLLM worker process (``Worker.load_model`` builds the
    engine through the factory) and on the driver (which validates
    ``WeightTransferConfig.backend`` against the registry).
    """
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory
    except ImportError:
        # vLLM is a Linux-only optional dependency; nothing to register.
        logger.debug("vLLM not importable; skipping SkyRL weight-transfer engine registration.")
        return

    for name, loader in (
        (SKYRL_NCCL_BACKEND, get_skyrl_nccl_engine),
        (SKYRL_IPC_BACKEND, get_skyrl_ipc_engine),
    ):
        if name in WeightTransferEngineFactory._registry:
            continue
        # Direct-class registration, so resolve now; vLLM is importable by here.
        WeightTransferEngineFactory.register_engine(name, loader())
    _REGISTERED = True
