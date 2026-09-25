"""SkyRL's receive-side weight-transfer engines (the inference-worker half).

vLLM's NCCL and IPC engines, subclassed to add two things, both of which hang
off the same seam: the engines call ``self.model.load_weights(...)`` directly
with no callback, so the only injection point is the ``self.model`` handle they
read, and both interpositions are a proxy over it.

* **Drafter reload.** The speculative-decoding drafter
  (``model_runner.drafter.model``) is a separate module that the main model's
  ``load_weights`` never touches.
* **LoRA staging.** When the round has been armed with a LoRA receive target
  (``weight_sync/lora_target.py``), the stream is a PEFT adapter rather than the
  model: the tensors are staged for vLLM's LoRA manager instead of loaded, and
  the base model is not touched at all.

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


class SkyrlLoraStagingMixin:
    """Apply one update round to a LoRA adapter instead of the base model.

    Armed per round by the trainer, before ``send_weights()``, through
    ``NewInferenceWorkerWrap.skyrl_set_lora_receive_target`` -- the round trip
    itself is vLLM's and carries only names, dtypes and shapes (see
    ``weight_sync/lora_target.py`` on why the target rides ``/collective_rpc``).

    Arming lasts exactly one round: ``finish`` disarms, so a failed or missing
    arm can never silently apply an adapter stream to the base model, or the
    reverse.
    """

    # Class-level defaults rather than an __init__: vLLM's factory constructs
    # these engines through the base __init__, which SkyRL does not override.
    _skyrl_lora_target: dict[str, Any] | None = None
    _skyrl_lora_staged: dict[str, torch.Tensor] | None = None

    def skyrl_set_lora_receive_target(self, receive_target: dict[str, Any]) -> None:
        """Arm the next update round to build the named adapter."""
        from skyrl.backends.skyrl_train.weight_sync.lora_target import (
            is_lora_receive_target,
        )

        if not is_lora_receive_target(receive_target):
            raise ValueError(f"Not a LoRA receive target: {receive_target!r}")
        if self._skyrl_lora_target is not None:
            raise RuntimeError(
                f"A LoRA receive target is already armed for "
                f"{self._skyrl_lora_target['lora_name']!r}; finish that update first."
            )
        if not self._skyrl_lora_capable():
            raise RuntimeError(
                "Received a LoRA weight update but this engine was started without --enable-lora "
                "(trainer.policy.model.lora.rank > 0 with merge_lora=false sets it)."
            )
        self._skyrl_lora_target = dict(receive_target)

    def _skyrl_lora_capable(self) -> bool:
        """Whether this vLLM worker can register an adapter at all."""
        return getattr(self.vllm_config, "lora_config", None) is not None

    def skyrl_lora_armed(self) -> bool:
        return self._skyrl_lora_target is not None

    def skyrl_begin_lora_update(self) -> None:
        self._skyrl_lora_staged = {}

    @contextmanager
    def skyrl_lora_staging(self) -> Iterator[None]:
        """Install a proxy whose ``load_weights`` stages instead of loading."""
        staged = self._skyrl_lora_staged
        if staged is None:
            raise RuntimeError("skyrl_begin_lora_update must run before receiving a LoRA stream.")
        model = self.model

        def load_weights(weights: Any, **kwargs: Any) -> set:
            loaded = set()
            for name, tensor in weights:
                if name in staged:
                    raise ValueError(f"LoRA tensor {name!r} received twice in one weight update")
                # The transport buffers (the IPC-mapped view, the packed NCCL
                # buffer) are reused or freed right after the chunk, so the
                # private copy is required, not an optimization.
                staged[name] = tensor.detach().clone()
                loaded.add(name)
            return loaded

        # Same swap discipline as skyrl_drafter_reload: restore the exact object
        # found, so this composes with set_weight_update_target.
        self.model = _LoadWeightsProxy(model, load_weights)
        try:
            yield
        finally:
            self.model = model

    def skyrl_finish_lora_update(self) -> None:
        """Hand the round's tensors to vLLM's LoRA manager, and disarm."""
        from skyrl.backends.skyrl_train.patches.vllm.patch_lora_in_memory import (
            stage_in_memory_adapter,
        )
        from skyrl.backends.skyrl_train.weight_sync.lora_target import (
            expand_lora_aliases,
        )

        target = self._skyrl_lora_target
        staged = self._skyrl_lora_staged
        # Disarm before the work: one round per arm, even if this raises.
        self._skyrl_lora_target = None
        self._skyrl_lora_staged = None
        if target is None:
            raise RuntimeError("skyrl_finish_lora_update called without an armed LoRA receive target.")
        if not staged:
            raise RuntimeError(f"LoRA weight update for {target['lora_name']!r} finished without receiving any tensors")
        tensors = expand_lora_aliases(staged, target.get("aliases") or {})
        stage_in_memory_adapter(target["lora_name"], tensors, target["adapter_config"])
        # Deliberately not logged here. Only loggers under vLLM's own namespace
        # are configured in a worker process, so an INFO line from this module
        # would be dropped without a trace; the trainer logs the same counts
        # (it has them before the send) where the output is captured.


class SkyrlReceiveLifecycleMixin(SkyrlLoraStagingMixin, SkyrlDrafterReloadMixin):
    """The update lifecycle both push engines share.

    Each engine brackets its lifecycle in ``torch.device(self.device)``. vLLM's
    own path passes ``device=`` where it matters instead; SkyRL's loaders rely on
    it being the default device that weight loading sees.

    Written once rather than per engine: NCCL and IPC differ only in the vLLM
    base class they mix into, and every step here is the same for both.
    """

    def start_weight_update(self) -> None:
        if self.skyrl_lora_armed():
            # No layerwise reload: an adapter update never writes to the base
            # model's parameters, so there is nothing to reload.
            self.skyrl_begin_lora_update()
            return
        with torch.device(self.device):
            super().start_weight_update()

    def receive_weights(self, update_info: Any) -> None:
        if self.skyrl_lora_armed():
            with torch.device(self.device), self.skyrl_lora_staging():
                super().receive_weights(update_info)
            return
        with torch.device(self.device), self.skyrl_drafter_reload():
            super().receive_weights(update_info)

    def finish_weight_update(self) -> None:
        if self.skyrl_lora_armed():
            # No layerwise finalize (nothing was reloaded) and no drafter reload
            # (the drafter does not carry the adapter).
            self.skyrl_finish_lora_update()
            return
        with torch.device(self.device):
            super().finish_weight_update()
        empty_cuda_cache_rocm()


def _build_skyrl_nccl_engine() -> type:
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

    class SkyrlNCCLWeightTransferEngine(SkyrlReceiveLifecycleMixin, NCCLWeightTransferEngine):
        """vLLM's dense NCCL receive engine plus the drafter reload and LoRA staging."""

    return SkyrlNCCLWeightTransferEngine


def _build_skyrl_ipc_engine() -> type:
    from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine

    class SkyrlIPCWeightTransferEngine(SkyrlReceiveLifecycleMixin, IPCWeightTransferEngine):
        """vLLM's CUDA IPC receive engine plus the drafter reload and LoRA staging."""

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
