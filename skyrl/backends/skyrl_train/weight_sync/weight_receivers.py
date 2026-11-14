"""SkyRL's receive-side weight-transfer engines (the inference-worker half).

vLLM's NCCL and IPC engines, subclassed to add two things that hang off the
same seam: the engines call ``self.model.load_weights(...)`` directly with no
callback, so the only injection point is the ``self.model`` handle they read.
That handle is whichever model the session targets -- the main model, or the
spec-decode drafter under ``/start_draft_weight_update``.

* **Checkpoint loading.** Split compact batched-MoE FP8 wire tensors from
  ordinary checkpoint weights before calling the target model's loader.
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

    def _skyrl_lora_expert_slice(self) -> tuple[int, int, int] | None:
        """``(start, count, total)`` of the routed experts this worker serves, when the model
        takes a LoRA adapter's 3D expert tensors (``[E, ...]``) cut to them.

        A model opts in with ``skyrl_lora_local_experts = True`` (its expert LoRA layer must
        accept ``[count, ...]`` as well as ``[total, ...]``) and declares its experts with
        ``skyrl_local_expert_ids() -> (ids, total)`` on its expert modules. Under vLLM-side
        expert sharding (Inkling2 at TP8 serves 48 of 384 experts per rank) the adapter's
        expert tensors otherwise arrive and are staged whole on every rank: 8x the bytes
        this rank's LoRA slots hold, which a colocated engine has no room for.
        """
        cached = self.__dict__.get("_skyrl_lora_expert_slice_cache", False)
        if cached is not False:
            return cached
        found = None
        if getattr(self.model, "skyrl_lora_local_experts", False):
            for module in self.model.modules():
                declared = getattr(module, "skyrl_local_expert_ids", None)
                if not callable(declared):
                    continue
                ids, total = declared()
                ids = list(ids)
                current = (ids[0], len(ids), int(total)) if ids else None
                if current is None or ids != list(range(ids[0], ids[0] + len(ids))):
                    found = None
                    break
                if found is not None and found != current:
                    found = None
                    break
                found = current
        if found is not None and found[1] == found[2]:
            found = None  # every expert is local: nothing to cut
        self.__dict__["_skyrl_lora_expert_slice_cache"] = found
        return found

    @contextmanager
    def skyrl_lora_staging(self) -> Iterator[None]:
        """Install a proxy whose ``load_weights`` stages instead of loading."""
        staged = self._skyrl_lora_staged
        if staged is None:
            raise RuntimeError("skyrl_begin_lora_update must run before receiving a LoRA stream.")
        model = self.model
        expert_slice = self._skyrl_lora_expert_slice()

        def load_weights(weights: Any, **kwargs: Any) -> set:
            loaded = set()
            for name, tensor in weights:
                if name in staged:
                    raise ValueError(f"LoRA tensor {name!r} received twice in one weight update")
                if (
                    expert_slice is not None
                    and tensor.dim() == 3
                    and ".experts." in name
                    and tensor.shape[0] == expert_slice[2]
                ):
                    # Keep only this rank's experts (see _skyrl_lora_expert_slice).
                    start, count, _ = expert_slice
                    tensor = tensor[start : start + count]
                # The transport buffers (the IPC-mapped view, the packed NCCL
                # buffer) are reused or freed right after the chunk, so the
                # private copy is required, not an optimization.
                staged[name] = tensor.detach().clone()
                loaded.add(name)
            return loaded

        # Restore the exact object found, so this composes with
        # set_weight_update_target.
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


class SkyrlReceiveLifecycleMixin(SkyrlLoraStagingMixin, SkyrlCheckpointLoadMixin):
    """The update lifecycle both push engines share.

    Each engine brackets its lifecycle in ``torch.device(self.device)``. vLLM's
    own path passes ``device=`` where it matters instead; SkyRL's loaders rely on
    it being the default device that weight loading sees.

    Written once rather than per engine: NCCL and IPC differ only in the vLLM
    base class they mix into, and every step here is the same for both.
    """

    def skyrl_loads_in_place(self) -> bool:
        """Whether the target model loads synced weights straight into its
        parameters (``skyrl_load_weights_in_place``), bypassing vLLM's layerwise
        reload. Only for a model whose weights need no post-load processing:
        the reload re-materializes each layer next to the live one, which a
        model that fills the GPU (Inkling2 BF16 at TP8) has no room for."""
        return bool(getattr(self.model, "skyrl_load_weights_in_place", False))

    def start_weight_update(self) -> None:
        if self.skyrl_lora_armed():
            # No layerwise reload: an adapter update never writes to the base
            # model's parameters, so there is nothing to reload.
            self.skyrl_begin_lora_update()
            return
        if self.skyrl_loads_in_place():
            return
        skyrl_before_weight_update()
        from skyrl.backends.skyrl_train.weight_sync.colocated import selective_sleep

        if selective_sleep.is_enabled():
            # Record what this reload rewrites; the next level-2 sleep discards only that.
            selective_sleep.begin_recording()
        with torch.device(self.device):
            super().start_weight_update()

    def receive_weights(self, update_info: Any) -> None:
        try:
            if self.skyrl_lora_armed():
                with torch.device(self.device), self.skyrl_lora_staging():
                    super().receive_weights(update_info)
                return
            with torch.device(self.device), self.skyrl_checkpoint_load():
                super().receive_weights(update_info)
        finally:
            # vLLM's NCCL receiver caches each round's update info keyed by `self.model`, which here
            # is a per-round proxy whose `load_weights` closure holds the round's staged tensors (a
            # LoRA round's adapter copy on the GPU). SkyRL's trainer always ships the names, so the
            # cache is never read: drop it so no proxy, or what it holds, outlives its round.
            last = getattr(self, "_last_update_info", None)
            if last is not None:
                last.clear()

    def skyrl_release_receive_buffers(self) -> None:
        """Release receive buffers the vLLM engine holds until the end of a round.

        vLLM's ``finish_weight_update`` releases them after its layerwise
        finalize. For an adapter round, ``finish_weight_update`` calls this hook
        in place of vLLM's. No-op for transports whose receive buffers do not
        outlive ``receive_weights``.
        """

    def finish_weight_update(self) -> None:
        if self.skyrl_lora_armed():
            # No layerwise finalize: an adapter update never writes to the base
            # model's parameters.
            try:
                self.skyrl_finish_lora_update()
            finally:
                self.skyrl_release_receive_buffers()
            return
        from skyrl.backends.skyrl_train.patches.vllm.patch_moe_kernel_constants import (
            restore_moe_kernel_constants,
        )
        from skyrl.backends.skyrl_train.weight_sync.colocated import selective_sleep

        if self.skyrl_loads_in_place():
            # A level-2 sleep zeroes MoE kernel constants that no reload rebuilds.
            restore_moe_kernel_constants(self.model)
            self.skyrl_release_receive_buffers()
            return
        ok = False
        try:
            # Wake-on-write (M4b): map whatever the reload did not write before finalize,
            # which runs process_weights_after_loading on deferred attention layers'
            # kernel tensors directly, outside the write hook.
            selective_sleep.finish_staged_wake()
            with torch.device(self.device):
                super().finish_weight_update()
                # A level-2 sleep zeroes MoE kernel constants that no reload rebuilds.
                restore_moe_kernel_constants(self.model)
            ok = True
        finally:
            if selective_sleep.is_enabled():
                selective_sleep.end_recording(ok)
        skyrl_after_weight_update()
        empty_cuda_cache_rocm()


def _build_skyrl_nccl_engine() -> type:
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

    class SkyrlNCCLWeightTransferEngine(SkyrlReceiveLifecycleMixin, NCCLWeightTransferEngine):
        """vLLM's dense NCCL receive engine plus SkyRL receive targets."""

    return SkyrlNCCLWeightTransferEngine


def _build_skyrl_ipc_engine() -> type:
    from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine

    class SkyrlIPCWeightTransferEngine(SkyrlReceiveLifecycleMixin, IPCWeightTransferEngine):
        """vLLM's CUDA IPC receive engine plus SkyRL receive targets."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            from skyrl.backends.skyrl_train.weight_sync.colocated.slot_ring import (
                MultiSlotImporter,
            )

            super().__init__(*args, **kwargs)
            # One mapping per IPC slot of a transfer, released together at finish:
            # the colocated send may ring over several packed buffers, and vLLM's
            # one-slot importer would rebuild/release on every chunk (see
            # MultiSlotImporter). Identical to vLLM's with a single slot.
            self._packed_importer = MultiSlotImporter()
            self.skyrl_receive_stats = {"chunks": 0, "bytes": 0, "tensors": 0}

        def receive_weights(self, update_info: Any) -> None:
            sizes = getattr(update_info, "tensor_sizes", None) or []
            stats = self.skyrl_receive_stats
            stats["chunks"] += 1
            stats["bytes"] += int(sum(sizes))
            stats["tensors"] += len(getattr(update_info, "names", None) or [])
            if self.packed and self.skyrl_loads_in_place() and not self.skyrl_lora_armed():
                self._skyrl_receive_packed_in_place(update_info)
                return
            super().receive_weights(update_info)

        def _skyrl_receive_packed_in_place(self, update_info: Any) -> None:
            """Load a packed chunk straight from the trainer's IPC buffer, without cloning it.

            vLLM's ``packed_ipc_consumer`` clones every chunk because its layerwise
            reload may keep the views past this call. An in-place model copies each
            tensor into its parameter during ``load_weights`` and keeps nothing, so
            the clone is only a second, slot-sized transient allocation -- which a
            GPU filled by the woken weights (Inkling2 BF16 at TP8, colocated) has no
            room for. The stream is drained before returning: the trainer reuses the
            slot as soon as this round trip ends.
            """
            from vllm.distributed.weight_transfer.packed_tensor import unpack_tensor
            from vllm.model_executor.model_loader.mtp_validation import (
                disable_mtp_completeness_check,
            )

            if update_info.tensor_sizes is None:
                raise ValueError("`tensor_sizes` is required when packed=True")
            device_index = self.device.index
            uuid = str(torch.cuda.get_device_properties(device_index).uuid)
            handles = update_info.ipc_handles
            if uuid not in handles:
                raise ValueError(f"IPC handle not found for GPU UUID {uuid}. Available UUIDs: {list(handles)}")
            list_args = list(handles[uuid])
            list_args[6] = device_index
            packed = self._packed_importer.rebuild(list_args)[: sum(update_info.tensor_sizes)]
            dtypes = [getattr(torch, dn) for dn in update_info.dtype_names]
            weights = unpack_tensor(packed, update_info.names, update_info.shapes, dtypes, update_info.tensor_sizes)
            with torch.device(self.device), self.skyrl_checkpoint_load(), disable_mtp_completeness_check():
                self.model.load_weights(weights)
            del weights, packed
            torch.cuda.current_stream(self.device).synchronize()

        def start_weight_update(self) -> None:
            self.skyrl_receive_stats = {"chunks": 0, "bytes": 0, "tensors": 0}
            super().start_weight_update()

        def skyrl_release_receive_buffers(self) -> None:
            # The importer holds this worker's mapping of the trainer's packed
            # IPC buffer. Closing it once per export releases the producer-side
            # refcount, so the trainer's ``ipc_collect`` can free the buffer.
            self._packed_importer.close()

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
