"""vLLM worker extension for the two weight-sync things an engine cannot do.

Weight transfer itself does not route through here: the receive path is an engine
subclass (``weight_sync/skyrl_engines.py``, ``weight_sync/delta_engine.py``,
``weight_sync/sharded_rdt/sharded_rdt_engine.py``) driven over vLLM's native RLHF
routes, which wrap ``set_current_vllm_config`` themselves and give each engine
its own layerwise-reload lifecycle.

What remains are two limits of *dispatch*:

``fetch_weights``
    ``/collective_rpc`` dispatches to worker methods by name and refuses
    callables (``entrypoints/serve/dev/rpc/api_router.py``), and no native route
    can invoke an arbitrary engine method. SkyRL's ``/fetch_weights`` route
    (``vllm_server_actor``) collective-RPCs into this method. It is called
    *before* ``pause_generation`` so the checkpoint-delta download overlaps live
    generation.

sleep / wake
    ``EngineCore.sleep`` hardcodes ``clear_prefix_cache = level >= 1``
    (``v1/engine/core.py``) with no parameter, and ``CuMemBackend.suspend`` maps
    level to tags as ``("weights",)`` / ``()`` with no way to express "discard
    weights, offload kv_cache". A custom ``SleepModeBackend`` does not help: the
    problem is on the dispatch path, not in the suspend mechanism.

Usage:
    Pass as --worker-extension-cls to vLLM:

    vllm serve ... --worker-extension-cls \
        skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap.NewInferenceWorkerWrap
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

# Everything below must run inside EVERY vLLM worker process: Worker.load_model
# builds the weight-transfer engine through the factory, and the model-runner
# recorder must be installed before load_model runs. vLLM loads this module
# before model init, which is what guarantees it. Each is guarded because this
# module is also imported from processes without the optional deps.
try:
    from skyrl.backends.skyrl_train.patches.vllm.patch_model_runner_registry import (
        apply_model_runner_registry_patch,
    )

    apply_model_runner_registry_patch()
except ModuleNotFoundError:
    pass

try:
    from skyrl.backends.skyrl_train.weight_sync.skyrl_engines import (
        register_skyrl_engines,
    )

    register_skyrl_engines()
except ModuleNotFoundError:
    pass

try:
    from skyrl.backends.skyrl_train.weight_sync.delta_engine import (
        register_delta_weight_transfer_engine,
    )

    register_delta_weight_transfer_engine()
except ModuleNotFoundError:
    pass

try:
    from skyrl.backends.skyrl_train.weight_sync.sharded_rdt import (
        rdt_vllm_register,  # noqa: F401
    )

    rdt_vllm_register.ensure_registered()
except ModuleNotFoundError:
    pass

VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS = f"{__name__}.NewInferenceWorkerWrap"


class NewInferenceWorkerWrap:
    """The weight-sync methods that must live on the worker, not an engine.

    Attributes come from the host GPUWorker: vLLM appends this class to
    ``Worker.__bases__``.
    """

    vllm_config: "VllmConfig"
    model_runner: "GPUModelRunner"
    model_config: "ModelConfig"
    device: torch.device

    def fetch_weights(self, target_version: int, sync_dir: str | None = None, uri: str | None = None):
        """Fetch/apply a checkpoint delta before the paused reload phase."""
        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. Please set weight_transfer_config to enable weight transfer."
            )
        fetch = getattr(self.weight_transfer_engine, "fetch_weights", None)
        if fetch is None:
            raise RuntimeError(f"{type(self.weight_transfer_engine).__name__} does not support fetch_weights")
        return fetch(target_version=target_version, sync_dir=sync_dir, uri=uri)

    # Suspend / resume for non-colocated weight sync.
    #
    # Drives the per-worker CuMemAllocator directly instead of GPUWorker.sleep/
    # wake_up, which is only reachable via EngineCore.sleep and force-clears the
    # prefix cache and preempts running requests at level >= 1. Touching the
    # allocator alone lets the caller hold a KEEP pause across the sync and
    # resume frozen requests with their KV at the same virtual addresses -- no
    # abort, no prefill recompute. Mirrors GPUWorker.sleep/wake_up; re-verify on
    # vLLM bumps.

    def skyrl_sleep_for_weight_sync(self, offload_kv: bool = True) -> None:
        """Free GPU memory for weight sync by sleeping the allocator.

        Weights are discarded rather than backed up since the broadcast overwrites
        every parameter on wake. ``offload_kv`` controls whether the KV cache is
        offloaded to CPU (preserved for frozen in-flight requests) or discarded. Model
        buffers live in the weights pool but are not sent by the broadcast (e.g.
        non-persistent rotary ``inv_freq``), so save them here and restore on
        wake -- as GPUWorker.sleep(level=2) does.

        The drafter's buffers are saved too: the weight sync reloads its
        parameters, but nothing else restores its buffers.
        """
        from vllm.device_allocator import get_mem_allocator_instance

        self._skyrl_saved_buffers = {name: buf.cpu().clone() for name, buf in self.model_runner.model.named_buffers()}
        draft = self._skyrl_draft_model()
        self._skyrl_saved_draft_buffers = (
            {name: buf.cpu().clone() for name, buf in draft.named_buffers()} if draft is not None else {}
        )
        get_mem_allocator_instance().sleep(offload_tags=("kv_cache",) if offload_kv else ())

    def skyrl_wake_for_weight_sync(self, tags: list) -> None:
        """Wake the given allocator tags, restoring CPU-backed contents.

        Call ``["weights"]`` before the broadcast and ``["kv_cache"]`` after. Does
        not resume the scheduler; the caller does that via ``/resume``.
        """
        from vllm.device_allocator import get_mem_allocator_instance

        # Return the broadcast's reserved-but-unallocated blocks to CUDA so cumem can
        # remap the KV pool at its fixed virtual addresses.
        torch.cuda.empty_cache()

        get_mem_allocator_instance().wake_up(tags)
        # Restore buffers (not covered by the broadcast) once weights remap.
        if tags is None or "weights" in tags:
            self._skyrl_restore_buffers(self.model_runner.model, "_skyrl_saved_buffers")
            draft = self._skyrl_draft_model()
            if draft is not None:
                self._skyrl_restore_buffers(draft, "_skyrl_saved_draft_buffers")
        # Re-init fp8 KV scales after the KV pool remaps (no-op without fp8 KV cache).
        if tags is None or "kv_cache" in tags:
            post_wake = getattr(self.model_runner, "post_kv_cache_wake_up", None)
            if post_wake is not None:
                post_wake()

    def _skyrl_draft_model(self):
        """The spec-decode drafter module, or None when there is no drafter."""
        get_draft_model = getattr(self.model_runner, "get_draft_model", None)
        return get_draft_model() if callable(get_draft_model) else None

    def _skyrl_restore_buffers(self, module, attr: str) -> None:
        saved = getattr(self, attr, None)
        if not saved:
            return
        for name, buf in module.named_buffers():
            if name in saved:
                buf.data.copy_(saved[name].data)
        setattr(self, attr, {})
