"""
vLLM Worker Extension for native weight sync with chunked transfer support.

This module provides NewInferenceWorkerWrap, a vLLM worker extension that
enables chunked weight updates from training to inference using the
start/update/finish lifecycle:

    skyrl_start_weight_update   ->  one or more update_weights_ipc  ->  skyrl_finish_weight_update

This separates the layerwise reload initialization/finalization from individual
chunk transfers, allowing weights to be sent in bounded-memory chunks rather
than all at once.

TODO: Once https://github.com/vllm-project/vllm/pull/39212 lands, vLLM will
natively support start_weight_update / update_weights / finish_weight_update
on GPUWorker with dedicated HTTP endpoints. At that point this worker extension
can be removed and SkyRL can call the native endpoints directly instead of
routing through /collective_rpc.

Usage:
    Pass as --worker-extension-cls to vLLM:

    vllm serve ... --worker-extension-cls \
        skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap.NewInferenceWorkerWrap
"""

import torch

from skyrl.backends.skyrl_train.inference_servers.layerwise_reload import (
    LayerwiseReloadWorkerMixin,
    _empty_cuda_cache_rocm,
)
from skyrl.backends.skyrl_train.weight_sync.delta_engine import (
    register_delta_weight_transfer_engine,
)

try:
    register_delta_weight_transfer_engine()
except ModuleNotFoundError:
    pass

VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS = f"{__name__}.NewInferenceWorkerWrap"


class NewInferenceWorkerWrap(LayerwiseReloadWorkerMixin):
    """
    vLLM worker extension for chunked weight sync (new inference path).

    Provides a three-phase weight update protocol via collective_rpc:
        1. skyrl_start_weight_update: Prepare model for receiving weights
        2. update_weights_ipc: Receive and load one chunk of weights
        3. skyrl_finish_weight_update: Finalize the model after all chunks

    Attributes accessed from the host GPUWorker (via mixin inheritance):
        self.weight_transfer_engine
        self.model_runner
        self.model_config
        self.device
    """

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

    def update_weights_ipc(self, update_info: dict) -> None:
        """
        Receive and load a single chunk of weights.

        SkyRL packs each chunk's tensors into a single contiguous CUDA buffer and sends
        one IPC handle per rank plus per-param `sizes` metadata. We rebuild
        the packed tensor here, slice it per param, and hand the list to
        model.load_weights (checkpoint format) or copy per-param directly
        (kernel format).

        Args:
            update_info: Dict with keys:
                - names: list[str]
                - dtype_names: list[str]
                - shapes: list[list[int]]
                - sizes: list[int]  (element count per param; used for slicing)
                - ipc_handles_pickled: b64(pickle({gpu_uuid: (func, args)}))
        """
        if not getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError("skyrl_start_weight_update must be called before update_weights_ipc.")

        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. " "Please set weight_transfer_config to enable weight transfer."
            )

        # --- unpack SkyRL packed CUDA IPC format ---
        import base64
        import pickle

        names = update_info["names"]
        shapes = update_info["shapes"]
        sizes = update_info["sizes"]
        pickled = update_info["ipc_handles_pickled"]
        handles = pickle.loads(base64.b64decode(pickled))

        device_index = torch.cuda.current_device()
        physical_gpu_id = str(torch.cuda.get_device_properties(device_index).uuid)
        if physical_gpu_id not in handles:
            raise ValueError(f"IPC handle not found for GPU UUID {physical_gpu_id}. " f"Available: {list(handles)}")
        func, args = handles[physical_gpu_id]
        # Remap device index to the LOCAL current-device.
        list_args = list(args)
        list_args[6] = device_index
        packed_tensor = func(*list_args)

        weights: list[tuple[str, torch.Tensor]] = []
        offset = 0
        for name, shape, size in zip(names, shapes, sizes):
            weights.append((name, packed_tensor[offset : offset + size].view(*shape)))
            offset += size

        # process_weights_after_loading reads get_current_vllm_config() (e.g.
        # flashinfer_cutlass_moe needs the compilation config to build kernels),
        # and vllm only sets that context around init_device / load_model.
        from vllm.config import set_current_vllm_config

        model = self.model_runner.model
        with set_current_vllm_config(self.vllm_config), torch.device(self.device):
            if self._skyrl_is_checkpoint_format:
                model.load_weights(weights=weights)
                # vLLM's load only updates the main model; the spec-decode (MTP/Eagle)
                # drafter is a separate module and must be reloaded from the same
                # checkpoint-format weights (see spec_decode_utils).
                from skyrl.backends.skyrl_train.inference_servers.spec_decode_utils import (
                    _reload_spec_decode_drafter,
                )

                _reload_spec_decode_drafter(self.model_runner, weights)
            else:
                for name, weight in weights:
                    param = model.get_parameter(name)
                    param.copy_(weight)

        # Ensure consumption of packed_tensor finishes before we return (and
        # before the sender drops its reference on the next barrier).
        torch.accelerator.synchronize()

    def update_weights_nccl(self, update_info: dict) -> None:
        """
        Receive a batched weight update via vLLM's NCCL weight transfer engine.

        Alternative to update_weights_ipc for the broadcast (non-IPC) sender:
        the trainer initiates an NCCL broadcast via
        NCCLWeightTransferEngine.trainer_send_weights, and each inference
        worker calls weight_transfer_engine.receive_weights here.

        Routed through this skyrl wrap (rather than vLLM's native
        /update_weights endpoint) so the load is wrapped with
        set_current_vllm_config — process_weights_after_loading on MoE
        models can otherwise instantiate kernels (e.g. FlashInfer CUTLASS)
        whose __init__ reads get_current_vllm_config().

        TODO: remove once the upstream vLLM patch lands (vllm-project/vllm
        weight-sync-fix), then route via the native /update_weights endpoint.
        https://github.com/vllm-project/vllm/pull/42577
        """
        if not getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError("skyrl_start_weight_update must be called before update_weights_nccl.")

        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. Please set weight_transfer_config to enable weight transfer."
            )

        from vllm.config import set_current_vllm_config

        from skyrl.backends.skyrl_train.inference_servers.spec_decode_utils import (
            _reload_spec_decode_drafter,
        )

        typed_update_info = self.weight_transfer_engine.parse_update_info(update_info)
        model = self.model_runner.model

        def _load_weights(weights):
            weights = list(weights)
            loaded = model.load_weights(weights=weights)
            _reload_spec_decode_drafter(self.model_runner, weights)
            return loaded

        with set_current_vllm_config(self.vllm_config), torch.device(self.device):
            self.weight_transfer_engine.receive_weights(
                typed_update_info,
                load_weights=_load_weights,
            )

        torch.accelerator.synchronize()
        _empty_cuda_cache_rocm()

    # Suspend / resume for non-colocated weight sync.
    #
    # Drive the per-worker CuMemAllocator directly instead of GPUWorker.sleep/
    # wake_up (only reachable via EngineCore.sleep, which force-clears the prefix
    # cache and preempts running requests at level >= 1). Touching the allocator
    # alone lets the caller hold a KEEP pause across the sync and resume frozen
    # requests with their KV restored to the same virtual addresses -- no abort,
    # no prefill recompute. Mirrors GPUWorker.sleep/wake_up; re-verify on vLLM bumps.

    def skyrl_sleep_for_weight_sync(self, offload_kv: bool = True) -> None:
        """Free GPU memory for weight sync by sleeping the allocator.

        Weights are discarded rather than backed up since the broadcast overwrites
        every parameter on wake. ``offload_kv`` controls whether the KV cache is
        offloaded to CPU (preserved for frozen in-flight requests) or discarded. Model
        buffers live in the weights pool but are not sent by the broadcast (e.g.
        non-persistent rotary ``inv_freq``), so save them here and restore on wake --
        as GPUWorker.sleep(level=2) does.
        """
        from vllm.device_allocator import get_mem_allocator_instance

        model = self.model_runner.model
        self._skyrl_saved_buffers = {name: buf.cpu().clone() for name, buf in model.named_buffers()}
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
        # Restore model buffers (not covered by the broadcast) once weights remap.
        saved = getattr(self, "_skyrl_saved_buffers", None)
        if saved and (tags is None or "weights" in tags):
            model = self.model_runner.model
            for name, buf in model.named_buffers():
                if name in saved:
                    buf.data.copy_(saved[name].data)
            self._skyrl_saved_buffers = {}
        # Re-init fp8 KV scales after the KV pool remaps (no-op without fp8 KV cache).
        if tags is None or "kv_cache" in tags:
            post_wake = getattr(self.model_runner, "post_kv_cache_wake_up", None)
            if post_wake is not None:
                post_wake()
