"""
vLLM Worker Extension for native weight sync with chunked transfer support.

This module provides NewInferenceWorkerWrap, a vLLM worker extension that
enables chunked weight updates from training to inference using the
start/update/finish lifecycle:

    start_weight_update   ->  one or more update_weights_ipc  ->  finish_weight_update

This separates the layerwise reload initialization/finalization from individual
chunk transfers, allowing weights to be sent in bounded-memory chunks rather
than all at once.

Used only with the new inference path (_SKYRL_USE_NEW_INFERENCE=1).

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

# Importing this registers the SkyRL ``sharded_rdt`` weight-transfer engine into
# vLLM's factory and relaxes WeightTransferConfig.backend to accept it. vLLM
# loads this module via --worker-extension-cls before model init on every
# worker, so the registration is correctly ordered (it must run before
# GPUWorker.__init__ builds the engine from weight_transfer_config.backend).
from skyrl.backends.skyrl_train.weight_sync import rdt_vllm_register  # noqa: F401

# Workaround for a vLLM layerwise-reload corruption affecting NemotronH/Mamba.
# MambaMixer2 registers `conv_weights` as a non-persistent buffer that is a
# view of `self.conv1d.weight.data` (shared storage). vLLM's reload code path
# (model_executor/model_loader/reload/layerwise.py) materializes the buffer
# into a fresh uninitialized GPU tensor and then runs
# `kernel_conv_weights.data.copy_(fresh)` in `_copy_and_restore_kernel_tensors`.
# Because the kernel buffer shares storage with `conv1d.weight.data`, this
# writes garbage (NaN-bit-pattern bytes in bf16) into the conv1d weight,
# corrupting all 23 Mamba layers after every weight sync.
#
# Adding "conv_weights" to vLLM's SKIP_TENSORS makes capture/restore/materialize
# skip the buffer entirely, so the view stays intact and conv1d.weight is
# preserved. Must be applied before `record_metadata_for_reloading` runs at
# model construction; this module is imported by vLLM via
# --worker-extension-cls before model init, so the import-time patch is
# correctly ordered.
# Remove this pending https://github.com/vllm-project/vllm/pull/42481 which should
# be included in vLLM 0.21.0
try:
    from vllm.model_executor.model_loader.reload.meta import (
        SKIP_TENSORS as _VLLM_SKIP_TENSORS,
    )

    _VLLM_SKIP_TENSORS.add("conv_weights")
except ImportError:
    pass

VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS = f"{__name__}.NewInferenceWorkerWrap"


class NewInferenceWorkerWrap:
    """
    vLLM worker extension for chunked weight sync (new inference path).

    Provides a three-phase weight update protocol via collective_rpc:
        1. start_weight_update: Prepare model for receiving weights
        2. update_weights_ipc: Receive and load one chunk of weights
        3. finish_weight_update: Finalize the model after all chunks

    Attributes accessed from the host GPUWorker (via mixin inheritance):
        self.weight_transfer_engine
        self.model_runner
        self.model_config
        self.device
    """

    def start_weight_update(self, is_checkpoint_format: bool = True) -> None:
        """
        Prepare the model for a new weight update.

        For checkpoint-format weights, initializes the layerwise reload
        machinery which moves layers to meta device and wraps weight loaders
        to defer processing until all weights for each layer are loaded.

        Must be called before any update_weights_ipc calls.

        Args:
            is_checkpoint_format: True if incoming weights are in checkpoint
                format (need layerwise processing). False if weights are
                already in kernel format (direct copy).
        """
        if getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError(
                "start_weight_update called while a weight update is "
                "already active. Call finish_weight_update first."
            )

        if is_checkpoint_format:
            from vllm.config import set_current_vllm_config
            from vllm.model_executor.model_loader.reload import (
                initialize_layerwise_reload,
            )

            model = self.model_runner.model
            with set_current_vllm_config(self.vllm_config), torch.device(self.device):
                initialize_layerwise_reload(model)

        self._skyrl_is_checkpoint_format = is_checkpoint_format
        self._skyrl_weight_update_active = True

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
            raise RuntimeError("start_weight_update must be called before update_weights_ipc.")

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
            raise RuntimeError("start_weight_update must be called before update_weights_nccl.")

        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. Please set weight_transfer_config to enable weight transfer."
            )

        from vllm.config import set_current_vllm_config

        typed_update_info = self.weight_transfer_engine.parse_update_info(update_info)
        model = self.model_runner.model

        with set_current_vllm_config(self.vllm_config), torch.device(self.device):
            self.weight_transfer_engine.receive_weights(
                typed_update_info,
                load_weights=model.load_weights,
            )

        torch.accelerator.synchronize()

    def _inject_rdt_model_device(self) -> None:
        """Give the sharded_rdt engine the model/device/vllm_config it needs.

        vLLM 0.20.2 builds the weight-transfer engine in GPUWorker.__init__,
        before the model is loaded and with only (config, parallel_config) — so
        the sharded_rdt engine has no model/device. This worker extension is
        mixed into the GPUWorker, so it CAN reach ``self.model_runner.model``,
        ``self.device``, and ``self.vllm_config`` and inject them. Idempotent.
        """
        engine = self.weight_transfer_engine
        engine.model = self.model_runner.model
        engine.device = self.device
        engine.vllm_config = self.vllm_config

    def init_weight_transfer_engine_rdt(self, init_info: dict) -> None:
        """Initialize the sharded_rdt engine and bake the replay plan.

        Routed here (via collective_rpc) instead of vLLM's native
        /init_weight_transfer_engine because the native GPUWorker path passes no
        model and does not wrap the call in set_current_vllm_config — but the
        sharded_rdt bake drives model.load_weights + layerwise internals and
        needs both. We inject the model/device, run the (model-free) actor
        resolution + warmup in init_transfer_engine, then bake under the vLLM
        config context.
        """
        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. Please set weight_transfer_config " "to enable weight transfer."
            )
        from vllm.config import set_current_vllm_config

        self._inject_rdt_model_device()
        typed_init_info = self.weight_transfer_engine.parse_init_info(init_info)
        self.weight_transfer_engine.init_transfer_engine(typed_init_info)
        with set_current_vllm_config(self.vllm_config), torch.device(self.device):
            self.weight_transfer_engine.bake()

    def update_weights_rdt(self, update_info: dict) -> None:
        """Receive one layer-group of weights via the sharded_rdt engine.

        Mirror of update_weights_nccl: the engine's receive_weights pulls the
        slices each worker consumes from the trainer actor over NIXL and scatters
        them in. Bracketed by start_weight_update / finish_weight_update (which
        run initialize/finalize_layerwise_reload) — the driver calls this once
        per layer-aligned group between that pair.
        """
        if not getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError("start_weight_update must be called before update_weights_rdt.")

        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. Please set weight_transfer_config to enable weight transfer."
            )

        from vllm.config import set_current_vllm_config

        self._inject_rdt_model_device()
        typed_update_info = self.weight_transfer_engine.parse_update_info(update_info)
        model = self.model_runner.model

        with set_current_vllm_config(self.vllm_config), torch.device(self.device):
            self.weight_transfer_engine.receive_weights(
                typed_update_info,
                load_weights=model.load_weights,
            )

        torch.accelerator.synchronize()

    def finish_weight_update(self) -> None:
        """
        Finalize the current weight update.

        For checkpoint-format weights, runs layerwise postprocessing
        (quantization repacking, attention weight processing, etc.).
        Must be called after all update_weights_ipc calls are done.
        """
        if not getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError("start_weight_update must be called before finish_weight_update.")

        if self._skyrl_is_checkpoint_format:
            from vllm.config import set_current_vllm_config
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload,
            )

            model = self.model_runner.model
            with set_current_vllm_config(self.vllm_config), torch.device(self.device):
                finalize_layerwise_reload(model, self.model_config)

        self._skyrl_weight_update_active = False
        self._skyrl_is_checkpoint_format = True
