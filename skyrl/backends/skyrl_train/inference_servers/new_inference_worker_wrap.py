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

from functools import cache
from typing import Any

import torch

from skyrl.backends.skyrl_train.inference_servers.layerwise_reload import (
    LayerwiseReloadWorkerMixin,
)
from skyrl.backends.skyrl_train.weight_sync.base import cuda_uuid_to_str
from skyrl.backends.skyrl_train.weight_sync.serialization import (
    LEGACY_BATCHED_MOE_PREFIX,
    MODEL_QUANTIZATION_SPECS,
    SERIALIZED_WEIGHT_PREFIX,
    SERIALIZED_WEIGHT_STRATEGIES,
    get_serialized_weight_strategy,
)

VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS = f"{__name__}.NewInferenceWorkerWrap"


def _decode_serialized_name(name: str) -> tuple[str | None, str] | None:
    if name.startswith(LEGACY_BATCHED_MOE_PREFIX):
        return None, name.removeprefix(LEGACY_BATCHED_MOE_PREFIX)
    if not name.startswith(SERIALIZED_WEIGHT_PREFIX):
        return None
    payload = name.removeprefix(SERIALIZED_WEIGHT_PREFIX)
    mode, separator, checkpoint_name = payload.partition(":")
    if not separator or not mode or not checkpoint_name:
        raise ValueError(f"Invalid serialized weight name {name!r}")
    return mode, checkpoint_name


@cache
def _get_receiver_strategy(mode: str):
    return get_serialized_weight_strategy(mode)


def _resolve_serialized_receiver_target(mode: str | None, checkpoint_name: str) -> tuple[str, str] | None:
    modes = (mode,) if mode is not None else tuple(SERIALIZED_WEIGHT_STRATEGIES)
    matches = {
        target
        for candidate_mode in modes
        if (
            target := _get_receiver_strategy(candidate_mode).receiver_target(
                checkpoint_name,
                MODEL_QUANTIZATION_SPECS,
            )
        )
        is not None
    }
    if len(matches) > 1:
        raise ValueError(f"Ambiguous serialized weight receiver target for {checkpoint_name!r}")
    return next(iter(matches), None)


def _map_hf_weight_name(model: torch.nn.Module, name: str) -> str:
    """Apply a top-level vLLM model's HF-to-runtime prefix mapping."""
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is None:
        return name
    mapped = mapper.apply_list([name])
    if len(mapped) != 1:
        raise ValueError(f"Unable to map batched MoE checkpoint name {name!r}")
    return mapped[0]


def _load_serialized_moe_tensor(
    model: torch.nn.Module,
    params_dict: dict[str, torch.nn.Parameter],
    wire_name: str,
    decoded_name: tuple[str | None, str],
    loaded_weight: torch.Tensor,
) -> bool:
    """Load one serialized expert tensor through FusedMoE's loader.

    Marked tensors must resolve successfully so a protocol mismatch cannot
    silently leave stale rollout weights behind.
    """
    if loaded_weight.ndim != 3:
        raise ValueError(
            f"Batched MoE wire tensor must be 3D, got name={wire_name!r}, shape={tuple(loaded_weight.shape)}"
        )

    mode, checkpoint_name = decoded_name
    mapped_name = _map_hf_weight_name(model, checkpoint_name)
    receiver_target = _resolve_serialized_receiver_target(mode, mapped_name)
    if receiver_target is None:
        raise ValueError(f"Unsupported batched MoE wire tensor name {wire_name!r}")
    target_name, shard_id = receiver_target
    if target_name not in params_dict:
        raise ValueError(f"Batched MoE target parameter {target_name!r} was not found for wire tensor {wire_name!r}")

    param = params_dict[target_name]
    weight_loader = getattr(param, "weight_loader", None)
    if weight_loader is None or not getattr(weight_loader, "supports_moe_loading", False):
        # Layerwise reload wraps the loader with functools.wraps, which copies
        # this marker from FusedMoE.weight_loader onto the wrapper.
        raise ValueError(f"Parameter {target_name!r} does not expose a FusedMoE weight loader")

    if param.shape[0] == loaded_weight.shape[0]:
        success = weight_loader(
            param,
            loaded_weight,
            target_name,
            shard_id=shard_id,
            expert_id=0,
            return_success=True,
        )
        if not success:
            raise ValueError(f"Fused loading failed for batched MoE tensor {wire_name!r}")
        return True

    # Expert-parallel vLLM keeps only a subset locally. Retain the compact wire
    # format, but let the loader map global expert IDs one view at a time.
    loaded_any = False
    for expert_id, expert_weight in enumerate(loaded_weight.unbind(0)):
        loaded_any = (
            bool(
                weight_loader(
                    param,
                    expert_weight,
                    target_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
            )
            or loaded_any
        )
    if not loaded_any:
        raise ValueError(f"No local expert accepted batched MoE tensor {wire_name!r}")
    return True


def _load_batched_moe_fp8_tensor(
    model: torch.nn.Module,
    params_dict: dict[str, torch.nn.Parameter],
    wire_name: str,
    loaded_weight: torch.Tensor,
) -> bool:
    """Compatibility wrapper for serialized MoE tensors."""

    decoded_name = _decode_serialized_name(wire_name)
    if decoded_name is None:
        return False
    return _load_serialized_moe_tensor(model, params_dict, wire_name, decoded_name, loaded_weight)


def _load_checkpoint_weights(model: torch.nn.Module, weights: list[tuple[str, torch.Tensor]]) -> Any:
    """Load ordinary HF tensors plus SkyRL's compact batched-MoE tensors."""
    params_dict: dict[str, torch.nn.Parameter] | None = None
    ordinary_weights: list[tuple[str, torch.Tensor]] = []
    for name, weight in weights:
        decoded_name = _decode_serialized_name(name)
        if decoded_name is not None:
            if params_dict is None:
                params_dict = dict(model.named_parameters())
            _load_serialized_moe_tensor(model, params_dict, name, decoded_name, weight)
        else:
            ordinary_weights.append((name, weight))
    if ordinary_weights:
        return model.load_weights(weights=ordinary_weights)
    return set()


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
                "Weight transfer not configured. Please set weight_transfer_config to enable weight transfer."
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
        physical_gpu_id = cuda_uuid_to_str(torch.cuda.get_device_properties(device_index).uuid)
        if physical_gpu_id not in handles:
            raise ValueError(f"IPC handle not found for GPU UUID {physical_gpu_id}. Available: {list(handles)}")
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
                _load_checkpoint_weights(model, weights)
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
            loaded = _load_checkpoint_weights(model, weights)
            _reload_spec_decode_drafter(self.model_runner, weights)
            return loaded

        with set_current_vllm_config(self.vllm_config), torch.device(self.device):
            self.weight_transfer_engine.receive_weights(
                typed_update_info,
                load_weights=_load_weights,
            )

        torch.accelerator.synchronize()
