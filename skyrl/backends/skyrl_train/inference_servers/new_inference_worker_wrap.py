"""vLLM worker extension for the two weight-sync things an engine cannot do.

Weight transfer itself does not route through here: the receive path is an engine
subclass (``weight_sync/weight_receivers.py``, ``weight_sync/delta/engine.py``,
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

import logging
from typing import TYPE_CHECKING, Any

import torch

from skyrl.backends.skyrl_train.weight_sync.fp8 import (
    SKYRL_BATCHED_MOE_FP8_PREFIX,
    batched_moe_wire_targets,
)

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
    from skyrl.backends.skyrl_train.weight_sync.register import (
        register_receive_engines,
    )

    register_receive_engines()
except ModuleNotFoundError:
    logging.getLogger(__name__).debug(
        "skyrl.weight_sync.register not importable; receive engines are unregistered "
        "and any weight sync in this worker will fail at create_engine.",
        exc_info=True,
    )


# Runs in every vLLM worker process before the model is loaded, so the
# LoRA-capability declaration is in place for the supports_lora() gate.
from skyrl.backends.skyrl_train.patches.vllm_kimi_k25_lora import (  # noqa: E402
    apply_kimi_k25_lora_patch,
)

apply_kimi_k25_lora_patch()


VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS = f"{__name__}.NewInferenceWorkerWrap"

# Checkpoint-name suffix -> (fused vLLM parameter suffix, FusedMoE shard id).
# The model specs own this mapping so sender and receiver cannot drift.
_BATCHED_MOE_TARGETS = batched_moe_wire_targets()


def _map_hf_weight_name(model: torch.nn.Module, name: str) -> str:
    """Apply a top-level vLLM model's HF-to-runtime prefix mapping."""
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is None:
        return name
    mapped = mapper.apply_list([name])
    if len(mapped) != 1:
        raise ValueError(f"Unable to map batched MoE checkpoint name {name!r}")
    return mapped[0]


def _load_batched_moe_fp8_tensor(
    model: torch.nn.Module,
    params_dict: dict[str, torch.nn.Parameter],
    wire_name: str,
    loaded_weight: torch.Tensor,
) -> bool:
    """Load one expert-batched FP8 weight or scale through FusedMoE's loader."""
    if not wire_name.startswith(SKYRL_BATCHED_MOE_FP8_PREFIX):
        return False
    if loaded_weight.ndim != 3:
        raise ValueError(
            f"Batched MoE wire tensor must be 3D, got name={wire_name!r}, shape={tuple(loaded_weight.shape)}"
        )

    checkpoint_name = wire_name.removeprefix(SKYRL_BATCHED_MOE_FP8_PREFIX)
    mapped_name = _map_hf_weight_name(model, checkpoint_name)
    target_name = None
    shard_id = None
    for checkpoint_suffix, (target_suffix, candidate_shard_id) in _BATCHED_MOE_TARGETS.items():
        if mapped_name.endswith(checkpoint_suffix):
            target_name = mapped_name[: -len(checkpoint_suffix)] + target_suffix
            shard_id = candidate_shard_id
            break
    if target_name is None or shard_id is None:
        raise ValueError(f"Unsupported batched MoE wire tensor name {wire_name!r}")
    if target_name not in params_dict:
        module_path, _, param_leaf = target_name.rpartition(".")
        nested_name = f"{module_path}.routed_experts.{param_leaf}"
        if nested_name not in params_dict:
            raise ValueError(
                f"Batched MoE target parameter was not found for wire tensor {wire_name!r}: "
                f"tried {target_name!r} and {nested_name!r}"
            )
        target_name = nested_name

    param = params_dict[target_name]
    weight_loader = getattr(param, "weight_loader", None)
    if weight_loader is None or not getattr(weight_loader, "supports_moe_loading", False):
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

    loaded_any = False
    for expert_id, expert_weight in enumerate(loaded_weight.unbind(0)):
        loaded_any = bool(
            weight_loader(
                param,
                expert_weight,
                target_name,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=True,
            )
        ) or loaded_any
    if not loaded_any:
        raise ValueError(f"No local expert accepted batched MoE tensor {wire_name!r}")
    return True


def _load_checkpoint_weights(
    model: torch.nn.Module,
    weights: list[tuple[str, torch.Tensor]],
    **kwargs: Any,
) -> Any:
    """Load ordinary checkpoint tensors and compact batched-MoE FP8 tensors."""
    params_dict: dict[str, torch.nn.Parameter] | None = None
    ordinary_weights: list[tuple[str, torch.Tensor]] = []
    for name, weight in weights:
        if name.startswith(SKYRL_BATCHED_MOE_FP8_PREFIX):
            if params_dict is None:
                params_dict = dict(model.named_parameters())
            _load_batched_moe_fp8_tensor(model, params_dict, name, weight)
        else:
            ordinary_weights.append((name, weight))
    if ordinary_weights:
        return model.load_weights(weights=ordinary_weights, **kwargs)
    return set()


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
            raise RuntimeError("Weight transfer not configured: set weight_transfer_config on the engine.")
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
