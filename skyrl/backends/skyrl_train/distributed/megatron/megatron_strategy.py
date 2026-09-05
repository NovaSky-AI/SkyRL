import hashlib
import os
import random
import re
import shutil
import tempfile
from typing import List, Optional, Union

import megatron.core.parallel_state as mpu
import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from torch import distributed as dist
from torch import optim
from torch.distributed.checkpoint import FileSystemReader
from transformers import PreTrainedTokenizer

from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
    load_megatron_grads_to_gpu,
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_grads_to_cpu,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)
from skyrl.backends.skyrl_train.distributed.strategy import DistributedStrategy
from skyrl.backends.skyrl_train.distributed.utils import ModelOrModelOptimPair
from skyrl.backends.skyrl_train.utils.io import io
from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
    MegatronModelWrapper,
)

# Seed offset per pipeline-parallel rank, matching Megatron's standard practice.
_PP_SEED_OFFSET = 100

# Process-wide queue of in-flight async checkpoint writes. Megatron-core has no
# global for this; It has to be
# module-level rather than an instance attribute because `_finalize_async_calls`
# is a staticmethod, invoked from a finalization callback with no strategy handle.
_async_calls: Optional[AsyncCallsQueue] = None


def _stage_async_request_to_host(async_request):
    """Run an async checkpoint request's GPU->host staging in *this* process.

    Megatron's ``PersistentAsyncCaller`` pickles the whole ``AsyncRequest`` to a
    spawned worker and calls ``preload_fn`` *there*, so the still-on-GPU shards
    cross the process boundary via CUDA IPC.

    That can break under ``expandable_segments:True`` -- which SkyRL turns on for
    training workers (see ``Worker._set_expandable_segments``). Expandable
    segments are ``cuMemCreate``-backed, and their shareable handle is a POSIX
    file descriptor rather than a copyable ``cudaIpcMemHandle_t``, so the
    importing process has to pull the fd out of the producer with
    ``pidfd_getfd``. That syscall requires ptrace-attach permission.
    The checkpoint worker is a child of the training rank,
    so it can error out with ``pidfd_getfd: Operation not permitted`` on machines
    with restricted ptrace permissions and the training rank
    then hangs forever on the preload barrier.

    Staging here avoids CUDA IPC altogether: only CPU tensors get pickled to the
    worker, and those cross by fd-passing over a unix socket, which needs no ptrace
    permission. Reference:
    https://github.com/NVIDIA/Megatron-LM/blob/b78cfd5279be41ced082d344e9380a09a146c458/megatron/core/dist_checkpointing/strategies/async_utils.py#L578-L584

    Takes and returns an ``AsyncRequest``; both the ``mcore`` and ``nvrx`` request types
    are named tuples with the same ``async_fn_args``/``preload_fn`` fields. The attribute
    access is deliberately unguarded so a future upstream change to that contract fails
    loudly here rather than silently restoring the hang.
    """
    if async_request.preload_fn is None:
        return async_request
    args = list(async_request.async_fn_args)
    # ``preload_fn`` stages ``async_fn_args[1]`` (the write buckets); Megatron
    # asserts the same arity in ``AsyncRequest.execute_sync``.
    assert len(args) == 3, f"expected 3 async_fn_args, got {len(args)}: {args!r}"
    args[1] = async_request.preload_fn()
    return async_request._replace(async_fn_args=tuple(args), preload_fn=None)


def _patched_update_fp32_params_by_new_state(self):
    """Monkeypatch for megatron-core HybridDeviceOptimizer._update_fp32_params_by_new_state.

    Upstream bug: params that are already fp32 are not present in
    ``self.param_to_fp32_param`` but the original code does an unconditional
    dict lookup, causing a ``KeyError`` during checkpoint loading with CPU
    offloading enabled.
    """
    if not self.param_update_in_fp32:
        return
    for param, v in self.state.items():
        if param not in self.param_to_fp32_param:
            continue
        fp32_param = self.param_to_fp32_param[param]
        fp32_param.data.copy_(v["master_param"])


_orig_load_parameter_state_from_dp_reshardable = DistributedOptimizer.load_parameter_state_from_dp_reshardable


def _patched_load_parameter_state_from_dp_reshardable(self, state_dict):
    """Wrapper around the original method that preserves the Adam step counter.

    Upstream bug: each bucket element carries a ``step`` tensor wrapped as a
    ``LocalNonpersistentObject``.  During the load-time round-trip this object
    picks up the *current* (stale) step count instead of the checkpoint value.
    The correct step is already set from ``param_groups`` by
    ``load_state_dict`` before this method runs, but the original
    ``_set_main_param_and_optimizer_states`` overwrites it with the stale one.
    ``load_parameter_state_from_fs_model_space`` already filters ``step``;
    here we save/restore it around the original method instead.
    """
    correct_step = None
    for v in self.optimizer.state.values():
        if "step" in v and isinstance(v["step"], torch.Tensor):
            correct_step = v["step"].clone()
            break

    _orig_load_parameter_state_from_dp_reshardable(self, state_dict)

    if correct_step is not None:
        for v in self.optimizer.state.values():
            if "step" in v and isinstance(v["step"], torch.Tensor):
                v["step"].copy_(correct_step)

    if isinstance(self.optimizer, HybridDeviceOptimizer):
        self.optimizer._sync_hdo_state_to_sub_optimizers()


# HybridDeviceOptimizer stores optimizer class references (cpu_optimizer_cls,
# gpu_optimizer_cls) in its ``defaults`` dict, which bleed into param_groups
# and get pickled into the "common" checkpoint file.  PyTorch 2.6+ defaults
# torch.load to weights_only=True, which rejects these class globals.
_safe_globals = [torch.optim.Adam, torch.optim.AdamW]
try:
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import (
        HybridDeviceOptimizer,
    )

    HybridDeviceOptimizer._update_fp32_params_by_new_state = _patched_update_fp32_params_by_new_state
    DistributedOptimizer.load_parameter_state_from_dp_reshardable = _patched_load_parameter_state_from_dp_reshardable

    try:
        from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam

        _safe_globals.append(FusedAdam)
    except ImportError:
        pass
except ImportError:
    pass
torch.serialization.add_safe_globals(_safe_globals)


class MegatronStrategy(DistributedStrategy):
    """
    The strategy for training with Megatron.
    """

    def __init__(
        self,
        megatron_config,
        optimizer_config=None,
        seed: int = 42,
        is_lora: bool = False,
        node_local_rank: int = 0,
    ) -> None:
        super().__init__()
        self.megatron_config = megatron_config
        self.optimizer_config = optimizer_config
        self.seed = seed
        self.hf_config = None  # Set by the megatron worker once configs are initialized.
        self.is_lora = is_lora
        self.node_local_rank = node_local_rank

        # NOTE: Set Megatron dist checkpoint async backend to persistent to avoid `os.fork()`-ing
        # short-lived background workers, which does not work well with Ray.
        global _async_calls
        _async_calls = AsyncCallsQueue(persistent=True)

    def set_seed(self, seed: int) -> None:
        # Vary seed by pipeline parallel rank so that different PP stages get
        # different dropout masks and stochastic noise (matches Megatron standard
        # practice).
        seed = seed + _PP_SEED_OFFSET * mpu.get_pipeline_model_parallel_rank()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.device_count() > 0:
            from megatron.core import tensor_parallel

            tensor_parallel.model_parallel_cuda_manual_seed(seed)

    def setup_distributed(self) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.megatron_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.megatron_config.pipeline_model_parallel_size,
            expert_model_parallel_size=self.megatron_config.expert_model_parallel_size,
            expert_tensor_parallel_size=self.megatron_config.expert_tensor_parallel_size,
            use_sharp=False,
            context_parallel_size=self.megatron_config.context_parallel_size,
            nccl_communicator_config_path=None,
        )
        self.set_seed(self.seed)
        self.world_size = dist.get_world_size()

    def offload_to_cpu(self, model, optimizer, offload_optimizer=True, offload_model=True):
        """
        Offload model weights and optimizer to CPU memory.

        The grad buffer belongs to the DDP-wrapped model, not the optimizer,
        so it is offloaded whenever ``offload_optimizer`` is requested even if
        ``optimizer is None`` (e.g. ``policy.inference_only_init=True`` flows).
        """
        if offload_model:
            offload_megatron_model_to_cpu(model)
        if offload_optimizer:
            offload_megatron_grads_to_cpu(model)
            if optimizer is not None:
                offload_megatron_optimizer(optimizer)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def backload_to_gpu(self, model, optimizer, backload_optimizer=True, backload_model=True):
        """Reload model weights back to GPU.

        See :meth:`offload_to_cpu` for why the grad-buffer half is decoupled
        from optimizer existence.
        """
        if backload_model:
            load_megatron_model_to_gpu(model)
        if backload_optimizer:
            load_megatron_grads_to_gpu(model)
            if optimizer is not None:
                load_megatron_optimizer(optimizer)
        torch.cuda.synchronize()

    def backward(self, loss: torch.Tensor, model, optimizer: optim.Optimizer, **kwargs) -> None:
        raise NotImplementedError()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model,
        scheduler,
        name="model",
        **kwargs,
    ) -> Optional[Float[torch.Tensor, "1"]]:
        """Perform optimizer step"""
        _, grad_norm, _ = optimizer.step()
        scheduler.step(1)
        optimizer.zero_grad()
        return grad_norm

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        raise NotImplementedError()

    @property
    def _dist_ckpt_optim_metadata(self) -> dict:
        if self.megatron_config.dist_ckpt_optim_fully_reshardable:
            return {"distrib_optim_sharding_type": "fully_reshardable"}
        return {"distrib_optim_sharding_type": "dp_reshardable"}

    @staticmethod
    def _ensure_optimizer_state_initialized(optimizer):
        """Ensure Adam optimizer state (exp_avg, exp_avg_sq) exists before checkpointing.

        Megatron's DistributedOptimizer lazily initializes optimizer state on the first
        training step. If we checkpoint before any step, the save template will lack
        exp_avg/exp_avg_sq entries, while the load template (which pre-allocates state)
        will include them, causing a key mismatch.
        """
        optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
        for opt in optimizers:
            init_fn = getattr(opt, "init_state_fn", None)
            inner_opt = getattr(opt, "optimizer", None)
            cfg = getattr(opt, "config", None)
            if init_fn is not None and inner_opt is not None and cfg is not None and len(inner_opt.state) == 0:
                init_fn(inner_opt, cfg)

    def save_checkpoint(
        self,
        model: MegatronModelWrapper,
        ckpt_dir: str,
        node_local_rank: int,
        optimizer: Optional[DistributedOptimizer] = None,
        scheduler: Optional[OptimizerParamScheduler] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        global _async_calls

        # Extract base model.
        model: List[nn.Module] = model.actor_module
        assert len(model) == 1, "Megatron virtual pipeline parallel is not yet supported"
        unwrapped_model = model[0]
        while hasattr(unwrapped_model, "module"):
            unwrapped_model = unwrapped_model.module

        # Create checkpoint directory if it doesn't exist.
        if node_local_rank == 0:
            io.makedirs(ckpt_dir, exist_ok=True)

        # All ranks wait for the checkpoint directory to be created before saving.
        dist.barrier()

        # Collect the sharded state dicts for model and optimizer, and full state dict for the scheduler.
        sharded_state_dict = {}
        model_sharded_state_dict = unwrapped_model.sharded_state_dict()
        if not self.is_lora:
            sharded_state_dict["model"] = model_sharded_state_dict
        if optimizer:
            self._ensure_optimizer_state_initialized(optimizer)
            sharded_state_dict["optimizer"] = optimizer.sharded_state_dict(
                model_sharded_state_dict,
                is_loading=False,
                metadata=self._dist_ckpt_optim_metadata,
            )
        if scheduler:
            sharded_state_dict["lr_scheduler"] = scheduler.state_dict()

        # Save RNG state.
        sharded_state_dict["rng"] = self.get_rng_state()

        # Save the checkpoint across ranks in parallel.
        save_strategy = get_default_save_sharded_strategy("torch_dist")
        save_strategy = FullyParallelSaveStrategyWrapper(
            save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
        )

        async_save = self.megatron_config.async_dist_ckpt_save
        if async_save and io.is_cloud_path(ckpt_dir):
            # local_work_dir uploads on context exit, which would race the still-running
            # background write. Async is only safe writing in-place to a local/shared FS.
            self.print(f"async_dist_ckpt_save unsupported for cloud path {ckpt_dir}; saving synchronously")
            async_save = False
        if async_save:
            # The queue holds one in-flight write; the prior checkpoint dir must be
            # fully written before this save reuses it.
            self._finalize_async_calls()

        with io.local_work_dir(ckpt_dir) as work_dir:
            async_save_request = dist_checkpointing.save(
                sharded_state_dict=sharded_state_dict,
                checkpoint_dir=work_dir,
                sharded_strategy=save_strategy,
                async_sharded_save=async_save,
                async_strategy=self.megatron_config.async_dist_ckpt_strategy,
                validate_access_integrity=True,
            )
            if async_save:
                # Shards are staged to host; the disk write runs in the background.
                if self.megatron_config.async_save_prestage_to_cpu:
                    # Keeps GPU tensors from crossing the process boundary, which the writer
                    # cannot always do -- see `_stage_async_request_to_host`.
                    async_save_request = _stage_async_request_to_host(async_save_request)
                _async_calls.schedule_async_request(async_save_request)
            else:
                assert async_save_request is None, "save() must not return a request when sync"

            # Only global rank 0 saves the Huggingface config and tokenizer.
            if self.is_rank_0():
                hf_dir = os.path.join(work_dir, "huggingface")
                self.save_hf_configs(self.hf_config, hf_dir, tokenizer)

        if self.is_lora:
            self._save_lora_adapters(unwrapped_model, ckpt_dir)

        dist.barrier()
        if not async_save:
            # Async path keeps the pending request alive in the queue until its finalize;
            # tearing it down here would orphan that write.
            _async_calls.close()
            _async_calls = AsyncCallsQueue(persistent=True)
        self.print(f"Checkpoint successfully saved to {ckpt_dir}")

    @staticmethod
    def _finalize_async_calls() -> None:
        # Finalization may run in a helper thread, whose current CUDA device defaults to 0.
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None and torch.cuda.is_available():
            torch.cuda.set_device(int(local_rank))
        if _async_calls is not None:
            _async_calls.maybe_finalize_async_calls(blocking=True)

    def finalize_pending_saves(self) -> None:
        """Block until any in-flight async checkpoint write completes.

        No-op when ``async_dist_ckpt_save`` is off or nothing is pending.
        """
        self._finalize_async_calls()

    def _get_rank_path(self, ckpt_dir):
        tp_rank = mpu.get_tensor_model_parallel_rank()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        cp_rank = mpu.get_context_parallel_rank()
        dp_rank = mpu.get_data_parallel_rank()
        ep_rank = mpu.get_expert_model_parallel_rank()
        etp_rank = mpu.get_expert_tensor_parallel_rank()

        return os.path.join(
            ckpt_dir, f"adapter_tp{tp_rank}_pp{pp_rank}_cp{cp_rank}_dp{dp_rank}_ep{ep_rank}_etp{etp_rank}.pt"
        )

    def _save_lora_adapters(self, model, ckpt_dir):
        """Save LoRA adapters to checkpoint."""
        if not self.is_lora:
            return

        assert isinstance(model, nn.Module), "Model must be a nn.Module"

        model_state_dict = {}
        for name, param in model.named_parameters():
            if ".adapter" in name.lower():
                model_state_dict[name] = param.data

        with io.local_work_dir(ckpt_dir) as work_dir:
            adapter_path = self._get_rank_path(work_dir)
            torch.save({"model_state_dict": model_state_dict}, adapter_path)
            self.print(f"Saved {len(model_state_dict)} LoRA adapter parameters to {adapter_path}")

    def load_checkpoint(
        self,
        model: MegatronModelWrapper,
        ckpt_dir: str,
        optimizer: Optional[DistributedOptimizer] = None,
        scheduler: Optional[OptimizerParamScheduler] = None,
        load_module_strict: bool = True,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ):
        if not ckpt_dir or not io.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        # A reload must observe a fully-written checkpoint.
        self.finalize_pending_saves()

        # Extract base model.
        model: List[nn.Module] = model.actor_module
        assert len(model) == 1, "Megatron virtual pipeline parallel is not yet supported"
        unwrapped_model = model[0]
        while hasattr(unwrapped_model, "module"):
            unwrapped_model = unwrapped_model.module

        # Extract sharded state dicts.
        sharded_state_dict = {}
        model_sharded_state_dict = unwrapped_model.sharded_state_dict()
        if not self.is_lora:
            sharded_state_dict["model"] = model_sharded_state_dict
        if optimizer and load_optimizer_states:
            sharded_state_dict["optimizer"] = optimizer.sharded_state_dict(
                model_sharded_state_dict,
                is_loading=True,
                metadata=self._dist_ckpt_optim_metadata,
            )
        if scheduler and load_lr_scheduler_states:
            sharded_state_dict["lr_scheduler"] = scheduler.state_dict()

        if io.is_cloud_path(ckpt_dir):
            state_dict = self._load_dist_checkpoint_from_cloud(ckpt_dir, sharded_state_dict)
        else:
            # Load from local filesystem with full parallel strategy.
            load_strategy = get_default_load_sharded_strategy(ckpt_dir)
            load_strategy = FullyParallelLoadStrategyWrapper(
                load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
            )
            state_dict = dist_checkpointing.load(
                sharded_state_dict=sharded_state_dict, checkpoint_dir=ckpt_dir, sharded_strategy=load_strategy
            )

        if not self.is_lora:
            # Load the model, optimizer, and scheduler state dicts.
            assert (
                "model" in state_dict
            ), f"Model state dict not found in checkpoint loaded from {ckpt_dir}. Available keys: {state_dict.keys()}"
            model[0].load_state_dict(state_dict["model"], strict=load_module_strict)
            self.print("Loaded model state dict.")
        else:
            self._load_lora_adapters(unwrapped_model, ckpt_dir)

        if optimizer and load_optimizer_states:
            assert (
                "optimizer" in state_dict
            ), f"Optimizer state dict not found in checkpoint loaded from {ckpt_dir}. Available keys: {state_dict.keys()}"
            optimizer.load_state_dict(state_dict["optimizer"])
            self.print("Loaded optimizer state dict.")

        if scheduler and load_lr_scheduler_states:
            assert (
                "lr_scheduler" in state_dict
            ), f"LR scheduler state dict not found in checkpoint loaded from {ckpt_dir}. Available keys: {state_dict.keys()}"
            scheduler.load_state_dict(state_dict["lr_scheduler"])
            self.print("Loaded LR scheduler state dict.")

        # Load RNG state, if present.
        if "rng" in state_dict:
            self.load_rng_state(state_dict["rng"])

        return ckpt_dir, {}

    _SHARD_FILE_PATTERN = re.compile(r"__(\d+)_\d+\.distcp$")

    def _load_dist_checkpoint_from_cloud(self, ckpt_dir: str, sharded_state_dict: dict) -> dict:
        """Download checkpoint shards from cloud storage with per-rank parallelism.

        All ranks on the same node share a single node-local directory.
        Each node downloads its ranks' assigned shards plus files backing the
        common state, rather than downloading the full checkpoint on every rank.

        Local rank 0 creates the directory and downloads common metadata files
        and any shard containing the common state read independently by every rank.
        After a barrier, all shard files are present and every rank can load.

        Does not currently support flexible trainer resharding.
        """
        global_rank = dist.get_rank()
        node_local_rank = self.node_local_rank

        dir_hash = hashlib.md5(ckpt_dir.encode()).hexdigest()[:12]
        local_dir = os.path.join(tempfile.gettempdir(), f"skyrl_ckpt_load_{dir_hash}")

        try:
            all_entries = io.list_dir(ckpt_dir)

            # Local rank 0: create dir and download common/metadata files.
            if node_local_rank == 0:
                if os.path.exists(local_dir):
                    shutil.rmtree(local_dir)
                os.makedirs(local_dir)

                for entry in all_entries:
                    name = entry.rstrip("/").split("/")[-1]
                    if not name:
                        continue
                    if self._SHARD_FILE_PATTERN.search(name):
                        continue
                    cloud_entry = ckpt_dir.rstrip("/") + "/" + name
                    if io.isdir(cloud_entry):
                        continue
                    io.download_file(cloud_entry, os.path.join(local_dir, name))

                if not os.path.isfile(os.path.join(local_dir, "common.pt")):
                    # Current Megatron checkpoints embed common state in a DCP
                    # shard. Every rank reads it before distributed shard loading.
                    common_key = dist_checkpointing.ShardedObject("common_state", None, (1,), (0,)).unique_key
                    metadata = FileSystemReader(local_dir).read_metadata()
                    common_shards = {
                        storage.relative_path
                        for index, storage in metadata.storage_data.items()
                        if index.fqn == common_key
                    }
                    for relative_path in sorted(common_shards):
                        local_path = os.path.join(local_dir, relative_path)
                        if not os.path.isfile(local_path):
                            os.makedirs(os.path.dirname(local_path), exist_ok=True)
                            io.download_file(f"{ckpt_dir.rstrip('/')}/{relative_path}", local_path)

            # Wait for the directory and common files to be ready.
            dist.barrier()

            # Each rank downloads its own shard file(s) into the shared dir.
            for entry in all_entries:
                name = entry.rstrip("/").split("/")[-1]
                if not name:
                    continue
                match = self._SHARD_FILE_PATTERN.search(name)
                if match and int(match.group(1)) == global_rank:
                    cloud_path = ckpt_dir.rstrip("/") + "/" + name
                    local_path = os.path.join(local_dir, name)
                    if not os.path.isfile(local_path):
                        io.download_file(cloud_path, local_path)

            # Wait for all ranks to finish downloading their shards.
            dist.barrier()

            self.print(f"All ranks downloaded checkpoint shards from {ckpt_dir}")

            load_strategy = get_default_load_sharded_strategy(local_dir)
            load_strategy = FullyParallelLoadStrategyWrapper(
                load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
            )
            return dist_checkpointing.load(
                sharded_state_dict=sharded_state_dict,
                checkpoint_dir=local_dir,
                sharded_strategy=load_strategy,
            )
        finally:
            dist.barrier()
            if node_local_rank == 0:
                shutil.rmtree(local_dir, ignore_errors=True)

    def _load_lora_adapters(self, model, ckpt_dir):
        """Load LoRA adapters from checkpoint."""
        # TODO (erictang000): Update this logic once LoRA checkpointing is upstreamed to Megatron-Bridge
        if not self.is_lora:
            return

        assert isinstance(model, nn.Module), "Model must be a nn.Module"

        with io.local_read_dir(ckpt_dir) as read_dir:
            adapter_path = self._get_rank_path(read_dir)
            state_dict = torch.load(adapter_path, map_location="cpu")
            _, unexpected = model.load_state_dict(state_dict["model_state_dict"], strict=False)
            if len(unexpected) > 0:
                raise ValueError(f"Unexpected keys in LoRA adapter state dict: {unexpected}")
            self.print(f"Loaded {len(state_dict['model_state_dict'])} LoRA adapters from {adapter_path}.")

    def save_hf_model(
        self,
        bridge,
        model: MegatronModelWrapper,
        output_dir: str,
        tokenizer=None,
        *,
        distributed_save: bool = False,
        save_every_n_ranks: int = 1,
        **kwargs,
    ) -> None:
        if distributed_save and io.is_cloud_path(output_dir):
            raise ValueError(
                f"distributed_save=True is incompatible with cloud paths (got {output_dir}). "
                "Please use a shared filesystem path for distributed_save."
            )
        # Create checkpoint directory if it doesn't exist.
        if self.is_rank_0():
            io.makedirs(output_dir, exist_ok=True)
        dist.barrier()

        # All ranks call into bridge.
        with io.local_work_dir(output_dir) as work_dir:
            # strict=False is required for partial exports (e.g. language_model_only
            # on a Qwen3.5 VL checkpoint, whose shards co-mingle vision and text
            # weights): the bridge writes a shard only once all its keys are yielded,
            # so strict=True silently writes zero weights. No-op for complete exports.
            #
            # distributed_save fans the shard writes across ranks (one saver per
            # save_every_n_ranks) instead of serializing them on rank 0 -- the same
            # standard HF sharded layout, just written in parallel.
            bridge.save_hf_weights(
                model.actor_module,
                work_dir,
                strict=False,
                distributed_save=distributed_save,
                save_every_n_ranks=save_every_n_ranks,
            )
            self.print(f"Successfully saved HF safetensors model to {output_dir}")

            # Only rank 0 saves the Huggingface config and tokenizer.
            if self.is_rank_0():
                # Preserve any custom modeling artifacts (e.g. modeling_*.py,
                # special_tokens_map.json, auto_map-referenced files) that
                # trust_remote_code models depend on. save_hf_configs below
                # overwrites config.json/tokenizer files with the strategy's
                # current view, but save_artifacts is required to copy the
                # custom Python modules and other artifacts that
                # save_pretrained() alone does not emit.
                bridge.hf_pretrained.save_artifacts(work_dir)
                self.save_hf_configs(self.hf_config, work_dir, tokenizer)
                self.print(f"Successfully saved HF config and tokenizer to {output_dir}")

        dist.barrier()
