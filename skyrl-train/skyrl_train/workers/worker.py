import asyncio
import logging
import os
import socket
from collections import defaultdict
from ctypes import CDLL, POINTER, Structure, c_char_p, c_int, c_ulong, c_void_p
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import ray
import torch
import torch.distributed
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from ray import ObjectRef
from ray.util.placement_group import (
    PlacementGroup,
    PlacementGroupSchedulingStrategy,
    placement_group,
    placement_group_table,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel

from skyrl_train.dataset.replay_buffer import Experience
from skyrl_train.distributed.dispatch import (
    ActorInfo,
    Dispatch,
    DispatchRegistry,
    MeshRank,
)
from skyrl_train.distributed.strategy import DistributedStrategy
from skyrl_train.distributed.ulysses import (
    apply_monkey_patch,
    set_ulysses_sequence_parallel_group,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.utils import (
    get_ray_pg_ready_with_timeout,
    get_reordered_bundle_indices,
    ray_noset_visible_devices,
)
from skyrl_train.env_vars import (
    SKYRL_RAY_PG_TIMEOUT_IN_S,
    SKYRL_WORKER_NCCL_TIMEOUT_IN_S,
)
from skyrl_train.utils.io import io
from skyrl_train.utils.ppo_utils import (
    PolicyLossRegistry,
    compute_approx_kl,
    masked_mean,
    ppo_critic_loss,
)
from skyrl_train.utils.utils import configure_ray_worker_logging
from skyrl_train.workers.worker_utils import BatchIterator, reduce_metrics

_SET_AFFINITY = False


# Adapted from OpenRLHF: https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ray/launcher.py#L17
class DistributedTorchRayActor:
    def __init__(
        self, world_size, rank, local_rank, master_addr, master_port, sequence_parallel_size, record_memory=False
    ):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the CUDA_VISIBLE_DEVICES
        # environment variable for each actor, so always set device to 0
        # os.environ["LOCAL_RANK"] = str(self._local_rank)
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"
        self.sequence_parallel_size: int = sequence_parallel_size

        self.record_memory = record_memory
        if record_memory:
            torch.cuda.memory._record_memory_history()
        configure_ray_worker_logging()

    def get_node_local_rank(self):
        return self._local_rank

    def init_worker_process_group(self):
        if not torch.distributed.is_initialized():
            # Default torch dist pg init timeout is 10 minutes (600 seconds)
            torch.distributed.init_process_group(
                backend="nccl", timeout=timedelta(seconds=SKYRL_WORKER_NCCL_TIMEOUT_IN_S)
            )

        # setup device mesh
        # TODO: Support TP / PP for additional backends
        # NOTE (sumanthrh): Device mesh and mesh rank are rank specific attributes. For the current way the strategy is defined, it is only meant to interact with worker state; not hold worker state. Thus, this should live outside the strategy object.
        # This device mesh can be common across all the strategies we use
        dp_size = self._world_size // self.sequence_parallel_size
        device_mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda", mesh_shape=(dp_size, self.sequence_parallel_size), mesh_dim_names=("dp", "sp")
        )
        self.device_mesh = device_mesh
        self.mesh_rank = MeshRank(
            dp=self.device_mesh.get_local_rank(mesh_dim="dp"),
            sp=self.device_mesh.get_local_rank(mesh_dim="sp"),
            tp=0,
            pp=0,
            world_size=self._world_size,
            dp_size=self.device_mesh.size(0),
            pp_size=1,
        )

    def _seq_parallel_monkey_patch(self, model: PreTrainedModel, use_parent_class: bool = False):
        # NOTE (sumanthrh): This sets a global variable that is used during the forward pass for sequence parallelism
        # This works because each worker is it's own process and thus different worker types are isolated
        # TODO (sumanthrh): We should re-visit this and see if we should adopt a context-manager pattern for sequence parallelism
        if self.sequence_parallel_size > 1:
            set_ulysses_sequence_parallel_group(self.device_mesh["sp"].get_group())
            apply_monkey_patch(
                model=model, ulysses_sp_size=self.sequence_parallel_size, use_parent_class=use_parent_class
            )

    def get_mesh_rank(self):
        return self.mesh_rank

    def get_gpu_id(self):
        return ray.get_gpu_ids()[0]

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    def get_ray_node_id(self):
        return ray.get_runtime_context().get_node_id()

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port

    # TODO(tgriggs): For numa affinity, pass in the Worker._local_rank for the second arg here. Distinguish 'rank' and 'local_rank' differ here.
    def _set_numa_affinity(self, rank):
        def local_rank_to_real_gpu_id(local_rank):
            cuda_visible_devices = [
                int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
            ]
            return cuda_visible_devices[local_rank]

        rank = local_rank_to_real_gpu_id(rank)

        global _SET_AFFINITY
        if _SET_AFFINITY:
            return

        from ctypes.util import find_library

        class bitmask_t(Structure):
            _fields_ = [
                ("size", c_ulong),
                ("maskp", POINTER(c_ulong)),
            ]

        try:
            LIBNUMA = CDLL(find_library("numa"))
        except Exception as e:
            logger.error(f"Skipping NUMA affinity setup because libnuma is not installed: {e}")
            _SET_AFFINITY = True
            return

        LIBNUMA.numa_parse_nodestring.argtypes = [c_char_p]
        LIBNUMA.numa_parse_nodestring.restype = POINTER(bitmask_t)
        LIBNUMA.numa_run_on_node_mask.argtypes = [POINTER(bitmask_t)]
        LIBNUMA.numa_run_on_node_mask.restype = c_int
        LIBNUMA.numa_set_membind.argtypes = [POINTER(bitmask_t)]
        LIBNUMA.numa_set_membind.restype = c_void_p
        LIBNUMA.numa_num_configured_nodes.argtypes = []
        LIBNUMA.numa_num_configured_nodes.restype = c_int

        def numa_bind(nid: int):
            bitmask = LIBNUMA.numa_parse_nodestring(bytes(str(nid), "ascii"))
            LIBNUMA.numa_run_on_node_mask(bitmask)
            LIBNUMA.numa_set_membind(bitmask)

        numa_nodes = LIBNUMA.numa_num_configured_nodes()
        if numa_nodes <= 0:
            numa_nodes = 1
        num_gpu_pre_numa_node = max(1, 8 // numa_nodes)
        target_nid = min(numa_nodes - 1, self._local_rank // num_gpu_pre_numa_node)
        numa_bind(target_nid)
        _SET_AFFINITY = True


class Worker(DistributedTorchRayActor):
    def __init__(self, cfg: DictConfig, *args, **kwargs):
        from skyrl_train.weight_sync import get_transfer_strategy_cls

        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self._transfer_strategy_cls = get_transfer_strategy_cls(self.cfg)

    def init_model(self, *args, **kwargs):
        """Initialize worker state (model, and optimizer if applicable) on worker."""
        raise NotImplementedError()

    def empty_cache(self) -> None:
        """Empty GPU memory cache on Worker's CUDA device"""
        torch.cuda.empty_cache()

    def offload_to_cpu(self, pin_memory=True, non_blocking=True):
        """Offload all worker state to CPU.

        After this function runs, only temporary reserved memory and torch's pre-loaded cuda kernels (~ GB) will remain

        Args:
            pin_memory: Whether to use pinned/ paged-locked memory on CPU
            non_blocking: Whether the operation is non-blocking
        """
        raise NotImplementedError()

    def backload_to_gpu(self, non_blocking=True):
        """Backload worker state to GPU

        Args:
            non_blocking: Whether the operation is non-blocking
        """
        raise NotImplementedError()

    def get_cuda_memory(self) -> Dict[str, Any]:
        """Get CUDA memory usage on worker's CUDA device."""
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "free": free,
            "total": total,
        }

    def save_memory_snapshot(self, tag: str = ""):
        """Save a snapshot of memory usage on the Worker's CUDA device.

        No-ops if record_memory is False.

        Args:
            tag: Label for the snapshot (e.g., "forward_backward", "optim_step")

        .. note::
            This function should be called on all the ranks in the worker group simultaneously.
        """
        if not self.record_memory:
            return

        # Track snapshot count for unique filenames
        if not hasattr(self, "_snapshot_count"):
            self._snapshot_count = 0
        self._snapshot_count += 1

        rank = torch.distributed.get_rank()
        save_path = os.path.join(self.cfg.trainer.ckpt_path, "memory_snapshots")
        if self._local_rank == 0 and not io.exists(save_path):
            io.makedirs(save_path, exist_ok=True)
        torch.distributed.barrier()

        tag_str = f"_{tag}" if tag else ""
        file_name = f"rank_{rank}{tag_str}_{self._snapshot_count}.pickle"
        record_memory_path = os.path.join(save_path, file_name)
        if io.exists(record_memory_path):
            # seeing issues if we don't remove the file first
            io.remove(record_memory_path)
        torch.cuda.memory._dump_snapshot(record_memory_path)

    async def init_weight_sync_state(self, inference_engine_client: InferenceEngineClient):
        """Initialize state for weight syncing with Inference Engine Client

        Creates init info and sender, then sends init info to inference engines
        so they can create receivers.

        .. note::
            This function should be called on all the ranks in the worker group simultaneously.
        """

        assert inference_engine_client is not None

        # Create init info on all ranks (it's deterministic from cfg)
        init_info = self._transfer_strategy_cls.create_init_info(self.cfg)

        # Create sender on all ranks
        # Strategy implementations may have different logic for different ranks
        tasks = [
            asyncio.to_thread(
                self._transfer_strategy_cls.create_sender,
                init_info=init_info,
                inference_client=inference_engine_client,
            ),
        ]

        # Only rank 0 initializes receivers on inference engines
        # NOTE: For broadcast strategy, sender and receiver init must run concurrently
        # because both need to join the same process group to avoid deadlock
        if torch.distributed.get_rank() == 0:
            tasks.append(inference_engine_client.init_weight_update_communicator(init_info))

        results = await asyncio.gather(*tasks)
        self._weight_transfer_sender = results[0]  # sender is always first task

        # # Register signal handlers for termination only on rank 0
        # NOTE (sumanthrh): This doesn't work yet, and is thus commented out.
        # The better way is to just have this specified in __del__, but there is
        # no guarattee that __del__ will be called in general. Ray also doesn't
        # explictly call __del__ when the actor shuts down.
        # It's commented out so that we can fix this in the future.
        # atexit.register(self._handle_termination)

        torch.distributed.barrier()

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on the input batch in inference mode.

        This is a wrapper around `_forward_micro_batch` that runs in micro batches of `cfg.trainer.micro_forward_batch_size_per_gpu`.
        """
        # run in micro batches of cfg.trainer.micro_forward_batch_size_per_gpu
        # TODO (sumanthrh): this can be in the policy/critic impl if the micro batch size can be specific to policy, critic, etc.
        micro_batches = data.chunk(self.cfg.trainer.micro_forward_batch_size_per_gpu)

        outputs = []
        for micro_batch in micro_batches:
            outputs.append(self._forward_micro_batch(micro_batch))
        output = TrainingOutputBatch.cat(outputs)
        if output.device is not None and output.device != torch.device("cpu"):
            output = output.to("cpu")
        return output

    def _forward_micro_batch(self, micro_batch: TrainingInputBatch) -> TrainingOutputBatch:
        raise NotImplementedError()


# adapted from OpenReasonerZero: https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/orz/ppo/actors.py
class PPORayActorGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        cfg: config object for workers
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        ray_actor_type (Type[Worker]): PPO model type that this actor group serve on.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        cfg,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[Worker],
        pg: Optional[PlacementGroup] = None,
        num_gpus_per_actor: float = 1.0,
        resources: Optional[Dict[str, float]] = None,
        num_resources_per_node: Optional[int] = None,
        colocate_all: bool = False,
        sequence_parallel_size: int = 1,
        record_memory: bool = False,
    ) -> None:
        self.cfg = cfg
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        self.colocate_all = colocate_all
        self.sequence_parallel_size = sequence_parallel_size
        self.record_memory = record_memory
        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg: Optional[PlacementGroup], num_gpus_per_actor: float):
        """Initialize Ray actors in the worker group.

        Args:
            pg: The placement group for the worker group
            num_gpus_per_actor: The number of gpus to allocate per actor.
        """
        world_size = self._num_nodes * self._num_gpus_per_node
        if self.colocate_all:
            assert (
                pg is not None
            ), "if colocate_all is True, the shared placement group must be provided to PPORayActorGroup"
            pg_data = placement_group_table(pg)
            assert (
                len(pg_data["bundles"]) == world_size
            ), "if colocate_all is True, the number of bundles in the shared placement group must match the world size"

        reordered_bundle_indices = []
        if pg is not None:
            pg_data = placement_group_table(pg)
            should_reorder_bundles = len(pg_data["bundles"]) == world_size
            if should_reorder_bundles:
                reordered_bundle_indices = get_reordered_bundle_indices(pg)

        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [{"GPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node} for _ in range(self._num_nodes)]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node

            pg = placement_group(bundles, strategy="PACK")
            get_ray_pg_ready_with_timeout(pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
        if pg:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[0] if reordered_bundle_indices else 0,
                ),
            ).remote(
                cfg=self.cfg,
                world_size=world_size,
                rank=0,
                local_rank=0,
                master_addr=None,
                master_port=None,
                sequence_parallel_size=self.sequence_parallel_size,
                record_memory=self.record_memory,
            )
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(
                cfg=self.cfg,
                world_size=world_size,
                rank=0,
                local_rank=0,
                master_addr=None,
                master_port=None,
                sequence_parallel_size=self.sequence_parallel_size,
                record_memory=self.record_memory,
            )
        self._actor_handlers = [master_actor]
        # Create worker actors
        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                local_rank = rank % self._num_gpus_per_node

                if pg:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=(
                                reordered_bundle_indices[rank]
                                if reordered_bundle_indices
                                else rank // self._num_gpus_per_node
                            ),
                        ),
                    ).remote(
                        cfg=self.cfg,
                        world_size=world_size,
                        rank=rank,
                        local_rank=local_rank,
                        master_addr=master_addr,
                        master_port=master_port,
                        sequence_parallel_size=self.sequence_parallel_size,
                        record_memory=self.record_memory,
                    )
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                    ).remote(
                        cfg=self.cfg,
                        world_size=world_size,
                        rank=rank,
                        local_rank=local_rank,
                        master_addr=master_addr,
                        master_port=master_port,
                        sequence_parallel_size=self.sequence_parallel_size,
                        record_memory=self.record_memory,
                    )
                self._actor_handlers.append(worker_actor)

        # Initialize process group
        logger.info("Initializing process group for RayActorGroup")
        ray.get([actor.init_worker_process_group.remote() for actor in self._actor_handlers])
        logger.info("Initialized process group for RayActorGroup")
        self.actor_infos = [ActorInfo(actor, ray.get(actor.get_mesh_rank.remote())) for actor in self._actor_handlers]
        logger.info(f"Mesh Ranks: {[actor_info.rank for actor_info in self.actor_infos]}")

    def async_init_model(
        self,
        *args,
        **kwargs,
    ) -> List[ObjectRef]:
        """Asynchronously initialize worker state (model, and optimizer if applicable) from model path
        on all the workers.

        Returns:
            A list of ray object refs.
        """
        return [actor.init_model.remote(*args, **kwargs) for actor in self._actor_handlers]

    def offload_to_cpu(self, nonblocking=False, offload_optimizer=True, offload_model=True):
        """Offload all worker state to CPU.

        Args:
            nonblocking: Whether this operation is synchronous or asynchronous.
            If `nonblocking=True`, then the function returns a list of object refs.
        """
        refs = [
            actor.offload_to_cpu.remote(offload_optimizer=offload_optimizer, offload_model=offload_model)
            for actor in self._actor_handlers
        ]
        if nonblocking:
            return refs
        return ray.get(refs)

    def backload_to_gpu(self, nonblocking=False, backload_optimizer=True, backload_model=True):
        """Backload worker state to GPU

        Args:
            nonblocking: Whether this operation is synchronous or asynchronous.
            If `nonblocking=True`, then the function returns a list of ObjectRefs.
        """
        refs = [
            actor.backload_to_gpu.remote(backload_optimizer=backload_optimizer, backload_model=backload_model)
            for actor in self._actor_handlers
        ]
        if nonblocking:
            return refs
        return ray.get(refs)

    def run_method(self, dispatch_type: str, method_name: str, *args, **kwargs) -> Optional[TrainingOutputBatch]:
        """Run a method on all actors using specified dispatch type synchronously.

        The method should either return `None` or a `TrainingOutputBatch` object.

        Args:
            dispatch_type: Type of dispatch to use ("mesh" or "pass_through")
            method_name: Name of the method to call on actors
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            Collect results from all the actors.
        """
        dispatch_class: Dispatch = DispatchRegistry.get(dispatch_type)
        # validate the dispatch args to be sent to `.dispatch`
        args, kwargs = dispatch_class.validate_dispatch_args(*args, **kwargs)

        # Dispatch the method call
        object_refs = dispatch_class.dispatch(self.actor_infos, method_name, *args, **kwargs)
        # Collect results from all the actors
        ret = dispatch_class.sync_collect(self.actor_infos, object_refs)
        return ret

    def async_run_ray_method(self, dispatch_type: str, method_name: str, *args, **kwargs) -> List[ObjectRef]:
        """Run a method on all actors using specified dispatch type asynchronously.

        Args:
            dispatch_type: Type of dispatch to use ("mesh" or "pass_through")
            method_name: Name of the method to call on actors
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            List of object references
        """
        dispatch_class: Dispatch = DispatchRegistry.get(dispatch_type)
        # validate the dispatch args to be sent to `.dispatch`
        args, kwargs = dispatch_class.validate_dispatch_args(*args, **kwargs)

        # Dispatch the method call
        object_refs = dispatch_class.dispatch(self.actor_infos, method_name, *args, **kwargs)
        return object_refs

    async def async_run_method(
        self, dispatch_type: str, method_name: str, *args, **kwargs
    ) -> Optional[TrainingOutputBatch]:
        """Run a method on all actors using specified dispatch type in an asyncio-compatible way.

        Args:
            dispatch_type: Type of dispatch to use ("mesh" or "pass_through")
            method_name: Name of the method to call on actors
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            TrainingOutputBatch: concatenated results from all actors
        """
        dispatch_class: Dispatch = DispatchRegistry.get(dispatch_type)
        # validate the dispatch args to be sent to `.dispatch`
        args, kwargs = dispatch_class.validate_dispatch_args(*args, **kwargs)

        # Dispatch the method call
        object_refs = dispatch_class.dispatch(self.actor_infos, method_name, *args, **kwargs)
        return await dispatch_class.async_collect(self.actor_infos, object_refs)


class PolicyWorkerBase(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: nn.Module = None
        self.scheduler: LRScheduler = None
        self.optimizer: Optimizer = None
        self.strategy: DistributedStrategy = None
        self.record_memory: bool = False
        self.mesh_rank: MeshRank = None
        self.policy_loss_fn: Callable = PolicyLossRegistry.get(self.cfg.trainer.algorithm.policy_loss_type)

    def _normalize_mini_batch_size(self):
        """
        Initialize micro batch tracking for gradient accumulation.

        The worker no longer needs to know mini batch size - it processes whatever
        batch it receives, breaking it into micro batches. Gradient scaling happens
        at optim_step time based on how many micro batches were accumulated.

        TODO: Rename to _init_gradient_accumulation_state once Megatron no longer
        requires mini-batch normalization in its override. The name is kept for
        backwards compatibility with Megatron which still does actual normalization.
        """
        if not hasattr(self, "mesh_rank") or self.mesh_rank is None:
            raise RuntimeError("mesh_rank must be initialized before calling _normalize_mini_batch_size()")

        # Track micro batches for gradient scaling at optim_step
        self._micro_batches_accumulated = 0

        # Gradient statistics for cross-step metrics (SNR, bits edited, update diversity)
        self._grad_stats = {
            "mean": {},  # Dict[param_name, Tensor] - running mean per param
            "var_numerator": {},  # Dict[param_name, Tensor] - running M2 for variance.
            "count": 0,  # int - number of steps accumulated
            "histograms": [],  # List[Dict[layer, histogram]] - for update diversity
        }
        return

    def _accumulate_gradient_stats(self):
        """Accumulate gradient statistics for cross-step metrics (SNR, bits edited, update diversity)."""
        self._grad_stats["count"] += 1
        n = self._grad_stats["count"]

        step_histograms = {}

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()

            # Initialize on first step
            if name not in self._grad_stats["mean"]:
                self._grad_stats["mean"][name] = torch.zeros_like(grad)
                self._grad_stats["var_numerator"][name] = torch.zeros_like(
                    grad
                )  # stores M2 (sum of squared deviations)

            # Welford's online update for running mean and variance
            delta = grad - self._grad_stats["mean"][name]
            self._grad_stats["mean"][name] += delta / n
            delta2 = grad - self._grad_stats["mean"][name]
            self._grad_stats["var_numerator"][name] += delta * delta2  # M2 accumulator

            # Histogram for this param
            hist = torch.histc(grad.flatten(), bins=50)
            hist = torch.nn.functional.softmax(hist, dim=0)
            # Handle DTensor (FSDP2) - convert to regular tensor first
            if hasattr(hist, "full_tensor"):
                hist = hist.full_tensor()
            step_histograms[name] = hist.detach().cpu().tolist()

        self._grad_stats["histograms"].append(step_histograms)
        return

    def compute_gradient_metrics(self) -> Dict[str, float | int | list[float]]:
        """Compute all 3 gradient-based metrics."""
        grad_metrics = {
            "grad:snr": self._compute_snr(),
            "grad:bits_edited": self._compute_bits_edited(),
            "grad:update_diversity": self._compute_update_diversity(),
        }
        return grad_metrics

    def _compute_snr(self, start_clip: int = 10) -> float:
        """
        Compute gradient Signal-to-Noise Ratio across all parameters.

        SNR = |E[∇θ]| / std(∇θ) measures consistency of gradient direction.
        High SNR = gradients point consistently in same direction across steps.
        Low SNR = noisy/inconsistent gradient directions.
        ---
        Args:
            1. start_clip: Minimum number of steps before computing SNR.
        """
        if self._grad_stats["count"] < start_clip:
            return 0.0

        snr_values = []

        for name in self._grad_stats["mean"]:
            mean = self._grad_stats["mean"][name]
            m2 = self._grad_stats["var_numerator"][name]  # This stores M2, not variance

            # Compute variance from M2 (use Bessel's correction for unbiased estimate)
            # When count <= 1, variance is undefined, so return 0
            if self._grad_stats["count"] <= 1:
                variance = torch.zeros_like(m2)
            else:
                variance = m2 / (self._grad_stats["count"] - 1)
            std = torch.sqrt(variance + 1e-8)  # Add epsilon to avoid division by zero

            # Compute per-element SNR and average
            snr = torch.abs(mean) / std
            snr_values.append(snr.mean().item())

        if len(snr_values) == 0:
            return 0.0

        return sum(snr_values) / len(snr_values)

    def _compute_bits_edited(self) -> int:
        """
        Estimate bits of information edited based on significant gradient updates.

        A gradient is "significant" if |grad| > 5% of |param|.
        Bits edited = floor(3.6 * number of significant gradients).
        """
        if self._grad_stats["count"] == 0:
            return 0

        significant_count = 0

        for name, param in self.model.named_parameters():
            if name not in self._grad_stats["mean"]:
                continue

            mean_grad = self._grad_stats["mean"][name]
            param_magnitude = torch.abs(param.data)

            # Count gradients where |mean_grad| > 5% of |param|
            threshold = 0.05 * param_magnitude
            significant = (torch.abs(mean_grad) > threshold).sum().item()
            significant_count += significant

        # 3.6 bits per significant gradient update
        # source: http://arxiv.org/abs/2505.24832
        return int(3.6 * significant_count)  # int() as an approximate for floor.

    def _compute_update_diversity(self) -> list[float]:
        """
        Compute update diversity as aggregated gradient histogram distribution.

        Aggregates all mean gradients into a single histogram and returns the
        normalized probability distribution. This can be used with cosine
        similarity to measure gradient pattern differences across environments.
        """
        if self._grad_stats["count"] == 0:
            return []

        # Collect all mean gradients into a single tensor
        all_grads = []
        for name in self._grad_stats["mean"]:
            all_grads.append(self._grad_stats["mean"][name].flatten())

        if len(all_grads) == 0:
            return []
        all_grads = torch.cat(all_grads)
        hist = torch.histc(all_grads, bins=50)
        hist = hist / hist.sum()  # Normalize to probability distribution
        hist = torch.nn.functional.softmax(hist, dim=0)  # Apply softmax for emphasis
        # Handle DTensor (FSDP2) - convert to regular tensor first
        if hasattr(hist, "full_tensor"):
            hist = hist.full_tensor()
        # Return probability distribution as list
        return hist.detach().cpu().tolist()

    def reset_gradient_stats(self):
        """Reset gradient statistics for next accumulation period."""
        self._grad_stats = {
            "mean": {},
            "var_numerator": {},
            "count": 0,
            "histograms": [],
        }

    def forward_backward(self, data: TrainingInputBatch) -> Dict[str, float]:
        """
        Perform forward and backward passes for a batch, handling micro-batching internally.

        The batch is split into micro batches based on micro_train_batch_size_per_gpu.
        Gradients accumulate across micro batches. Gradient scaling happens at optim_step.

        Args:
            data: TrainingInputBatch (already DP-sharded by WorkerDispatch/MeshDispatch)

        Returns:
            Aggregated metrics dict across all micro batches
        """
        micro_batch_size = self.cfg.trainer.micro_train_batch_size_per_gpu
        all_metrics = defaultdict(list)

        for micro_batch in BatchIterator(data, micro_batch_size, drop_last=False):
            metrics = self._forward_backward_micro(micro_batch)
            self._micro_batches_accumulated += 1
            for k, v in metrics.items():
                all_metrics[k].append(v)

        return reduce_metrics(dict(all_metrics), ignore_keys=["grad:update_diversity"])

    def _forward_backward_micro(self, experience: Experience) -> Dict[str, float]:
        """
        Perform forward and backward pass for one micro batch.

        Loss is not scaled here - gradient scaling happens at optim_step time.

        Args:
            experience: Experience object for one micro batch

        Returns:
            All-reduced metrics dict for this micro batch
        """
        self.model.train()

        experience.to_device(torch.cuda.current_device())

        sequences = experience.sequences
        old_action_log_probs = experience.action_log_probs
        base_action_log_probs = (
            experience.base_action_log_probs if experience.base_action_log_probs is not None else None
        )
        advantages = experience.advantages
        num_actions = experience.num_actions
        attention_mask = experience.attention_mask
        loss_mask = experience.loss_mask
        rollout_action_logprobs = experience.rollout_logprobs

        # TODO (sumanthrh): don't think this does anything for fsdp rn because autocast happens internally
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            # actor loss
            action_log_probs, output = self.model(
                sequences,
                num_actions,
                attention_mask=attention_mask,
                temperature=self.cfg.generator.sampling_params.temperature,
                return_output=True,
                compute_entropy=True,
                entropy_requires_grad=self.cfg.trainer.algorithm.use_entropy_loss,
            )
            # loss function
            # TODO: recompute advantages
            policy_loss, clip_ratio = self.policy_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages,
                config=self.cfg.trainer.algorithm,
                loss_mask=loss_mask,
                rollout_logprobs=rollout_action_logprobs,
            )

        # entropy loss
        with torch.set_grad_enabled(self.cfg.trainer.algorithm.use_entropy_loss):
            # batch_size, seqlen
            entropy_BS = output["entropy"]
            entropy_BS = entropy_BS[:, -num_actions - 1 : -1]
            entropy = masked_mean(entropy_BS, loss_mask)

        if self.cfg.trainer.algorithm.use_entropy_loss:
            entropy_loss_term = entropy * self.cfg.trainer.algorithm.entropy_loss_coef
        else:
            entropy_loss_term = torch.tensor(0.0)

        # kl loss
        if self.cfg.trainer.algorithm.use_kl_loss:
            kl_loss = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                loss_mask=loss_mask,
                kl_estimator_type=self.cfg.trainer.algorithm.kl_estimator_type,
            )
            kl_loss = masked_mean(kl_loss, loss_mask, dim=-1).mean()
        else:
            kl_loss = torch.tensor(0.0)
        kl_loss_term = kl_loss * self.cfg.trainer.algorithm.kl_loss_coef

        loss = policy_loss + kl_loss_term - entropy_loss_term
        self.strategy.backward(loss, self.model, self.optimizer)

        # Accumulate gradient statistics for cross-step metrics
        if self.cfg.trainer.policy.track_extra_gradient_metrics:
            self._accumulate_gradient_stats()

        status = {
            "final_loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "ppo_clip_ratio": clip_ratio,
            "policy_entropy": entropy.item(),
            "response_length": num_actions,
            "policy_lr": self.scheduler.get_last_lr()[0],
        }
        if self.cfg.trainer.algorithm.use_kl_loss:
            status["policy_kl"] = kl_loss.item()

        # All-reduce metrics across DP workers
        status = self.strategy.all_reduce(status)

        return status

    def optim_step(self) -> float:
        """
        Scale gradients by 1/micro_batches_accumulated, perform optimizer step, and reset counter.

        Returns:
            The gradient norm (before scaling, after clipping)
        """
        # Scale accumulated gradients by 1/N to get correct average
        if self._micro_batches_accumulated > 0:
            scale = 1.0 / self._micro_batches_accumulated
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)

        # Perform optimizer step (includes gradient clipping)
        grad_norm = self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler, name="actor")

        # Reset counter for next accumulation cycle
        self._micro_batches_accumulated = 0

        if grad_norm is not None:
            grad_norm = grad_norm.detach().cpu().item()
        return grad_norm

    def all_reduce_metrics(self, status: Dict[str, int | float | list[float]]) -> Dict[str, int | float | list[float]]:
        """
        All-reduce metrics across data parallel workers.
        """
        return self.strategy.all_reduce(status)

    def get_lr(self) -> float:
        """
        Get current learning rate from scheduler.
        """
        return self.scheduler.get_last_lr()[0]

    def barrier(self) -> None:
        """
        Synchronization barrier across all workers.
        """
        torch.distributed.barrier()

    def save_checkpoint(self, ckpt_dir: Path, tokenizer=None):
        self.strategy.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ckpt_dir=ckpt_dir,
            node_local_rank=self.get_node_local_rank(),
            tokenizer=tokenizer,
        )

    def load_checkpoint(
        self, ckpt_dir: Path, load_optimizer_states: bool = True, load_lr_scheduler_states: bool = True
    ):
        _, states = self.strategy.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer if load_optimizer_states else None,
            scheduler=self.scheduler if load_lr_scheduler_states else None,
            ckpt_dir=ckpt_dir,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
        )
        return states

    def save_hf_model(self, export_dir: str, tokenizer):
        # Save model in HuggingFace safetensors format
        self.strategy.save_hf_model(
            self.model,
            export_dir,
            tokenizer=tokenizer,
        )

    def _forward_micro_batch(self, micro_batch: TrainingInputBatch) -> TrainingOutputBatch:
        device = torch.cuda.current_device()
        micro_batch.to(device)
        self.model.eval()
        sequences = micro_batch["sequences"]
        response_length = micro_batch.metadata["response_length"]
        attention_mask = micro_batch["attention_mask"]

        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            policy_logprob = self.model(
                sequences,
                response_length,
                attention_mask,
                return_output=False,
                temperature=self.cfg.generator.sampling_params.temperature,
            )
        policy_logprob = policy_logprob.to("cpu")
        output = TrainingOutputBatch(
            {"output": policy_logprob},
        )
        output.metadata = micro_batch.metadata
        return output

    def process_sequences(self, sequences, input_len, eos_token_id, pad_token_id):
        return self.model.process_sequences(sequences, input_len, eos_token_id, pad_token_id)


class CriticWorkerBase(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: nn.Module = None
        self.scheduler: LRScheduler = None
        self.optimizer: Optimizer = None
        self.strategy: DistributedStrategy = None
        self.record_memory: bool = False
        self.mesh_rank: MeshRank = None
        self.critic_loss_fn: Callable = ppo_critic_loss

    def _normalize_mini_batch_size(self):
        """
        Initialize micro batch tracking for gradient accumulation.

        The worker no longer needs to know mini batch size - it processes whatever
        batch it receives, breaking it into micro batches. Gradient scaling happens
        at optim_step time based on how many micro batches were accumulated.

        TODO: Rename to _init_gradient_accumulation_state once Megatron no longer
        requires mini-batch normalization in its override. The name is kept for
        backwards compatibility with Megatron which still does actual normalization.
        """
        if not hasattr(self, "mesh_rank") or self.mesh_rank is None:
            raise RuntimeError("mesh_rank must be initialized before calling _normalize_mini_batch_size()")

        # Track micro batches for gradient scaling at optim_step
        self._micro_batches_accumulated = 0

    def forward_backward(self, data: TrainingInputBatch) -> Dict[str, float]:
        """
        Perform forward and backward passes for a batch, handling micro-batching internally.

        The batch is split into micro batches based on micro_train_batch_size_per_gpu.
        Gradients accumulate across micro batches. Gradient scaling happens at optim_step.

        Args:
            data: TrainingInputBatch (already DP-sharded by WorkerDispatch/MeshDispatch)

        Returns:
            Aggregated metrics dict across all micro batches
        """
        micro_batch_size = self.cfg.trainer.micro_train_batch_size_per_gpu
        all_metrics = defaultdict(list)

        for micro_batch in BatchIterator(data, micro_batch_size, drop_last=False):
            metrics = self._forward_backward_micro(micro_batch)
            self._micro_batches_accumulated += 1
            for k, v in metrics.items():
                all_metrics[k].append(v)

        return reduce_metrics(dict(all_metrics), ignore_keys=["grad:update_diversity"])

    def _forward_backward_micro(self, experience: Experience) -> Dict[str, float]:
        """
        Perform forward and backward pass for one micro batch.

        Loss is NOT scaled here - gradient scaling happens at optim_step time.

        Args:
            experience: Experience object for one micro batch

        Returns:
            All-reduced metrics dict for this micro batch
        """
        self.model.train()

        experience.to_device(torch.cuda.current_device())

        sequences = experience.sequences
        old_values = experience.values
        returns = experience.returns
        num_actions = experience.num_actions
        attention_mask = experience.attention_mask
        loss_mask = experience.loss_mask

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            # critic loss
            values, _ = self.model(
                sequences,
                num_actions=num_actions,
                attention_mask=attention_mask,
                return_output=True,
            )
            # loss function
            loss, clipfrac = self.critic_loss_fn(
                values,
                old_values,
                returns,
                config=self.cfg.trainer.algorithm,
                loss_mask=loss_mask,
            )
        # NO loss scaling here - gradient scaling happens at optim_step
        self.strategy.backward(loss, self.model, self.optimizer)

        status = {
            "critic_loss": loss.item(),
            "values_mean": masked_mean(values, loss_mask).item(),
            "values_clipfrac": clipfrac,
            "critic_lr": self.scheduler.get_last_lr()[0],
        }

        # All-reduce metrics across DP workers
        status = self.strategy.all_reduce(status)

        return status

    def optim_step(self) -> float:
        """
        Scale gradients by 1/micro_batches_accumulated, perform optimizer step, and reset counter.

        Returns:
            The gradient norm (before scaling, after clipping)
        """
        # Scale accumulated gradients by 1/N to get correct average
        if self._micro_batches_accumulated > 0:
            scale = 1.0 / self._micro_batches_accumulated
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)

        # Perform optimizer step (includes gradient clipping)
        grad_norm = self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler, name="critic")

        # Reset counter for next accumulation cycle
        self._micro_batches_accumulated = 0

        if grad_norm is not None:
            grad_norm = grad_norm.detach().cpu().item()
        return grad_norm

    def all_reduce_metrics(self, status: Dict[str, int | float | list[float]]) -> Dict[str, int | float | list[float]]:
        """
        All-reduce metrics across data parallel workers.
        """
        return self.strategy.all_reduce(status)

    def get_lr(self) -> float:
        """
        Get current learning rate from scheduler.
        """
        return self.scheduler.get_last_lr()[0]

    def barrier(self) -> None:
        """
        Synchronization barrier across all workers.
        """
        torch.distributed.barrier()

    def _forward_micro_batch(
        self,
        micro_batch: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Generates critic values."""
        device = torch.cuda.current_device()
        micro_batch.to(device)
        sequences = micro_batch["sequences"]
        response_length = micro_batch.metadata["response_length"]
        attention_mask = micro_batch["attention_mask"]
        self.model.eval()
        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            value = self.model(
                sequences,
                response_length,
                attention_mask,
            )
        self.model.train()  # reset model state
        value = value.to("cpu")
        output = TrainingOutputBatch(
            {"output": value},
        )
        output.metadata = micro_batch.metadata
        return output

    def save_hf_model(self, export_dir: str, tokenizer):
        # Save model in HuggingFace safetensors format
        self.strategy.save_hf_model(
            self.model,
            export_dir,
            tokenizer=tokenizer,
        )

    def save_checkpoint(self, ckpt_dir: str, tokenizer=None):
        self.strategy.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ckpt_dir=ckpt_dir,
            node_local_rank=self.get_node_local_rank(),
            tokenizer=tokenizer,
        )

    def load_checkpoint(self, ckpt_dir=None, load_optimizer_states=True, load_lr_scheduler_states=True):
        _, states = self.strategy.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer if load_optimizer_states else None,
            scheduler=self.scheduler if load_lr_scheduler_states else None,
            ckpt_dir=ckpt_dir,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
        )
        return states


class RefWorkerBase(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: nn.Module = None

    def _forward_micro_batch(self, micro_batch: TrainingInputBatch) -> TrainingOutputBatch:
        device = torch.cuda.current_device()
        micro_batch.to(device)
        sequences = micro_batch["sequences"]
        response_length = micro_batch.metadata["response_length"]
        attention_mask = micro_batch["attention_mask"]
        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            log_probs = self.model(sequences, response_length, attention_mask, return_output=False)
        log_probs = log_probs.to("cpu")
        output = TrainingOutputBatch(
            {"output": log_probs},
        )
        output.metadata = micro_batch.metadata
        return output
