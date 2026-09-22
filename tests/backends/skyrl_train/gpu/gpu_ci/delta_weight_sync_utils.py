"""Test utilities for delta weight sync"""

from typing import Any, Dict

import ray
import torch
from ray.util.placement_group import placement_group

from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.train.utils import get_ray_pg_ready_with_timeout
from skyrl.train.utils.utils import ResolvedPlacementGroup


def _apply_sparse_weight_perturbation(params, sparsity: float, delta: float, phase: int) -> Dict[str, Any]:
    """Deterministic sparse in-place perturbation for isolated weight-sync benchmarks.

    Updates a strided subset of each floating-point parameter without allocating a
    full random mask. Shared by the FSDP and Megatron benchmark worker subclasses.
    """
    if not 0.0 < sparsity <= 1.0:
        raise ValueError(f"sparsity must be in (0, 1], got {sparsity}")
    stride = max(1, round(1.0 / sparsity))
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    start_offset = (rank + phase) % stride
    total_elements = 0
    updated_elements = 0
    tensors = 0
    with torch.no_grad():
        for param in params:
            if not param.is_floating_point():
                continue
            flat = param.data.view(-1)
            total_elements += flat.numel()
            if flat.numel() <= start_offset:
                continue
            view = flat[start_offset::stride]
            if view.numel() == 0:
                continue
            view.add_(delta)
            updated_elements += view.numel()
            tensors += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return {
        "rank": rank,
        "phase": phase,
        "stride": stride,
        "tensors": tensors,
        "total_elements": total_elements,
        "updated_elements": updated_elements,
        "actual_sparsity": (updated_elements / total_elements) if total_elements else 0.0,
    }


def sparse_delta_benchmark_policy_worker_cls(strategy: str):
    """Ray actor class for the policy worker with a benchmark-only perturbation method.

    ``benchmark_apply_sparse_weight_delta`` lives here (test/benchmark support code)
    rather than on the production worker classes. It applies a real weight update
    without an optimizer step so the delta weight-sync e2e test and the isolated
    weight-sync benchmark can exercise publisher -> receiver end-to-end.
    """
    if strategy == "fsdp":
        from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import (
            FSDPPolicyWorkerBase as Base,
        )

        def benchmark_apply_sparse_weight_delta(self, sparsity: float = 0.04, delta: float = 1.0e-3, phase: int = 0):
            if self.model is None:
                raise RuntimeError("model is not initialized")
            return _apply_sparse_weight_perturbation(self.model.model.parameters(), sparsity, delta, phase)

    elif strategy == "megatron":
        from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
            MegatronPolicyWorkerBase as Base,
        )

        def benchmark_apply_sparse_weight_delta(self, sparsity: float = 0.04, delta: float = 1.0e-3, phase: int = 0):
            if self.actor_module is None:
                raise RuntimeError("actor_module is not initialized")
            params = (param for module in self.actor_module for _, param in module.named_parameters())
            return _apply_sparse_weight_perturbation(params, sparsity, delta, phase)

    else:
        raise ValueError(f"Unknown strategy type: {strategy}")

    subclass = type(
        f"SparseDeltaBenchmark{Base.__name__}",
        (Base,),
        {"benchmark_apply_sparse_weight_delta": benchmark_apply_sparse_weight_delta},
    )
    return ray.remote(num_gpus=1)(subclass)


def init_policy_worker_for_delta(
    cfg,
    shared_pg=None,
    colocate_all=False,
    num_gpus_per_node=1,
    num_nodes=1,
) -> PPORayActorGroup:

    if shared_pg is not None:
        pg = ResolvedPlacementGroup(shared_pg)
        num_gpus_per_actor = 0.2
    else:
        bundles = [{"GPU": num_gpus_per_node, "CPU": num_gpus_per_node} for _ in range(num_nodes)]
        raw_pg = placement_group(bundles, strategy="PACK")
        get_ray_pg_ready_with_timeout(raw_pg, timeout=30)
        pg = ResolvedPlacementGroup(raw_pg)
        num_gpus_per_actor = 0.75

    worker_cls = sparse_delta_benchmark_policy_worker_cls(cfg.trainer.strategy)
    model = PPORayActorGroup(
        cfg.trainer,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        ray_actor_type=worker_cls,
        pg=pg,
        num_gpus_per_actor=num_gpus_per_actor,
        colocate_all=colocate_all,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
        record_memory=cfg.trainer.policy.record_memory,
    )
    # we use policy model path for all tests (regardless of actor type)
    ray.get(model.async_init_model(cfg.trainer.policy.model.path))
    return model
