"""
Tests for WorkerDispatch forward/backward status aggregation.

uv run --isolated --extra skyrl-train --extra dev pytest tests/backends/skyrl_train/workers/test_worker_dispatch.py
"""

import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import (
    ActorInfo,
    MeshDispatch,
    MeshRank,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.train.config import SkyRLTrainConfig


@ray.remote
class FakeForwardBackwardWorker:
    def __init__(self, rank: int, dp_rank: int):
        self.rank = rank
        self.dp_rank = dp_rank

    def forward_backward(self, data: TrainingInputBatch, loss_fn=None, loss_fn_config=None):
        del loss_fn_config

        status = {
            "policy_loss": 1.25,
            "policy_lr": 0.5,
        }
        if loss_fn == "scalar_only":
            return status

        status["loss_fn_outputs"] = [
            {"sample_id": sample_id, "dp_rank": self.dp_rank} for sample_id in data["sample_id"].tolist()
        ]
        return status

    def save_memory_snapshot(self, tag: str):
        return tag


class StubActorGroup:
    def __init__(self, dp_size: int = 2):
        self.actors = [FakeForwardBackwardWorker.remote(rank=i, dp_rank=i) for i in range(dp_size)]
        self.actor_infos = [
            ActorInfo(
                actor,
                MeshRank(dp=i, sp=0, tp=0, pp=0, world_size=dp_size, dp_size=dp_size, pp_size=1),
            )
            for i, actor in enumerate(self.actors)
        ]

    def async_run_ray_method(self, dispatch_type: str, method_name: str, *args, **kwargs):
        if dispatch_type == "mesh":
            return MeshDispatch.dispatch(self.actor_infos, method_name, *args, **kwargs)
        if dispatch_type == "pass_through":
            return [getattr(actor_info.handle, method_name).remote(*args, **kwargs) for actor_info in self.actor_infos]
        raise AssertionError(f"Unsupported dispatch type: {dispatch_type}")


def _make_dispatch() -> WorkerDispatch:
    cfg = SkyRLTrainConfig()
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    return WorkerDispatch(cfg, policy_actor_group=StubActorGroup())


def _make_batch(batch_size: int = 4) -> TrainingInputBatch:
    return TrainingInputBatch(
        {
            "sample_id": torch.arange(batch_size, dtype=torch.long),
            "dummy": torch.arange(batch_size, dtype=torch.long),
        }
    )


def test_forward_backward_from_staged_matches_unstaged_loss_fn_outputs(ray_init):
    dispatch = _make_dispatch()
    batch = _make_batch()

    unstaged = dispatch.forward_backward("policy", batch)
    chunk_refs = dispatch.stage_data("policy", batch, [(0, len(batch))])[0]
    staged = dispatch.forward_backward_from_staged("policy", chunk_refs)

    assert staged == unstaged
    assert [output["sample_id"] for output in staged["loss_fn_outputs"]] == [0, 1, 2, 3]
    assert [output["dp_rank"] for output in staged["loss_fn_outputs"]] == [0, 0, 1, 1]


def test_forward_backward_from_staged_preserves_scalar_only_status(ray_init):
    dispatch = _make_dispatch()
    batch = _make_batch()

    unstaged = dispatch.forward_backward("policy", batch, loss_fn="scalar_only")
    chunk_refs = dispatch.stage_data("policy", batch, [(0, len(batch))])[0]
    staged = dispatch.forward_backward_from_staged("policy", chunk_refs, loss_fn="scalar_only")

    assert staged == {"policy_loss": 1.25, "policy_lr": 0.5}
    assert staged == unstaged
    assert "loss_fn_outputs" not in staged
