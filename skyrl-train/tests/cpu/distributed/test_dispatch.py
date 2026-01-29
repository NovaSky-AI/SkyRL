"""
uv run --isolated --extra dev pytest tests/cpu/distributed/test_dispatch.py
"""

from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.distributed.dispatch import (
    MeshDispatch,
    PassThroughDispatch,
    MeshRank,
    ActorInfo,
    DispatchRegistry,
    Dispatch,
)
import ray
import torch
from typing import List, Optional, Union
from ray import ObjectRef
import pytest


@ray.remote
class RayActor:
    def __init__(self, rank: int, dp_rank: int):
        self.rank = rank
        self.dp_rank = dp_rank

    def do_work(self, data: TrainingInputBatch):
        # intentionally create different outputs for each rank
        data["a"] += self.rank
        return data

    def do_work_from_staged(self, data: TrainingInputBatch, start_idx: int, end_idx: int):
        """
        Method compatible with dispatch_from_staged.
        Ray auto-resolves ObjectRef to actual data, so we receive TrainingInputBatch directly.
        """
        # Slice the data to get this worker's portion
        data = data[start_idx:end_idx]
        # Apply same transformation as do_work
        data["a"] = data["a"] + self.rank
        return data

    def dummy(self, a, b):
        return


class RayActorGroup:
    def __init__(self, num_actors: int):
        sp_size = 2
        dp_size = num_actors // sp_size
        self.actors = [RayActor.remote(i, i % dp_size) for i in range(num_actors)]
        self.actor_infos = [
            ActorInfo(
                actor,
                MeshRank(
                    dp=i % dp_size, sp=i // dp_size, tp=0, pp=0, world_size=num_actors, dp_size=dp_size, pp_size=1
                ),
            )
            for i, actor in enumerate(self.actors)
        ]

    def mesh_dispatch_and_collect(self, data: TrainingInputBatch):
        object_refs = MeshDispatch.dispatch(self.actor_infos, "do_work", data)
        ret = MeshDispatch.sync_collect(self.actor_infos, object_refs)
        return ret

    def pass_through_dispatch(self, a, b):
        # just pass values as is
        object_refs = PassThroughDispatch.dispatch(self.actor_infos, "dummy", a, b)
        ret = PassThroughDispatch.sync_collect(self.actor_infos, object_refs)
        return ret


def test_mesh_dispatch():
    num_actors = 8
    actor_group = RayActorGroup(num_actors)
    data = TrainingInputBatch({"a": torch.tensor([1, 2, 3, 4])})
    databatch = actor_group.mesh_dispatch_and_collect(data)
    # only dp rank 0, 1, 2, 3, sp 0 will have the contributed to the output.
    # In this case, the rank for these are 0, 1, 2, 3.
    assert torch.equal(databatch["a"], torch.tensor([1, 3, 5, 7]))


def test_dispatch_from_staged():
    """Test dispatch_from_staged: workers receive ObjectRef + slice indices and fetch/slice locally."""
    num_actors = 8
    actor_group = RayActorGroup(num_actors)

    # Create a larger batch that will be sliced into mini-batches
    # Full batch has 16 elements, we'll request a mini-batch of indices [4:12]
    full_data = TrainingInputBatch({"a": torch.arange(16)})

    # Stage the full data in Ray object store once
    data_ref = ray.put(full_data)

    # Request mini-batch from index 4 to 12 (size=8)
    # With dp_size=4, each worker gets chunk_size=2
    # dp_rank 0: [4:6], dp_rank 1: [6:8], dp_rank 2: [8:10], dp_rank 3: [10:12]
    start_idx = 4
    end_idx = 12

    object_refs = MeshDispatch.dispatch_from_staged(
        actor_group.actor_infos,
        "do_work_from_staged",
        data_ref=data_ref,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    # Collect results (only sp=0 workers contribute, which are ranks 0,1,2,3)
    results = MeshDispatch.sync_collect(actor_group.actor_infos, object_refs)

    # Expected: each dp_rank gets 2 elements, adds its rank
    # dp_rank 0 (rank 0): [4,5] + 0 = [4,5]
    # dp_rank 1 (rank 1): [6,7] + 1 = [7,8]
    # dp_rank 2 (rank 2): [8,9] + 2 = [10,11]
    # dp_rank 3 (rank 3): [10,11] + 3 = [13,14]
    # Concatenated: [4,5,7,8,10,11,13,14]
    expected = torch.tensor([4, 5, 7, 8, 10, 11, 13, 14])
    assert torch.equal(results["a"], expected), f"Expected {expected}, got {results['a']}"


def test_pass_through_dispatch():
    num_actors = 8
    actor_group = RayActorGroup(num_actors)
    ret = actor_group.pass_through_dispatch(1, 2)
    assert ret is None


def test_mesh_dispatch_with_mixed():
    num_actors = 8
    actor_group = RayActorGroup(num_actors)
    object_refs = MeshDispatch.dispatch(
        actor_group.actor_infos,
        "do_work",
        TrainingInputBatch({"a": torch.tensor([1, 2, 3, 4])}),
    )
    object_refs[0] = ray.put(None)
    with pytest.raises(AssertionError):
        MeshDispatch.sync_collect(actor_group.actor_infos, object_refs)


def test_dispatch_registry():
    # add a custom dispatch type
    try:

        class CustomDispatch(Dispatch):
            @classmethod
            def dispatch(cls, actor_infos: List[ActorInfo], method: str, *args, **kwargs) -> List[ObjectRef]:
                pass

            @classmethod
            def sync_collect(
                cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef], nonblocking: bool = False
            ) -> Union[List[ObjectRef], TrainingInputBatch]:
                pass

            @classmethod
            def async_collect(
                cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef]
            ) -> Optional[TrainingInputBatch]:
                pass

        DispatchRegistry.register("custom", CustomDispatch)
        assert DispatchRegistry.get("custom") == CustomDispatch
        assert DispatchRegistry.list_registered() == {
            "mesh": MeshDispatch,
            "pass_through": PassThroughDispatch,
            "custom": CustomDispatch,
        }
    finally:
        DispatchRegistry._registry.pop("custom")
