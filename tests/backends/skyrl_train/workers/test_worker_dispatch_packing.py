from types import SimpleNamespace

import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import ActorInfo, MeshDispatch, MeshRank
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl.backends.skyrl_train.utils.packing_balance import compute_slot_costs
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from tests.train.util import example_dummy_config


@ray.remote
class ForwardEchoActor:
    def forward(self, data: TrainingInputBatch) -> TrainingOutputBatch:
        output = TrainingOutputBatch({"output": data["sample_ids"].clone()})
        output.metadata = data.metadata
        return output


def _make_actor_infos(dp_size: int) -> list[ActorInfo]:
    actors = [ForwardEchoActor.remote() for _ in range(dp_size)]
    return [
        ActorInfo(
            actor,
            MeshRank(dp=dp_rank, sp=0, tp=0, pp=0, world_size=dp_size, dp_size=dp_size, pp_size=1),
        )
        for dp_rank, actor in enumerate(actors)
    ]


def _make_dispatch(cfg, dp_size: int) -> WorkerDispatch:
    actor_group = SimpleNamespace(
        actor_infos=_make_actor_infos(dp_size),
        backload_to_gpu=lambda **kwargs: None,
        offload_to_cpu=lambda **kwargs: None,
    )
    return WorkerDispatch(cfg, policy_actor_group=actor_group)


def _make_training_batch(lengths: list[int]) -> TrainingInputBatch:
    max_len = max(lengths)
    attention_mask = torch.zeros(len(lengths), max_len, dtype=torch.int64)
    for row, length in enumerate(lengths):
        attention_mask[row, :length] = 1

    batch = TrainingInputBatch(
        {
            "sample_ids": torch.arange(len(lengths), dtype=torch.int64).unsqueeze(-1),
            "attention_mask": attention_mask,
            "sequences": torch.arange(len(lengths) * max_len, dtype=torch.int64).reshape(len(lengths), max_len),
        }
    )
    batch.metadata = {
        "uids": [f"u{i}" for i in range(len(lengths))],
        "response_length": 1,
    }
    return batch


def test_worker_dispatch_stage_data_matches_mesh_dispatch_when_packing_disabled():
    cfg = example_dummy_config()
    cfg.trainer.use_sample_packing = False
    dispatch = _make_dispatch(cfg, dp_size=2)

    data = _make_training_batch([4, 4, 4, 4, 4, 4, 4, 4])
    expected = MeshDispatch.stage_chunks(2, data, mini_batch_size=4)
    actual = dispatch.stage_data("policy", data, mini_batch_size=4)

    expected_batches = [[ray.get(ref) for ref in mini_batch] for mini_batch in expected]
    actual_batches = [[ray.get(ref) for ref in mini_batch] for mini_batch in actual]
    assert actual_batches == expected_batches


def test_worker_dispatch_stage_data_balances_sample_packing():
    cfg = example_dummy_config()
    cfg.trainer.use_sample_packing = True
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    dispatch = _make_dispatch(cfg, dp_size=2)

    lengths = [100, 99, 98, 97, 4, 3, 2, 1]
    data = _make_training_batch(lengths)
    chunk_refs = dispatch.stage_data("policy", data, mini_batch_size=8)

    chunks = [ray.get(ref) for ref in chunk_refs[0]]
    permutation = torch.cat([chunk["sample_ids"].squeeze(-1) for chunk in chunks]).tolist()

    baseline_slot_costs = compute_slot_costs(lengths, dp_size=2, local_micro_bsz=2)
    balanced_slot_costs = compute_slot_costs(lengths, dp_size=2, local_micro_bsz=2, permutation=permutation)

    assert sorted(permutation) == list(range(len(lengths)))
    assert max(balanced_slot_costs) < max(baseline_slot_costs)
    assert chunks[0].metadata["uids"] == [data.metadata["uids"][idx] for idx in permutation]


def test_worker_dispatch_forward_restores_original_order_with_sample_packing():
    cfg = example_dummy_config()
    cfg.trainer.use_sample_packing = True
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    dispatch = _make_dispatch(cfg, dp_size=2)

    data = _make_training_batch([100, 99, 98, 97, 4, 3, 2, 1])
    output = dispatch.forward("policy", data)

    expected_ids = torch.arange(data.batch_size, dtype=torch.int64).unsqueeze(-1)
    assert torch.equal(output["output"], expected_ids)
    assert output.metadata["uids"] == data.metadata["uids"]
