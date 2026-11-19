"""Token-balanced data-parallel split (``trainer.balance_dp_tokens``): the deal, dispatch, and order restore.

uv run --isolated --extra skyrl-train --extra dev pytest tests/backends/skyrl_train/distributed/test_balanced_dp_split.py
"""

import random
from types import SimpleNamespace

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import (
    ActorInfo,
    MeshDispatch,
    MeshRank,
    WorkerOutput,
    balanced_dp_indices,
    restore_sample_order,
    split_for_dp,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

DP = 16


def _prompt_grouped_lengths(seed: int = 0, prompts: int = 32, samples_per_prompt: int = 8):
    """Samples of one prompt sit together and have similar lengths, as in an RL rollout batch."""
    rng = random.Random(seed)
    lengths = []
    for _ in range(prompts):
        base = rng.choice([900, 2500, 5000, 8192, 8192])
        lengths += [min(8192, max(200, int(base * rng.uniform(0.7, 1.2)))) for _ in range(samples_per_prompt)]
    return lengths


def test_deal_gives_every_rank_the_same_number_of_samples_and_similar_tokens():
    lengths = _prompt_grouped_lengths()
    per_rank = len(lengths) // DP
    indices = balanced_dp_indices(lengths, DP)

    assert [len(rank) for rank in indices] == [per_rank] * DP
    assert sorted(i for rank in indices for i in rank) == list(range(len(lengths)))
    for rank in indices:  # longest sample first
        assert [lengths[i] for i in rank] == sorted((lengths[i] for i in rank), reverse=True)

    mean = sum(lengths) / DP
    contiguous = [sum(lengths[r * per_rank : (r + 1) * per_rank]) for r in range(DP)]
    balanced = [sum(lengths[i] for i in rank) for rank in indices]
    assert max(contiguous) / mean > 1.3  # the default split is uneven on this batch
    assert max(balanced) / mean < 1.1
    assert min(balanced) / mean > 0.9


def test_deal_edge_cases():
    assert balanced_dp_indices([], 4) == [[], [], [], []]
    assert balanced_dp_indices([7, 9], 2) == [[1], [0]]
    assert balanced_dp_indices([5, 5, 5, 5], 1) == [[0, 1, 2, 3]]
    with pytest.raises(AssertionError):
        balanced_dp_indices([1, 2, 3], 2)


def _batch(lengths, seq_len=64):
    n = len(lengths)
    attention_mask = torch.zeros(n, seq_len, dtype=torch.long)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1
    batch = TrainingInputBatch(
        {"sequences": torch.arange(n).unsqueeze(1).repeat(1, seq_len), "attention_mask": attention_mask}
    )
    batch.metadata = {"response_length": 4}
    return batch


def test_without_the_option_the_split_is_unchanged():
    """``balance_tokens=False`` is the split before the option existed: ``data.chunk`` in batch order."""
    lengths = [5, 60, 7, 50, 9, 40, 11, 30]
    dp_size = 4
    data = _batch(lengths)
    expected = _batch(lengths).chunk(len(lengths) // dp_size)  # the old code
    for kwargs in ({}, {"balance_tokens": False}):
        chunks, order = split_for_dp(data, dp_size=dp_size, **kwargs)
        assert order is None
        assert len(chunks) == len(expected)
        for chunk, old_chunk in zip(chunks, expected):
            assert chunk.keys() == old_chunk.keys()
            for key in chunk.keys():
                assert torch.equal(chunk[key], old_chunk[key])
            assert chunk.metadata == old_chunk.metadata == {"response_length": 4}
    for balance_tokens in (False, True):  # the check on the batch size is kept in both modes
        with pytest.raises(AssertionError):
            split_for_dp(_batch([5, 60, 7]), dp_size=2, balance_tokens=balance_tokens)


def test_balanced_split_changes_only_the_assignment_of_samples_to_ranks():
    lengths = [60, 58, 55, 50, 6, 5, 4, 4, 3, 3, 2, 2]
    data = _batch(lengths)
    chunks, order = split_for_dp(data, dp_size=2, balance_tokens=True)
    default_chunks, _ = split_for_dp(data, dp_size=2)

    assert sorted(order) == list(range(len(lengths)))
    assert [len(c) for c in chunks] == [len(c) for c in default_chunks] == [6, 6]
    assert [c["sequences"][:, 0].tolist() for c in chunks] == [order[:6], order[6:]]
    for chunk, default_chunk in zip(chunks, default_chunks):
        assert chunk.keys() == default_chunk.keys()
        assert chunk.metadata == default_chunk.metadata  # nothing added for the worker
    tokens = [int(c["attention_mask"].sum()) for c in chunks]
    default_tokens = [int(c["attention_mask"].sum()) for c in default_chunks]
    assert abs(tokens[0] - tokens[1]) <= 8 < abs(default_tokens[0] - default_tokens[1])

    # Per-sample outputs come back in chunk order; the restore gives the batch order.
    assert restore_sample_order([f"out{i}" for i in order], order) == [f"out{i}" for i in range(len(lengths))]
    assert restore_sample_order([], order) == []
    with pytest.raises(ValueError):
        restore_sample_order(["x"], order)


def test_batches_that_cannot_be_balanced_get_the_default_split():
    data = _batch([60, 5, 50, 4])
    del data["attention_mask"]
    chunks, order = split_for_dp(data, dp_size=2, balance_tokens=True)
    assert order is None
    assert [c["sequences"][:, 0].tolist() for c in chunks] == [[0, 1], [2, 3]]


@pytest.mark.parametrize("value, expected", [(False, False), (True, True), (None, False), ("true", False)])
def test_worker_dispatch_reads_the_option_from_the_config(value, expected):
    dispatch = WorkerDispatch.__new__(WorkerDispatch)
    # The option does not depend on ``max_tokens_per_microbatch``.
    dispatch.cfg = SimpleNamespace(trainer=SimpleNamespace(balance_dp_tokens=value, max_tokens_per_microbatch=-1))
    assert dispatch._dp_balance_tokens is expected
    del dispatch.cfg  # a dispatch without a config (as some tests build it) uses the default split
    assert dispatch._dp_balance_tokens is False


@ray.remote
class _EchoActor:
    def echo(self, data: TrainingInputBatch):
        ids = data["sequences"][:, 0].tolist()
        return WorkerOutput(loss_fn_outputs=[{"id": i} for i in ids], metrics={"n": len(ids)})


def _actor_infos(dp_size):
    return [
        ActorInfo(_EchoActor.remote(), MeshRank(dp=i, sp=0, tp=0, pp=0, world_size=dp_size, dp_size=dp_size, pp_size=1))
        for i in range(dp_size)
    ]


def test_mesh_dispatch_balanced_round_trip():
    lengths = [60, 58, 55, 50, 6, 5, 4, 4, 3, 3, 2, 2]
    data = _batch(lengths)
    infos = _actor_infos(2)

    refs = MeshDispatch.dispatch(infos, "echo", data, dp_balance_tokens=True)
    shards = ray.get(refs)
    assert [len(s.loss_fn_outputs) for s in shards] == [6, 6]
    assert [x["id"] for s in shards for x in s.loss_fn_outputs] != list(range(len(lengths)))
    out = WorkerOutput.cat(infos, shards, sample_order=refs.sample_order)
    assert [x["id"] for x in out.loss_fn_outputs] == list(range(len(lengths)))

    refs = MeshDispatch.dispatch(infos, "echo", data)  # default: batch order, no permutation
    assert refs.sample_order is None
    out = WorkerOutput.cat(infos, ray.get(refs), sample_order=refs.sample_order)
    assert [x["id"] for x in out.loss_fn_outputs] == list(range(len(lengths)))

    staged = MeshDispatch.stage_chunks(2, data, [(0, 6), (6, 12)], dp_balance_tokens=True)
    for (start, end), chunk_refs in zip([(0, 6), (6, 12)], staged):
        refs = MeshDispatch.dispatch_from_staged(infos, "echo", chunk_refs=chunk_refs)
        out = WorkerOutput.cat(infos, ray.get(refs), sample_order=chunk_refs.sample_order)
        assert [x["id"] for x in out.loss_fn_outputs] == list(range(start, end))
