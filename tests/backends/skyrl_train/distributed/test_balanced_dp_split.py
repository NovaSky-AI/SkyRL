"""Token-balanced data-parallel split (``trainer.balance_dp_tokens``): packer, dispatch, worker iterator.

uv run --isolated --extra skyrl-train --extra dev pytest tests/backends/skyrl_train/distributed/test_balanced_dp_split.py
"""

import random

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import (
    ActorInfo,
    MeshDispatch,
    MeshRank,
    WorkerOutput,
    restore_sample_order,
    split_for_dp,
)
from skyrl.backends.skyrl_train.training_batch import (
    MICROBATCH_SIZES_KEY,
    TrainingInputBatch,
)
from skyrl.backends.skyrl_train.workers.worker_utils import TokenBasedBatchIterator
from skyrl.train.dataset.bin_packing import balanced_dp_assignment, make_seq_packer

CAP, DP = 10752, 16


def _prompt_grouped_lengths(seed: int = 0, prompts: int = 32, samples_per_prompt: int = 8):
    """Samples of one prompt sit together and have similar lengths, as in an RL rollout batch."""
    rng = random.Random(seed)
    lengths = []
    for _ in range(prompts):
        base = rng.choice([900, 2500, 5000, 8192, 8192])
        lengths += [min(8192, max(200, int(base * rng.uniform(0.7, 1.2)))) for _ in range(samples_per_prompt)]
    return lengths


def _contiguous_split_stats(lengths, dp, cap):
    """Today's split: equal sample counts in batch order, each rank packs for itself."""
    per_rank = len(lengths) // dp
    packer = make_seq_packer("balanced", bin_capacity=cap)
    slot_tokens = []
    for r in range(dp):
        chunk = lengths[r * per_rank : (r + 1) * per_rank]
        slot_tokens.append(sorted((sum(chunk[i] for i in b) for b in packer.pack(chunk)), reverse=True))
    num_slots = max(len(s) for s in slot_tokens)
    return num_slots, [sum(s) for s in slot_tokens]


def test_assignment_covers_every_sample_once_and_respects_the_cap():
    lengths = _prompt_grouped_lengths()
    assignment = balanced_dp_assignment(lengths, DP, CAP)
    assert len(assignment) == DP
    assert len({len(slots) for slots in assignment}) == 1  # same number of micro-batches on every rank
    seen = sorted(i for slots in assignment for mb in slots for i in mb)
    assert seen == list(range(len(lengths)))
    for slots in assignment:
        for mb in slots:
            assert mb and sum(lengths[i] for i in mb) <= CAP


def test_assignment_needs_fewer_slots_and_balances_ranks_and_slots():
    lengths = _prompt_grouped_lengths()
    assignment = balanced_dp_assignment(lengths, DP, CAP)
    old_slots, old_rank_tokens = _contiguous_split_stats(lengths, DP, CAP)
    num_slots = len(assignment[0])
    rank_tokens = [sum(lengths[i] for mb in slots for i in mb) for slots in assignment]
    mean = sum(rank_tokens) / DP

    assert num_slots < old_slots
    assert max(old_rank_tokens) / mean > 1.3  # the contiguous split is uneven on this batch
    assert max(rank_tokens) / mean < 1.1
    for slot in range(num_slots):
        tokens = [sum(lengths[i] for i in slots[slot]) for slots in assignment]
        assert max(tokens) - min(tokens) <= 8192  # at most one sample apart


def test_assignment_edge_cases():
    assert balanced_dp_assignment([10, 20], 4, 100) is None  # fewer samples than ranks
    # A sample longer than the cap gets a micro-batch of its own.
    assignment = balanced_dp_assignment([500, 10, 10, 10], 2, 100)
    assert sorted(len(mb) for slots in assignment for mb in slots) == [1, 3]
    # As many micro-batches as samples: one sample each, no empty micro-batch.
    assignment = balanced_dp_assignment([90, 90, 90, 90], 2, 100)
    assert [[len(mb) for mb in slots] for slots in assignment] == [[1, 1], [1, 1]]


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


def test_split_for_dp_default_is_the_contiguous_split():
    data = _batch([5, 60, 7, 50, 9, 40, 11, 30])
    chunks, order = split_for_dp(data, dp_size=4)
    assert order is None
    assert [c["sequences"][:, 0].tolist() for c in chunks] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert MICROBATCH_SIZES_KEY not in chunks[0].metadata


def test_split_for_dp_balanced_chunks_and_order_restore():
    lengths = [60, 58, 55, 50, 6, 5, 4, 4, 3, 3, 2, 2]
    data = _batch(lengths)
    chunks, order = split_for_dp(data, dp_size=2, balance_max_tokens=64)
    assert sorted(order) == list(range(len(lengths)))
    assert MICROBATCH_SIZES_KEY not in data.metadata  # the input metadata is not changed
    flat = []
    for chunk in chunks:
        ids = chunk["sequences"][:, 0].tolist()
        sizes = chunk.metadata[MICROBATCH_SIZES_KEY]
        assert sum(sizes) == len(ids)
        start = 0
        for size in sizes:
            assert sum(lengths[i] for i in ids[start : start + size]) <= 64
            start += size
        flat += ids
    assert flat == order
    assert len({len(c.metadata[MICROBATCH_SIZES_KEY]) for c in chunks}) == 1
    tokens = [int(c["attention_mask"].sum()) for c in chunks]
    assert abs(tokens[0] - tokens[1]) <= 8
    # Per-sample outputs come back in chunk order; the restore gives the batch order.
    assert restore_sample_order([f"out{i}" for i in order], order) == [f"out{i}" for i in range(len(lengths))]
    assert restore_sample_order([], order) == []
    with pytest.raises(ValueError):
        restore_sample_order(["x"], order)


def test_worker_iterator_keeps_the_driver_packing():
    lengths = [60, 58, 55, 50, 6, 5, 4, 4, 3, 3, 2, 2]
    chunks, _ = split_for_dp(_batch(lengths), dp_size=2, balance_max_tokens=64)
    for chunk in chunks:
        iterator = TokenBasedBatchIterator(chunk, max_tokens_per_microbatch=64)
        sizes = chunk.metadata[MICROBATCH_SIZES_KEY]
        assert iterator.num_padding_microbatches == 0
        microbatches = list(iterator)
        assert [len(mb) for mb in microbatches] == sizes
        assert torch.cat([mb["sequences"][:, 0] for mb in microbatches]).tolist() == chunk["sequences"][:, 0].tolist()
        # A slice of the chunk still carries the chunk's list; the iterator then packs for itself.
        part = chunk[0:2]
        assert sum(len(mb) for mb in TokenBasedBatchIterator(part, max_tokens_per_microbatch=64)) == 2


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

    refs = MeshDispatch.dispatch(infos, "echo", data, dp_balance_max_tokens=64)
    out = WorkerOutput.cat(infos, ray.get(refs), sample_order=refs.sample_order)
    assert [x["id"] for x in out.loss_fn_outputs] == list(range(len(lengths)))

    refs = MeshDispatch.dispatch(infos, "echo", data)  # default: contiguous, no permutation
    assert refs.sample_order is None
    assert [len(s.loss_fn_outputs) for s in ray.get(refs)] == [6, 6]

    staged = MeshDispatch.stage_chunks(2, data, [(0, 6), (6, 12)], dp_balance_max_tokens=64)
    for (start, end), chunk_refs in zip([(0, 6), (6, 12)], staged):
        refs = MeshDispatch.dispatch_from_staged(infos, "echo", chunk_refs=chunk_refs)
        out = WorkerOutput.cat(infos, ray.get(refs), sample_order=chunk_refs.sample_order)
        assert [x["id"] for x in out.loss_fn_outputs] == list(range(start, end))
