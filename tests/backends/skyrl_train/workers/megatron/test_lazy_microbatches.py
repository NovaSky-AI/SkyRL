"""CPU tests for building host microbatches one at a time.

uv run --isolated --extra dev --extra skyrl-train pytest tests/backends/skyrl_train/workers/megatron/test_lazy_microbatches.py
"""

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import TensorList, TrainingInputBatch
from skyrl.backends.skyrl_train.workers.megatron.lazy_microbatches import (
    LazyMicrobatches,
    MicrobatchSummary,
    microbatch_seqlens,
    per_token_loss_counts,
    summarize_microbatches,
)
from skyrl.backends.skyrl_train.workers.worker_utils import (
    SampleBasedBatchIterator,
    TokenBasedBatchIterator,
)
from skyrl.train.dataset.preprocess import RowCollation, rows_to_training_input

SUMMARY = MicrobatchSummary(num_actions=2, num_without_routes=0)


class _Item(dict):
    """A dict a weak reference can follow."""


def _counted(length, readers=1):
    built = []

    def build(index):
        built.append(index)
        return _Item(index=index)

    return LazyMicrobatches(build, length, SUMMARY, readers=readers), built


def test_each_microbatch_is_built_on_its_first_read_and_dropped_after_its_last():
    micro_batches, built = _counted(3)
    assert built == [] and len(micro_batches) == 3
    assert [item["index"] for item in micro_batches] == [0, 1, 2]
    assert built == [0, 1, 2] and not micro_batches._cache
    with pytest.raises(IndexError):
        micro_batches[3]


@pytest.mark.parametrize("vpp", [2, 3])
def test_virtual_pipeline_chunks_share_each_build(vpp):
    micro_batches, built = _counted(4, readers=vpp)
    chunks = [iter(micro_batches) for _ in range(vpp)]
    # The interleaved schedule reads chunk by chunk, each microbatch once per chunk.
    for index in range(4):
        for chunk in chunks:
            assert next(chunk)["index"] == index
    assert built == [0, 1, 2, 3] and not micro_batches._cache


def test_a_consumed_microbatch_is_freed():
    micro_batches, _ = _counted(2)
    first = weakref.ref(micro_batches[0])
    gc.collect()
    assert first() is None


def test_a_list_of_microbatches_summarizes_itself():
    # Stand-ins for HostMicrobatch, whose module needs Megatron.
    batches = [
        SimpleNamespace(
            num_actions=5,
            rollout_expert_indices=None,
            loss=SimpleNamespace(num_tokens_global=9),
        )
        for _ in range(2)
    ]
    summary = summarize_microbatches(batches)
    assert (summary.num_actions, summary.num_without_routes, summary.num_tokens_global) == (5, 2, 9)
    lazy, _ = _counted(2)
    assert summarize_microbatches(lazy) is SUMMARY


def _batch():
    """Left-padded rows with R3 routes, as the trainer collates them."""
    g = torch.Generator().manual_seed(0)
    lengths = [(3, 2), (5, 2), (2, 2), (6, 2), (1, 2)]
    rows = {
        "prompt_token_ids": [torch.randint(1, 50, (p,), generator=g).tolist() for p, _ in lengths],
        "response_ids": [torch.randint(1, 50, (r,), generator=g).tolist() for _, r in lengths],
        "loss_masks": [[1] * r for _, r in lengths],
        "rollout_expert_indices": [np.full((p + r, 2, 2), i, dtype=np.uint8) for i, (p, r) in enumerate(lengths)],
    }
    batch = rows_to_training_input(rows, RowCollation(pad_token_id=0, max_seq_len=None, require_rollout_logprobs=False))
    batch["advantages"] = torch.randn(batch.batch_size, 2, generator=g)
    batch.metadata["response_length"] = 2
    return batch


def _assert_same(built, eager):
    assert built.keys() == eager.keys() and built.metadata == eager.metadata
    for key, value in eager.items():
        if value is None:
            assert built[key] is None, key
        elif isinstance(value, torch.Tensor):
            assert torch.equal(built[key], value), key
        else:
            # Packed fields (routes): compare their buffers.
            assert all(torch.equal(a, b) for a, b in zip(vars(built[key]).values(), vars(value).values())), key


def test_each_microbatch_built_on_request_equals_the_iterated_one():
    batch = _batch()
    iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=9)
    # Two schedule-padding microbatches, as a rank with fewer microbatches than its peers gets.
    iterator._num_padding_microbatches = 2

    rows = iterator.microbatch_rows()
    assert rows[-2:] == [None, None] and sorted(i for r in rows[:-2] for i in r) == list(range(5))
    eager = list(iterator)
    assert len(eager) == len(rows) == len(iterator)
    for index, eager_microbatch in enumerate(eager):
        built = iterator.microbatch(index)
        _assert_same(built, eager_microbatch)
        if rows[index] is not None:
            assert built.batch_size == len(rows[index])
            assert built["rollout_expert_indices"].cu_seqlens.diff().tolist() == [
                int(batch["attention_mask"][row].sum()) for row in rows[index]
            ]
        else:
            assert built.metadata["is_padding_batch"] and built["loss_mask"].count_nonzero() == 0


def test_sample_based_rows_are_its_chunks():
    batch = _batch()
    iterator = SampleBasedBatchIterator(batch, sample_batch_size=2)
    assert iterator.microbatch_rows() == [[0, 1], [2, 3], [4]]
    for index, chunk in enumerate(iterator):
        _assert_same(iterator.microbatch(index), chunk)


@pytest.mark.parametrize("controller_packed", [False, True])
def test_planned_token_counts_equal_the_built_microbatches(controller_packed):
    """The per-token loss counts the plan gives match counting each built, padded microbatch."""
    batch = _batch()
    if controller_packed:
        batch["sub_seq_lengths"] = TensorList([torch.tensor([n]) for n in batch["attention_mask"].sum(1).tolist()])
    iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=9)
    iterator._num_padding_microbatches = 1
    rows = iterator.microbatch_rows()
    padded = max(len(r) if r is not None else 1 for r in rows)

    if controller_packed:
        row_seqlens = [row.tolist() for row in batch["sub_seq_lengths"]]
    else:
        row_seqlens = [[n] for n in batch["attention_mask"].sum(1).tolist()]
    planned = per_token_loss_counts(microbatch_seqlens(row_seqlens, rows, padded), align=4, cp_size=2)

    built = []
    for index in range(len(rows)):
        microbatch = iterator.microbatch(index)
        # A padding microbatch carries no sub-sequences; its one row attends one token.
        sub = microbatch.get("sub_seq_lengths")
        seqlens = (
            [n for row in sub for n in row.tolist()]
            if sub is not None
            else microbatch["attention_mask"].sum(1).tolist()
        )
        # Padding rows attend one token, as the worker's padding gives them.
        built.append(seqlens + [1] * (padded - microbatch.batch_size))
    assert planned == per_token_loss_counts(built, align=4, cp_size=2)


def test_building_one_microbatch_reads_only_its_rows():
    batch = _batch()
    iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=9)
    first = iterator.microbatch(0)
    assert first.batch_size == len(iterator.microbatch_rows()[0]) < batch.batch_size
    assert isinstance(first, TrainingInputBatch)
