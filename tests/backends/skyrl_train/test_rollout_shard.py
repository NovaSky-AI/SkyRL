"""uv run --isolated --extra dev pytest tests/backends/skyrl_train/test_rollout_shard.py"""

import pickle
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch

from skyrl.backends.skyrl_train import rollout_shard
from skyrl.backends.skyrl_train.distributed.dispatch import (
    ActorInfo,
    MeshRank,
    WorkerOutput,
)
from skyrl.backends.skyrl_train.rollout_shard import (
    RolloutShard,
    ShardRows,
    StoredSampleSupport,
    build_training_batch,
    fetch_payloads,
)
from skyrl.backends.skyrl_train.training_batch import TensorBatch
from skyrl.backends.skyrl_train.utils.loss_normalization import (
    MinibatchLossNormalization,
)
from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import PackedRaggedTensor
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.backends.skyrl_train.utils.sample_support import CompactSampleSupport
from skyrl.backends.skyrl_train.workers import worker_utils
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.backends.skyrl_train.workers.worker_utils import (
    StoredShardBatchIterator,
    TokenBasedBatchIterator,
    select_forward_inputs,
)
from skyrl.train.dataset.preprocess import RowCollation, rows_to_training_input
from skyrl.train.generators.base import StoredRolloutGroup

COLLATION = RowCollation(pad_token_id=0, max_seq_len=None, require_rollout_logprobs=True)
R3 = replace(COLLATION, routes_expected=True)
ADVANTAGES = (0.5, 0.0, -0.25, 1.0, 2.0)


@pytest.fixture(scope="module", autouse=True)
def local_ray():
    if not ray.is_initialized():
        ray.init(num_cpus=4, include_dashboard=False, object_store_memory=200 * 1024**2)
    yield


def _read_only(values, dtype):
    array = np.asarray(values, dtype=dtype)
    array.flags.writeable = False
    return array


def _group():
    """A trajectory with two step-wise rows, then one with a single row; payloads as numpy, read-only."""
    output = {
        "prompt_token_ids": [_read_only(p, np.int32) for p in ([1], [1, 2, 3], [4])],
        "response_ids": [_read_only(r, np.int32) for r in ([2, 3], [5], [6, 7, 8])],
        "loss_masks": [_read_only(m, np.int8) for m in ([1, 1], [1], [1, 0, 1])],
        "rollout_logprobs": [_read_only(lp, np.float32) for lp in ([-0.1, -0.2], [-0.3], [-0.4, -0.5, -0.6])],
        "rewards": [0.0, 1.0, 0.5],
        "is_last_step": [False, True, True],
    }
    return StoredRolloutGroup.from_output(output), output


def _routed(support_id_bits=None):
    """Five rows with R3 routes and sample support, in four stored trajectories (rows 1-2 are one
    step-wise trajectory). Row 1 trains nothing."""
    prompts = [[1, 2], [3], [4, 5, 6], [7], [8, 9]]
    responses = [[10, 11, 12], [13], [14, 15], [16, 17, 18, 19], [20]]
    masks = [[1, 1, 1], [0], [1, 1], [1, 0, 1, 1], [1]]
    rows = {
        "prompt_token_ids": [_read_only(p, np.int32) for p in prompts],
        "response_ids": [_read_only(r, np.int32) for r in responses],
        "loss_masks": [_read_only(m, np.int8) for m in masks],
        "rollout_logprobs": [_read_only(np.linspace(-1, 0, len(r)), np.float32) for r in responses],
        "rollout_expert_indices": [
            np.full((len(p) + len(r), 2, 2), row, dtype=np.uint8) for row, (p, r) in enumerate(zip(prompts, responses))
        ],
        "rollout_sample_support": [
            CompactSampleSupport.from_token_ids(np.arange(2 * len(r), dtype=np.int32), np.full(len(r), 2), top_k=4)
            for r in responses
        ],
        "rewards": [1.0] * 5,
        "is_last_step": [True, False, True, True, True],
    }
    spans = [(0, 1), (1, 3), (3, 4), (4, 5)]
    groups = [
        StoredRolloutGroup.from_output({key: v[a:b] for key, v in rows.items()}, support_id_bits=support_id_bits)
        for a, b in spans
    ]
    return [t for group in groups for t in group.trajectories], rows


def _shard(trajectories, rows, advantages, collation=COLLATION, padding=False):
    return RolloutShard(
        payload_refs=tuple(t.payload for t in trajectories),
        payload_rows=tuple(t.num_rows for t in trajectories),
        prompt_lengths=tuple(len(p) for p in rows["prompt_token_ids"]),
        response_lengths=tuple(len(r) for r in rows["response_ids"]),
        row_advantages=tuple(advantages),
        collation=collation,
        padding=padding,
    )


def _routed_shard(support_id_bits=None):
    trajectories, rows = _routed(support_id_bits)
    return _shard(trajectories, rows, ADVANTAGES, collation=R3), trajectories


def _tensors(field):
    """A tensor field's tensors, or a packed or listed field's parts, in a fixed order."""
    if isinstance(field, torch.Tensor):
        return [field]
    if isinstance(field, PackedTensor):
        return [field.values, field.cu_seqlens]
    if isinstance(field, PackedRaggedTensor):
        return [*field.columns, field.row_offsets, field.cu_seqlens]
    return list(field.tensors)


def _assert_same_batch(actual, expected):
    assert set(actual.keys()) == set(expected.keys())
    for key in expected.keys():
        a, e = actual[key], expected[key]
        if e is None:
            assert a is None, key
            continue
        assert type(a) is type(e), key
        for mine, theirs in zip(_tensors(a), _tensors(e), strict=True):
            assert mine.dtype == theirs.dtype and torch.equal(mine, theirs), key
    assert actual.metadata == expected.metadata


def test_a_built_batch_is_the_collated_rows_with_each_rows_advantage_over_its_response():
    group, output = _group()
    shard = _shard(group.trajectories, output, (0.5, 0.5, -0.25))

    batch = build_training_batch(shard)

    expected = rows_to_training_input(
        {key: output[key] for key in ("prompt_token_ids", "response_ids", "loss_masks", "rollout_logprobs")},
        COLLATION,
    )
    for key in ("sequences", "attention_mask", "response_mask", "loss_mask", "rollout_logprobs"):
        assert torch.equal(batch[key], expected[key]), key
    assert batch["rewards"] is None
    assert torch.equal(batch["advantages"], torch.tensor([[0.5], [0.5], [-0.25]]) * expected["response_mask"])
    assert batch.metadata["response_length"] == expected["response_mask"].shape[1]


def test_a_padding_shard_trains_nothing_and_counts_no_real_rows():
    group, output = _group()
    last = {key: output[key][2:] for key in ("prompt_token_ids", "response_ids")}
    shard = _shard(group.trajectories[1:], last, (0.7,), padding=True)

    for batch in (build_training_batch(shard), ShardRows(shard).collate([0])):
        assert batch.real_sample_mask.tolist() == [False]
        assert not batch["loss_mask"].any() and not batch["advantages"].any()
        assert MinibatchLossNormalization.from_batch(batch).num_sequences == 0


@pytest.mark.parametrize("rows", [[0, 1, 2, 3, 4], [4, 0], [2], [1], [1, 3], [3, 2, 1]])
def test_a_microbatch_is_exactly_those_rows_of_the_shard_collated_whole(rows):
    shard, _ = _routed_shard()
    whole = build_training_batch(shard)

    index = torch.tensor(rows)
    expected = type(whole)({key: None if value is None else value[index] for key, value in whole.items()})
    expected.metadata = whole.metadata
    _assert_same_batch(ShardRows(shard).collate(rows), expected)


@pytest.mark.parametrize("budget", [4, 7, 100])
@pytest.mark.parametrize("forward", [False, True])
def test_stored_microbatches_match_the_collated_shards_including_padding_microbatches(monkeypatch, budget, forward):
    shard, _ = _routed_shard()
    whole = build_training_batch(shard)
    select = select_forward_inputs if forward else None
    # Another DP rank has two more microbatches, so this one pads.
    monkeypatch.setattr(TokenBasedBatchIterator, "_sync_num_microbatches", lambda self: len(self._microbatches) + 2)
    # Padding microbatches draw their ids at random; fix them so both iterators' match.
    monkeypatch.setattr(torch, "randint", lambda low, high, size, device=None: torch.zeros(size, dtype=torch.long))

    expected = TokenBasedBatchIterator(select(whole) if forward else whole, max_tokens_per_microbatch=budget)
    stored = StoredShardBatchIterator(ShardRows(shard), budget, select=select)

    assert stored._microbatches == expected._microbatches
    assert stored.num_padding_microbatches == 2
    assert (stored.num_rows, stored.num_actions) == (expected.num_rows, expected.num_actions)
    for actual, reference in zip(stored, expected, strict=True):
        _assert_same_batch(actual, reference)


@pytest.mark.parametrize("budget", [4, 100])
def test_packed_support_ids_collate_exactly_as_unpacked_ones(monkeypatch, budget):
    monkeypatch.setattr(TokenBasedBatchIterator, "_sync_num_microbatches", lambda self: len(self._microbatches) + 1)
    monkeypatch.setattr(torch, "randint", lambda low, high, size, device=None: torch.zeros(size, dtype=torch.long))
    plain, _ = _routed_shard()
    packed, trajectories = _routed_shard(support_id_bits=17)

    stored = [row for t in trajectories for row in ray.get(t.payload)["rollout_sample_support"]]
    assert all(isinstance(row, StoredSampleSupport) and row.bits == 17 for row in stored)
    _assert_same_batch(build_training_batch(packed), build_training_batch(plain))
    pairs = zip(
        StoredShardBatchIterator(ShardRows(packed), budget),
        StoredShardBatchIterator(ShardRows(plain), budget),
        strict=True,
    )
    for actual, reference in pairs:
        _assert_same_batch(actual, reference)


def test_a_microbatch_owns_its_buffers_and_never_writes_the_shared_payloads():
    shard, _ = _routed_shard()
    rows = ShardRows(shard)
    expected = pickle.loads(pickle.dumps(rows.collate([2])))
    first = rows.collate([2])
    for value in (
        first["sequences"],
        first["loss_mask"],
        first["rollout_expert_indices"].values,
        *first["rollout_sample_support"].columns,
    ):
        value.add_(1)

    _assert_same_batch(rows.collate([2]), expected)


def test_old_logprobs_reach_each_row_of_its_microbatch():
    shard, _ = _routed_shard()
    rows = ShardRows(shard)
    rows.old_logprobs = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)

    assert torch.equal(rows.collate([3, 0])["action_log_probs"], rows.old_logprobs[[3, 0]])


def test_a_microbatch_loads_only_the_payloads_holding_its_rows(monkeypatch):
    shard, trajectories = _routed_shard()
    fetched = []
    monkeypatch.setattr(rollout_shard, "fetch_payloads", lambda refs: fetched.append(refs) or ray.get(refs))

    ShardRows(shard).collate([2, 4])

    assert fetched == [[trajectories[1].payload, trajectories[3].payload]]


@pytest.mark.parametrize("pull_ahead", [1, 2])
def test_building_a_microbatch_pulls_only_the_next_few_ahead(monkeypatch, pull_ahead):
    shard, _ = _routed_shard()
    pulls = []
    monkeypatch.setattr(worker_utils, "PULL_AHEAD_MICROBATCHES", pull_ahead)
    monkeypatch.setattr(worker_utils, "pull_payloads", pulls.append)
    iterator = StoredShardBatchIterator(ShardRows(shard), 4)
    rows = iterator._microbatches
    assert len(rows) >= 3

    for index, _ in enumerate(iterator):
        # Each microbatch within reach is pulled once, in order, and none beyond it.
        reach = min(index + pull_ahead, len(rows) - 1)
        assert pulls == [ShardRows(shard).payload_refs(r) for r in rows[1 : reach + 1]]


def test_pulling_ahead_copies_the_payloads_without_holding_them():
    ref = ray.put(np.zeros(1 << 20, dtype=np.uint8))
    rollout_shard.pull_payloads([ref])
    rollout_shard._PULLS.shutdown(wait=True)
    rollout_shard._PULLS = None
    assert ray.get(ref).nbytes == 1 << 20


def test_a_stalled_fetch_raises_instead_of_waiting_for_ever(monkeypatch):
    @ray.remote
    def never():
        time.sleep(60)

    monkeypatch.setattr(rollout_shard, "SKYRL_PAYLOAD_FETCH_TIMEOUT_S", 0.5)
    with pytest.raises(TimeoutError, match="SKYRL_PAYLOAD_FETCH_TIMEOUT_S"):
        fetch_payloads([never.remote()])


def test_microbatch_outputs_restore_input_order_without_a_collated_batch():
    shard, _ = _routed_shard()
    iterator = StoredShardBatchIterator(ShardRows(shard), 4)

    outputs = [TensorBatch({"output": mb["sequences"][:, -1:].float()}) for mb in iterator]

    expected = build_training_batch(shard)["sequences"][:, -1:].float()
    assert torch.equal(iterator.reorder_and_combine_batches(outputs)["output"], expected)


@ray.remote
class _Worker:
    def forward_backward_rollout_shard(self, shard, loss_normalization, compute_old_logprobs):
        # The worker loads its own rows, a microbatch at a time.
        rows = ShardRows(shard)
        count = sum(rows.collate([row]).batch_size for row in range(rows.num_rows))
        return WorkerOutput(loss_fn_outputs=[{"rows": count}], metrics={"rows": float(count)})

    def save_memory_snapshot(self, tag):
        return None


class _Group:
    """Four workers: DP=2 by TP=2."""

    def __init__(self):
        layout = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (dp, tp)
        self.actor_infos = [
            ActorInfo(_Worker.remote(), MeshRank(dp=dp, sp=0, tp=tp, pp=0, world_size=4, dp_size=2, pp_size=1))
            for dp, tp in layout
        ]

    def async_run_ray_method(self, dispatch, method, **kwargs):
        return [getattr(info.handle, method).remote(**kwargs) for info in self.actor_infos]


def _dispatch_and_shards():
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(placement=SimpleNamespace(colocate_all=False, colocate_policy_ref=False))
    )
    group, output = _group()
    first, second = group.trajectories

    def rows(a, b):
        return {key: output[key][a:b] for key in ("prompt_token_ids", "response_ids")}

    shards = [_shard([first], rows(0, 2), (1.0, 1.0)), _shard([second], rows(2, 3), (1.0,))]
    return WorkerDispatch(cfg, policy_actor_group=_Group()), shards


def test_each_worker_trains_on_its_dp_ranks_shard_and_the_trainer_puts_no_batch(monkeypatch):
    dispatch, shards = _dispatch_and_shards()
    monkeypatch.setattr(ray, "put", lambda *a, **k: pytest.fail("the trainer put a batch into the object store"))

    output = dispatch.forward_backward_rollout_shards(
        "policy",
        shards,
        loss_normalization=MinibatchLossNormalization(3, 4, 12, 3, 3),
        compute_old_logprobs=False,
    )

    assert [item["rows"] for item in output.loss_fn_outputs] == [2, 1]
