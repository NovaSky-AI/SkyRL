"""uv run --isolated --extra dev -- pytest tests/train/generators/test_stored_rollouts.py"""

import numpy as np
import pytest
import ray
import torch

from skyrl.backends.skyrl_train.rollout_shard import (
    PAYLOAD_KEYS,
    StoredSampleSupport,
    support_id_bits,
)
from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import PackedRaggedTensor
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.backends.skyrl_train.utils.sample_support import CompactSampleSupport
from skyrl.train.dataset.preprocess import RowCollation, rows_to_training_input
from skyrl.train.generators.base import (
    GeneratorInterface,
    RolloutSummary,
    StoredRolloutGroup,
    SupportCounts,
    TrajectoryID,
    sample_support_metrics,
)


class _Generator(GeneratorInterface):
    def __init__(self, output):
        self.output = output

    async def generate(self, input_batch):
        return self.output


def _rows(group, key):
    """``key``'s rows across ``group``'s trajectories, from their payloads or metadata; stored arrays as lists."""
    parts = [{**t.metadata, **ray.get(t.payload)}[key] for t in group.trajectories]
    if isinstance(parts[0], dict):
        return {name: [row for part in parts for row in part[name]] for name in parts[0]}
    return [row.tolist() if isinstance(row, np.ndarray) else row for part in parts for row in part]


def _step_wise_output():
    """Two trajectories: the first spans two step-wise rows, the second one."""
    return {
        "prompt_token_ids": [[1], [1, 2, 3], [4]],
        "response_ids": [[2, 3], [5], [6, 7, 8]],
        "rewards": [0.0, 1.0, 0.5],
        "loss_masks": [[1, 1], [1], [0, 0, 0]],
        "rollout_logprobs": [np.asarray([-0.1, -0.2]), np.asarray([-0.3]), np.asarray([-0.4, -0.5, -0.6])],
        "stop_reasons": ["tool", "complete", "complete"],
        "trajectory_ids": [TrajectoryID("u", 0), TrajectoryID("u", 0), TrajectoryID("u", 1)],
        "is_last_step": [False, True, True],
        "trajectory_time_splits": {"llm": [1.0, 2.0, 3.0], "env": [0.1, 0.2, 0.3]},
        "rollout_metrics": {"generate/custom_mean": 1.5},
        "rollout_expert_indices": None,
        "answer_outcomes": [None, "correct", "wrong"],
    }


def test_a_stored_group_round_trips_and_splits_at_trajectory_boundaries():
    output = _step_wise_output()

    stored = StoredRolloutGroup.from_output(output)

    assert [t.num_rows for t in stored.trajectories] == [2, 1]
    first = stored.trajectories[0]
    assert set(first.metadata) & PAYLOAD_KEYS == set()
    assert first.metadata["is_last_step"] == [False, True]
    assert first.metadata["trajectory_time_splits"] == {"llm": [1.0, 2.0], "env": [0.1, 0.2]}
    assert set(ray.get(first.payload)) == {"prompt_token_ids", "response_ids", "loss_masks", "rollout_logprobs"}
    assert stored.group_fields == {"rollout_metrics": {"generate/custom_mean": 1.5}, "rollout_expert_indices": None}
    assert stored.summary == RolloutSummary(rewards=[0.0, 1.0, 0.5], num_loss_tokens=[2, 1, 0])

    for key, value in output.items():
        if key in stored.group_fields:
            assert stored.group_fields[key] == value, key
        elif key == "rollout_logprobs":
            # Stored at the float32 collation casts them to.
            assert all(np.array_equal(a, np.float32(b)) for a, b in zip(_rows(stored, key), value))
        else:
            assert _rows(stored, key) == value, key


def test_each_row_is_a_trajectory_without_step_wise_rows():
    stored = StoredRolloutGroup.from_output(
        {"prompt_token_ids": [[1], [2]], "response_ids": [[3], [4]], "rewards": [1.0, 0.0], "loss_masks": [[1], [1]]}
    )

    assert [t.num_rows for t in stored.trajectories] == [1, 1]
    assert stored.trajectories[1].metadata == {"rewards": [0.0]}


def test_a_released_group_keeps_its_summary():
    stored = StoredRolloutGroup.from_output(_step_wise_output())

    stored.release()

    assert all(t.payload is None for t in stored.trajectories)
    assert stored.summary.num_loss_tokens == [2, 1, 0]


def test_a_step_wise_output_must_end_with_a_last_step():
    output = _step_wise_output()
    output["is_last_step"] = [False, True, False]

    with pytest.raises(ValueError, match="last step"):
        StoredRolloutGroup.from_output(output)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rewards, expected_rewards",
    [([1.0, 0.0], [1.0, 0.0]), ([[0.0, 0.5, 0.5], [0.0, 0.0, -1.0]], [1.0, -1.0])],
)
async def test_generate_to_ray_keeps_trainer_scalars_beside_the_payloads(rewards, expected_rewards):
    output = {
        "prompt_token_ids": [[1], [2]],
        "response_ids": [[3, 4, 5], [6, 7, 8]],
        "rewards": rewards,
        "loss_masks": [[1, 1, 0], [0, 0, 0]],
    }

    stored = await _Generator(output).generate_to_ray(None)

    assert stored.summary == RolloutSummary(rewards=expected_rewards, num_loss_tokens=[2, 0])
    assert {key: _rows(stored, key) for key in output} == output


def test_a_producer_attaches_the_payloads_it_stored_itself():
    group, payloads = StoredRolloutGroup.split_output(_step_wise_output())

    assert all(t.payload is None for t in group.trajectories)
    with pytest.raises(ValueError, match="2 trajectories"):
        group.attach_payloads([ray.put(payloads[0])])
    group.attach_payloads([ray.put(payload) for payload in payloads])
    assert _rows(group, "response_ids") == _step_wise_output()["response_ids"]


def _stored_rows(payloads):
    """Every row of ``payloads``, with support unpacked, as collation reads the stored arrays."""
    from skyrl.backends.skyrl_train.rollout_shard import unpack_support

    rows = {key: [row for payload in payloads for row in payload[key]] for key in payloads[0]}
    if "rollout_sample_support" in rows:
        rows["rollout_sample_support"] = unpack_support(rows["rollout_sample_support"])
    return rows


def _assert_same(expected, actual, path="batch"):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and torch.equal(actual, expected), path
    elif isinstance(expected, (PackedTensor, PackedRaggedTensor)):
        for name, value in vars(expected).items():
            _assert_same(value, vars(actual)[name], f"{path}.{name}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(expected, actual)):
            _assert_same(a, b, f"{path}[{i}]")
    else:
        assert actual == expected, path


def _random_output(rng, num_rows=6, vocab=1 << 17, top_k=8):
    prompts = [rng.integers(0, vocab, rng.integers(1, 9)).tolist() for _ in range(num_rows)]
    responses = [rng.integers(0, vocab, rng.integers(1, 9)).tolist() for _ in range(num_rows)]
    supports = []
    for response in responses:
        lengths = rng.integers(0, top_k + 1, len(response))
        members = rng.integers(0, vocab, int(lengths.sum())).astype(np.int32)
        supports.append(
            CompactSampleSupport.from_token_ids(
                members, lengths, top_k=top_k, logprobs=rng.normal(size=members.size).astype(np.float32)
            )
        )
    return {
        "prompt_token_ids": prompts,
        "response_ids": responses,
        "rewards": rng.normal(size=num_rows).tolist(),
        # Fractional weights as well as 0/1, and float64 logprobs that float32 cannot hold exactly.
        "loss_masks": [rng.choice([0, 1, 0.25], size=len(r)).tolist() for r in responses],
        "rollout_logprobs": [rng.normal(size=len(r)).tolist() for r in responses],
        "rollout_expert_indices": [
            rng.integers(0, 64, (len(p) + len(r) - 1, 2, 4), dtype=np.uint8) for p, r in zip(prompts, responses)
        ],
        "rollout_sample_support": supports,
    }


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("bits", [None, 17, 18])
def test_stored_payloads_collate_exactly_as_the_rows_they_store(seed, bits):
    output = _random_output(np.random.default_rng(seed))
    # Rewards are trajectory metadata, never a payload.
    rows = {key: value for key, value in output.items() if key != "rewards"}
    collation = RowCollation(pad_token_id=0, max_seq_len=None, require_rollout_logprobs=True)
    expected = rows_to_training_input(rows, collation)

    group, payloads = StoredRolloutGroup.split_output(output, support_id_bits=bits)
    stored = [ray.get(ray.put(payload)) for payload in payloads]
    actual = rows_to_training_input(_stored_rows(stored), collation)

    assert actual.keys() == expected.keys()
    for key in expected.keys():
        _assert_same(expected[key], actual[key], key)
    assert actual.metadata == expected.metadata


@pytest.mark.parametrize("seed", range(4))
def test_summary_support_counts_give_the_collated_batch_support_metrics(seed):
    output = _random_output(np.random.default_rng(seed))
    rows = {key: value for key, value in output.items() if key != "rewards"}
    collated = rows_to_training_input(
        rows, RowCollation(pad_token_id=0, max_seq_len=None, require_rollout_logprobs=False)
    )
    # The unstreamed trainer's definition: over the batch's response tokens with captured support.
    sizes = collated["rollout_sample_support"].row_lengths
    sizes = sizes[sizes > 0]
    top_k = output["rollout_sample_support"][0].top_k

    group = StoredRolloutGroup.split_output(output)[0]

    assert sample_support_metrics(group.summary.sample_support) == pytest.approx(
        {
            "generate/sample_support_size_mean": sizes.float().mean().item(),
            "generate/sample_support_full_fraction": (sizes == top_k).float().mean().item(),
        }
    )


def test_rows_without_sample_support_summarize_none():
    output = _step_wise_output()
    assert StoredRolloutGroup.split_output(output)[0].summary.sample_support is None
    assert sample_support_metrics([SupportCounts.of(None)]) == {}


def test_payloads_hold_typed_arrays_that_readers_map_read_only():
    rng = np.random.default_rng(0)
    response = rng.integers(0, 1 << 17, 40_000).tolist()
    output = {
        "prompt_token_ids": [[1, 2]],
        "response_ids": [response],
        "rewards": [1.0],
        "loss_masks": [[0.5] * len(response)],
        "rollout_logprobs": [[-0.1] * len(response)],
    }

    payload = ray.get(StoredRolloutGroup.from_output(output).trajectories[0].payload)

    assert {key: rows[0].dtype for key, rows in payload.items()} == {
        "prompt_token_ids": np.int32,
        "response_ids": np.int32,
        "loss_masks": np.float32,
        "rollout_logprobs": np.float32,
    }
    assert payload["response_ids"][0].tolist() == response
    assert set(payload["loss_masks"][0].tolist()) == {0.5}
    # Large enough for the object store, so the reader maps the stored buffer rather than a copy.
    assert not payload["response_ids"][0].flags.writeable


@pytest.mark.parametrize("bad", [-1, 2**31])
def test_a_token_id_outside_int32_is_refused(bad):
    with pytest.raises(ValueError, match="int32"):
        StoredRolloutGroup.split_output(
            {"prompt_token_ids": [[bad]], "response_ids": [[1]], "rewards": [0.0], "loss_masks": [[1]]}
        )


def test_support_member_ids_pack_and_unpack_in_order():
    output = _random_output(np.random.default_rng(1))

    _, payloads = StoredRolloutGroup.split_output(output, support_id_bits=17)
    stored = [row for payload in payloads for row in payload["rollout_sample_support"]]

    assert all(isinstance(row, StoredSampleSupport) and row.bits == 17 for row in stored)
    for original, row in zip(output["rollout_sample_support"], stored):
        unpacked = row.unpack()
        assert unpacked.token_ids.tolist() == original.token_ids.tolist()
        assert unpacked.offsets.tolist() == original.offsets.tolist()
        assert unpacked.logprobs.tolist() == original.logprobs.tolist()
        assert unpacked.top_k == original.top_k


def test_a_support_id_wider_than_the_bits_is_refused():
    support = CompactSampleSupport.from_token_ids(np.array([1 << 17], dtype=np.int32), [1], top_k=1)
    output = {
        "prompt_token_ids": [[1]],
        "response_ids": [[2]],
        "rewards": [0.0],
        "loss_masks": [[1]],
        "rollout_sample_support": [support],
    }

    with pytest.raises(ValueError, match="17 bits"):
        StoredRolloutGroup.split_output(output, support_id_bits=17)


@pytest.mark.parametrize(
    "vocab_size, bits", [(131_072, 17), (129_280, 17), (131_073, 18), (154_880, 18), (262_144, 18), (2, 1), (1, 1)]
)
def test_support_ids_take_ceil_log2_of_the_vocabulary_bits(vocab_size, bits):
    assert support_id_bits(vocab_size) == bits
