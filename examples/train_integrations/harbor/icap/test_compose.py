"""The adapter from capture's exported branches to a GeneratorOutput.

Runs without a cluster, an engine, or capture itself: the input is the row
shape `token-samples` emits, written out by hand.

    uv run --with pytest python -m pytest examples/train_integrations/harbor/icap/ -q
"""

from __future__ import annotations

import pytest

from .compose import compose, split_row


def row(input_ids, loss_mask, *, logprobs=None, path_id="p0", **extra):
    """One `token-samples` row, in the shape capture emits."""
    return {
        "schema_version": 2,
        "path_id": path_id,
        "trajectory_id": "tr_1",
        "node_ids": [],
        "abandoned": False,
        "labels": [],
        "annotations": {},
        "trainable_count": sum(loss_mask),
        "input_ids": list(input_ids),
        "loss_mask": list(loss_mask),
        "rollout_logprobs": list(logprobs if logprobs is not None else [0.0] * len(input_ids)),
        "rollout_expert_indices": None,
        "stop_reason": "stop",
        "tokenizer": "builtin",
        **extra,
    }


# -- splitting one branch ---------------------------------------------------
def test_the_split_is_the_first_trainable_token_not_the_first_turn():
    """Prompt is everything before anything this row may learn from."""
    split = split_row(row([1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 1, 0]))
    assert split["prompt_token_ids"] == [1, 2, 3]
    assert split["response_ids"] == [4, 5, 6]
    assert split["loss_mask"] == [1, 1, 0]
    assert len(split["loss_mask"]) == len(split["response_ids"])


def test_a_branch_with_nothing_trainable_is_dropped():
    """Capture never drops rows; it masks them. Here they carry no gradient."""
    assert split_row(row([1, 2, 3], [0, 0, 0])) is None


def test_a_row_whose_arrays_disagree_is_an_error():
    """Misalignment here would train on the wrong positions, silently."""
    broken = row([1, 2, 3], [0, 1, 1])
    broken["rollout_logprobs"] = [0.0, 0.0]
    with pytest.raises(ValueError, match="inconsistent"):
        split_row(broken)


def test_logprobs_follow_the_same_split():
    split = split_row(row([1, 2, 3, 4], [0, 0, 1, 1], logprobs=[0.0, 0.0, -0.5, -1.5]))
    assert split["rollout_logprobs"] == [-0.5, -1.5]


# -- composing a batch ------------------------------------------------------
def _compose(exports, stop_reasons=None, step_wise=True, rewards=None):
    count = len(exports)
    return compose(
        exports,
        trajectory_ids=[f"t{i}" for i in range(count)],
        rewards=rewards if rewards is not None else [1.0] * count,
        stop_reasons=stop_reasons or ["complete"] * count,
        step_wise=step_wise,
    )


def test_a_linear_rollout_is_one_row():
    out = _compose([[row([1, 2, 3], [0, 1, 1])]])
    assert out["response_ids"] == [[2, 3]]
    assert out["loss_masks"] == [[1, 1]]


def test_a_summarizing_trajectory_yields_a_row_per_branch():
    """A rewritten history branches, so one trial can produce several samples."""
    out = _compose([[row([1, 2], [0, 1], path_id="a"), row([1, 3, 4], [0, 0, 1], path_id="b")]])
    assert len(out["response_ids"]) == 2
    # One reward covers the whole tree, so every branch carries it.
    assert out["rewards"] == [1.0, 1.0]
    assert out["trajectory_ids"] == ["t0", "t0"]


def test_branching_is_masked_when_step_wise_is_off():
    """There is no single row for a branched trajectory, so do not guess one.

    This is the case a summarizing agent hits every time, which is why
    summarization wants step-wise on.
    """
    out = _compose([[row([1, 2], [0, 1], path_id="a"), row([1, 3], [0, 1], path_id="b")]], step_wise=False)
    assert len(out["response_ids"]) == 1
    assert out["loss_masks"] == [[0]], "the row must carry no gradient"


def test_one_branch_survives_step_wise_off():
    out = _compose([[row([1, 2, 3], [0, 1, 1])]], step_wise=False)
    assert out["loss_masks"] == [[1, 1]]


@pytest.mark.parametrize("reason", ["agent_timeout", "error"])
def test_a_failed_trial_is_masked_not_dropped(reason):
    """Only the harness knows the trial failed; the batch keeps its shape."""
    out = _compose([[row([1, 2, 3], [0, 1, 1])]], stop_reasons=[reason])
    assert len(out["response_ids"]) == 1
    assert out["loss_masks"] == [[0]]
    assert out["stop_reasons"] == [reason]


def test_every_trajectory_contributes_at_least_one_row():
    """A masked trajectory must not vanish from the batch."""
    out = _compose(
        [[], [row([1, 2], [0, 1])], [row([9], [0])]],
        rewards=[0.0, 1.0, 0.5],
    )
    assert len(out["response_ids"]) == 3
    assert out["trajectory_ids"] == ["t0", "t1", "t2"]
    assert out["rewards"] == [0.0, 1.0, 0.5]


def test_expert_indices_are_omitted_when_no_branch_has_them():
    assert _compose([[row([1, 2], [0, 1])]])["rollout_expert_indices"] is None


def test_expert_indices_survive_when_present():
    with_experts = row([1, 2], [0, 1], rollout_expert_indices=[[[0, 3]], [[1, 2]]])
    assert _compose([[with_experts]])["rollout_expert_indices"] == [[[[0, 3]], [[1, 2]]]]


def test_mismatched_input_lengths_are_refused():
    """A silent zip() truncation here would misalign rewards and rollouts."""
    with pytest.raises(ValueError, match="same length"):
        compose(
            [[row([1, 2], [0, 1])]],
            trajectory_ids=["t0", "t1"],
            rewards=[1.0],
            stop_reasons=["complete"],
            step_wise=True,
        )
