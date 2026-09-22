"""The adapter from capture's exported branches to a GeneratorOutput.

Runs without a cluster, an engine, or capture itself: the input is the row
shape `token-samples` emits, written out by hand.

    uv run --with pytest python -m pytest examples/train_integrations/harbor/icap/ -q
"""

from __future__ import annotations

import pytest

from examples.train_integrations.harbor.icap.compose import (
    compose,
    split_row,
    stepwise_rows,
)


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


def test_a_rollouts_paths_are_grouped_contiguously_and_the_last_is_marked():
    """The grouping the trainer computes one advantage from.

    `is_last_step` is borrowed here: it means "last path of this rollout", not
    "last turn of a conversation". Every path carries the same reward, so
    which one is marked does not change the advantage -- it decides which row
    step-wise evaluation keeps.
    """
    out = _compose(
        [
            [row([1, 2], [0, 1], path_id="a"), row([1, 3, 4], [0, 0, 1], path_id="b")],
            [row([5, 6], [0, 1], path_id="c")],
        ],
        rewards=[1.0, 0.25],
    )
    assert out["trajectory_ids"] == ["t0", "t0", "t1"], "contiguous, one id per rollout"
    assert out["is_last_step"] == [False, True, True]
    assert out["rewards"] == [1.0, 1.0, 0.25], "one reward per rollout, on every path"


def test_a_masked_rollout_is_still_marked_as_its_own_last_step():
    """Its placeholder is the only row it has, so it is also the last one --
    otherwise step-wise evaluation would drop the rollout entirely."""
    out = _compose([[]], rewards=[0.0])
    assert out["is_last_step"] == [True]


def test_the_step_wise_argument_is_ignored(caplog):
    """It used to mask a branched trajectory, which was always a workaround
    for a contract that could not express a group. Deprecated, not honoured."""
    rows = [row([1, 2], [0, 1], path_id="a"), row([1, 3], [0, 1], path_id="b")]
    off = _compose([rows], step_wise=False)
    on = _compose([rows], step_wise=True)
    assert len(off["response_ids"]) == 2, "both paths survive either way"
    assert off["response_ids"] == on["response_ids"]
    assert off["loss_masks"] == on["loss_masks"] == [[1], [1]]
    assert "ignored" in caplog.text


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


# -- the configuration this output shape requires -----------------------------
class _Cfg:
    def __init__(self, step_wise_trajectories=True, merge_stepwise_output=False):
        self.step_wise_trajectories = step_wise_trajectories
        self.merge_stepwise_output = merge_stepwise_output


def _require(cfg):
    from examples.train_integrations.harbor.icap.harbor_generator import (
        _require_grouped_output,
    )

    return _require_grouped_output(cfg)


def test_a_run_that_cannot_hold_several_rows_per_rollout_is_refused():
    """`step_wise_trajectories=false` makes the trainer assert one response per
    prompt -- and it does that after a whole batch of Harbor trials has been
    run and thrown away. The cost of finding out late is the reason this is
    checked at construction."""
    with pytest.raises(ValueError, match="step_wise_trajectories=true"):
        _require(_Cfg(step_wise_trajectories=False))


def test_prefix_merging_is_refused():
    """These rows are complete paths, not sequential turns. Merging them could
    fuse two distinct paths that merely look like a prefix of one another."""
    with pytest.raises(ValueError, match="merge_stepwise_output=false"):
        _require(_Cfg(merge_stepwise_output=True))


def test_the_configuration_this_integration_wants_is_accepted():
    assert _require(_Cfg()) is None


# -- parity against the harness-side collector --------------------------------
def test_a_path_cuts_back_into_the_per_turn_rows_the_sibling_emits():
    """The sibling integration emits, per turn, the prompt before that turn and
    that turn's completion with an all-ones mask. A captured path holds the
    same information as one sequence plus a mask, so parity is a token-for-token
    comparison once the path is cut at its trainable runs."""
    path = row(
        [1, 2, 3, 4, 5, 6, 7],
        [0, 0, 1, 1, 0, 1, 0],
        logprobs=[0.0, 0.0, -0.1, -0.2, 0.0, -0.3, 0.0],
    )
    turns = stepwise_rows(path)
    assert len(turns) == 2, "two sampled spans, two turns"
    assert turns[0]["prompt_token_ids"] == [1, 2]
    assert turns[0]["response_ids"] == [3, 4]
    assert turns[0]["loss_mask"] == [1, 1]
    assert turns[0]["rollout_logprobs"] == [-0.1, -0.2]
    assert turns[1]["prompt_token_ids"] == [1, 2, 3, 4, 5], "the prompt is the whole prefix"
    assert turns[1]["response_ids"] == [6]


def test_the_turns_tile_the_trainable_tokens_exactly():
    """Every trainable token belongs to exactly one turn, and prompts grow --
    which is what makes them successive turns rather than an arbitrary cut."""
    path = row([1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 1, 0, 0, 1, 0, 1])
    turns = stepwise_rows(path)
    assert sum(len(turn["response_ids"]) for turn in turns) == sum(
        int(value) for value in path["loss_mask"]
    )
    lengths = [len(turn["prompt_token_ids"]) for turn in turns]
    assert lengths == sorted(lengths) and len(set(lengths)) == len(lengths)


def test_a_path_with_nothing_trainable_has_no_turns():
    assert stepwise_rows(row([1, 2], [0, 0])) == []


def test_a_path_whose_arrays_disagree_is_an_error():
    with pytest.raises(ValueError, match="inconsistent"):
        stepwise_rows({"path_id": "p", "input_ids": [1, 2, 3], "loss_mask": [0, 1]})
