"""
Unit tests for prefix-aware step-wise merge (issue #1277).

Run: uv run --isolated --extra dev pytest tests/train/test_step_wise_merge.py -v
"""

from skyrl.train.step_wise_merge import (
    _is_prefix,
    merge_step_wise_turns_for_trajectory,
)


def test_is_prefix():
    assert _is_prefix([], [1, 2, 3]) is True
    assert _is_prefix([1, 2], [1, 2, 3]) is True
    assert _is_prefix([1, 2, 3], [1, 2, 3]) is True
    assert _is_prefix([1, 2, 3], [1, 2]) is False
    assert _is_prefix([1, 99], [1, 2, 3]) is False
    assert _is_prefix([1], [1]) is True


def test_merge_works():
    """Turn2 observation = Turn1(obs+act) + extra → 1 merged sample."""
    # Turn 1: prompt [10,20,30], response [40,50] → full sequence [10,20,30,40,50]
    # Turn 2: prompt [10,20,30,40,50,60,70] (prefix match + extra 60,70)
    prompt_ids = [
        [10, 20, 30],
        [10, 20, 30, 40, 50, 60, 70],
    ]
    response_ids = [[40, 50], [80, 90]]
    rewards = [[0.1, 0.2], [0.3, 0.4]]
    loss_masks = [[1, 1], [1, 1]]
    is_last_step = [False, True]

    merged, mismatch_count = merge_step_wise_turns_for_trajectory(
        prompt_token_ids=prompt_ids,
        response_ids=response_ids,
        rewards=rewards,
        loss_masks=loss_masks,
        is_last_step=is_last_step,
    )

    assert len(merged) == 1
    assert mismatch_count == 0
    # prompt = first-turn obs only; response = resp1 + delta_ob2 + resp2 (no duplicate tokens)
    assert merged[0].prompt_token_ids == [10, 20, 30]
    assert merged[0].response_ids == [40, 50, 60, 70, 80, 90]
    assert merged[0].is_last_step is True
    # prompt + response = full sequence with no overlap
    assert merged[0].prompt_token_ids + merged[0].response_ids == [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # Delta observation tokens: no loss / zero reward; action tokens keep per-turn values
    assert merged[0].loss_masks == [1, 1, 0, 0, 1, 1]
    assert merged[0].rewards == [0.1, 0.2, 0.0, 0.0, 0.3, 0.4]


def test_merge_works_with_rollout_logprobs():
    prompt_ids = [
        [10, 20, 30],
        [10, 20, 30, 40, 50, 60, 70],
    ]
    response_ids = [[40, 50], [80, 90]]
    rewards = [[0.0, 0.0], [0.0, 0.0]]
    loss_masks = [[1, 1], [1, 1]]
    is_last_step = [False, True]
    logprobs = [[-0.1, -0.2], [-0.3, -0.4]]

    merged, mismatch_count = merge_step_wise_turns_for_trajectory(
        prompt_token_ids=prompt_ids,
        response_ids=response_ids,
        rewards=rewards,
        loss_masks=loss_masks,
        is_last_step=is_last_step,
        rollout_logprobs=logprobs,
    )

    assert mismatch_count == 0
    assert merged[0].rollout_logprobs is not None
    assert merged[0].rollout_logprobs == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4]


def test_prefix_mismatch():
    """Turn2 observation does not start with previous sequence → 2 samples."""
    prompt_ids = [
        [10, 20, 30],
        [99, 88, 77],
    ]
    response_ids = [[40, 50], [11, 22]]
    rewards = [[0.0, 0.0], [0.0, 0.0]]
    loss_masks = [[1, 1], [1, 1]]
    is_last_step = [False, True]

    merged, mismatch_count = merge_step_wise_turns_for_trajectory(
        prompt_token_ids=prompt_ids,
        response_ids=response_ids,
        rewards=rewards,
        loss_masks=loss_masks,
        is_last_step=is_last_step,
    )

    assert len(merged) == 2
    assert mismatch_count == 1
    assert merged[0].prompt_token_ids == [10, 20, 30]
    assert merged[0].response_ids == [40, 50]
    assert merged[0].is_last_step is False
    assert merged[1].prompt_token_ids == [99, 88, 77]
    assert merged[1].response_ids == [11, 22]
    assert merged[1].is_last_step is True


def test_partial_merge():
    """Turn1→Turn2 merge, Turn3 mismatches → 2 samples."""
    # Turn 1: [1,2,3] + [4,5]
    # Turn 2: [1,2,3,4,5,6,7] (prefix) + [8,9]
    # Turn 3: [100,200] (mismatch) + [11,22]
    prompt_ids = [
        [1, 2, 3],
        [1, 2, 3, 4, 5, 6, 7],
        [100, 200],
    ]
    response_ids = [[4, 5], [8, 9], [11, 22]]
    rewards = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    loss_masks = [[1, 1], [1, 1], [1, 1]]
    is_last_step = [False, False, True]

    merged, mismatch_count = merge_step_wise_turns_for_trajectory(
        prompt_token_ids=prompt_ids,
        response_ids=response_ids,
        rewards=rewards,
        loss_masks=loss_masks,
        is_last_step=is_last_step,
    )

    assert len(merged) == 2
    assert mismatch_count == 1
    # First sample: merged turn1+2 (prompt = initial obs; response = resp1 + delta_ob2 + resp2)
    assert merged[0].prompt_token_ids == [1, 2, 3]
    assert merged[0].response_ids == [4, 5, 6, 7, 8, 9]
    assert merged[0].is_last_step is False
    # Second sample: turn3 only
    assert merged[1].prompt_token_ids == [100, 200]
    assert merged[1].response_ids == [11, 22]
    assert merged[1].is_last_step is True
