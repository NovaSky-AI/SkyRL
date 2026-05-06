import pytest

from skyrl.train.utils.reward_shaping import apply_dapo_soft_overlong_punishment


def test_apply_dapo_soft_overlong_punishment_scalar_response_rewards():
    rewards = apply_dapo_soft_overlong_punishment(
        response_ids=[[1, 2, 3], [4] * 7, [5] * 11],
        rewards=[1.0, 2.0, 3.0],
        overlong_buffer_len=4,
        overlong_buffer_penalty_factor=2.0,
        max_response_length=10,
    )

    assert rewards == pytest.approx([1.0, 1.5, 0.0])


def test_apply_dapo_soft_overlong_punishment_per_token_rewards():
    rewards = apply_dapo_soft_overlong_punishment(
        response_ids=[[1] * 7, [2] * 11],
        rewards=[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
        overlong_buffer_len=4,
        overlong_buffer_penalty_factor=2.0,
        max_response_length=10,
    )

    assert rewards[0] == pytest.approx([0.0, 0.0, 0.5])
    assert rewards[1] == pytest.approx([0.0, 0.0, 0.0])


def test_apply_dapo_soft_overlong_punishment_per_sample_caps():
    rewards = apply_dapo_soft_overlong_punishment(
        response_ids=[[1] * 6, [2] * 6],
        rewards=[1.0, 1.0],
        overlong_buffer_len=4,
        overlong_buffer_penalty_factor=1.0,
        max_response_lengths=[8, 5],
    )

    assert rewards == pytest.approx([0.5, 0.0])


def test_apply_dapo_soft_overlong_punishment_requires_one_cap_source():
    with pytest.raises(ValueError, match="exactly one"):
        apply_dapo_soft_overlong_punishment(
            response_ids=[[1, 2, 3]],
            rewards=[1.0],
            overlong_buffer_len=4,
            overlong_buffer_penalty_factor=1.0,
        )


def test_apply_dapo_soft_overlong_punishment_validates_buffer_len():
    with pytest.raises(ValueError, match="overlong_buffer_len"):
        apply_dapo_soft_overlong_punishment(
            response_ids=[[1, 2, 3]],
            rewards=[1.0],
            overlong_buffer_len=0,
            overlong_buffer_penalty_factor=1.0,
            max_response_length=10,
        )


def test_apply_dapo_soft_overlong_punishment_validates_buffer_len_against_per_sample_cap():
    with pytest.raises(ValueError, match="sample_max_response_length"):
        apply_dapo_soft_overlong_punishment(
            response_ids=[[1, 2, 3]],
            rewards=[1.0],
            overlong_buffer_len=4,
            overlong_buffer_penalty_factor=1.0,
            max_response_lengths=[3],
        )
