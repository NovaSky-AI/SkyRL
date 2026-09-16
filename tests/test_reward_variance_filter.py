import pytest

from skyrl.train.reward_variance_filter import reward_variance_filter


def test_selects_smallest_stable_variance_mass_prefix():
    rewards = [-3.0, 3.0, -2.0, 2.0, -1.0, 1.0]
    uids = ["high", "high", "medium", "medium", "low", "low"]

    kept_indices, metrics = reward_variance_filter(rewards, uids, top_p=0.8, selection_eps=0.0)

    assert kept_indices == [0, 1, 2, 3]
    assert metrics["selected_variance_ratio"] == pytest.approx(26.0 / 28.0)


def test_ties_use_first_seen_group_order():
    rewards = [-1.0, 1.0, -1.0, 1.0]
    uids = ["first", "first", "second", "second"]

    kept_indices, _ = reward_variance_filter(rewards, uids, top_p=0.5, selection_eps=0.0)

    assert kept_indices == [0, 1]


def test_all_zero_batch_and_already_masked_samples():
    excluded, _ = reward_variance_filter([1.0, 1.0, 2.0, 2.0], ["a", "a", "b", "b"])
    included, _ = reward_variance_filter([1.0, 1.0, 2.0, 2.0], ["a", "a", "b", "b"], include_zero=True)
    masked, _ = reward_variance_filter(
        [-2.0, 2.0, 100.0, 1.0, 1.0],
        ["signal", "signal", "signal", "flat", "flat"],
        loss_masks=[[1], [1], [0], [1], [1]],
        top_p=1.0,
        selection_eps=0.0,
    )

    assert excluded == []
    assert included == [0, 1, 2, 3]
    assert masked == [0, 1, 2]


def test_top_k_keeps_fixed_number_of_highest_variance_groups():
    rewards = [-3.0, 3.0, -2.0, 2.0, -1.0, 1.0]
    uids = ["high", "high", "medium", "medium", "low", "low"]

    kept_indices, metrics = reward_variance_filter(rewards, uids, strategy="top_k", top_k=2)

    assert kept_indices == [0, 1, 2, 3]
    assert metrics["num_kept_groups"] == 2.0


def test_top_k_excludes_zero_variance_groups_unless_included():
    rewards = [-1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    uids = ["signal", "signal", "flat_a", "flat_a", "flat_b", "flat_b"]

    excluded, _ = reward_variance_filter(rewards, uids, strategy="top_k", top_k=2)
    included, _ = reward_variance_filter(rewards, uids, strategy="top_k", top_k=2, include_zero=True)

    assert excluded == [0, 1]
    assert included == [0, 1, 2, 3]


def test_selection_epsilon_can_drop_a_near_zero_signal_batch():
    kept_indices, _ = reward_variance_filter([0.0, 0.1], ["tiny", "tiny"], top_p=0.9, selection_eps=0.01)

    assert kept_indices == []
