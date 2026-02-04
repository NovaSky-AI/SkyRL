"""Tests for reward_metrics module."""

from skyrl_train.metrics.reward_metrics import (
    compute_pass_at_n,
    compute_per_group_metrics,
    compute_reward_metrics,
    compute_variance_per_prompt,
    sanitize_metric_key,
)


class TestSanitizeMetricKey:
    def test_replaces_slash(self):
        assert sanitize_metric_key("foo/bar") == "foo_bar"

    def test_multiple_slashes(self):
        assert sanitize_metric_key("a/b/c") == "a_b_c"

    def test_no_slash(self):
        assert sanitize_metric_key("foobar") == "foobar"

    def test_empty_string(self):
        assert sanitize_metric_key("") == ""


class TestComputePassAtN:
    def test_all_pass(self):
        """All prompts have at least one positive reward."""
        rewards = [1.0, 1.0, 1.0, 1.0]
        uids = ["a", "a", "b", "b"]
        assert compute_pass_at_n(rewards, uids) == 1.0

    def test_none_pass(self):
        """No prompts have positive rewards."""
        rewards = [0.0, 0.0, 0.0, 0.0]
        uids = ["a", "a", "b", "b"]
        assert compute_pass_at_n(rewards, uids) == 0.0

    def test_partial_pass(self):
        """uid 'a' has one pass, uid 'b' has none -> 50% pass rate."""
        rewards = [1.0, 0.0, 0.0, 0.0]
        uids = ["a", "a", "b", "b"]
        assert compute_pass_at_n(rewards, uids) == 0.5

    def test_any_positive_counts(self):
        """uid 'a' has one pass out of 4 rollouts, still counts as pass."""
        rewards = [0.0, 0.0, 0.0, 1.0]
        uids = ["a", "a", "a", "a"]
        assert compute_pass_at_n(rewards, uids) == 1.0

    def test_empty_input(self):
        """Empty input returns 0.0."""
        assert compute_pass_at_n([], []) == 0.0

    def test_single_prompt_pass(self):
        """Single prompt with positive reward."""
        rewards = [1.0]
        uids = ["a"]
        assert compute_pass_at_n(rewards, uids) == 1.0

    def test_single_prompt_fail(self):
        """Single prompt with zero reward."""
        rewards = [0.0]
        uids = ["a"]
        assert compute_pass_at_n(rewards, uids) == 0.0

    def test_fractional_positive_reward(self):
        """Fractional positive reward counts as pass."""
        rewards = [0.5, 0.0]
        uids = ["a", "a"]
        assert compute_pass_at_n(rewards, uids) == 1.0

    def test_negative_reward_not_pass(self):
        """Negative rewards don't count as pass."""
        rewards = [-1.0, -0.5]
        uids = ["a", "a"]
        assert compute_pass_at_n(rewards, uids) == 0.0


class TestComputeVariancePerPrompt:
    """Tests for within-prompt variance (GRPO learning signal)."""

    def test_high_variance(self):
        """High variance within prompt = good learning signal."""
        rewards = [1.0, 0.0, 1.0, 0.0]  # All for same prompt
        uids = ["a", "a", "a", "a"]
        variance = compute_variance_per_prompt(rewards, uids)
        assert variance == 0.25  # np.var([1, 0, 1, 0]) = 0.25

    def test_zero_variance(self):
        """Zero variance = no learning signal."""
        rewards = [1.0, 1.0, 1.0, 1.0]  # All same reward
        uids = ["a", "a", "a", "a"]
        variance = compute_variance_per_prompt(rewards, uids)
        assert variance == 0.0

    def test_multiple_prompts_average(self):
        """Variance is averaged across prompts."""
        # Prompt "a": rewards [1.0, 0.0] -> variance = 0.25
        # Prompt "b": rewards [1.0, 1.0] -> variance = 0.0
        # Mean variance = (0.25 + 0.0) / 2 = 0.125
        rewards = [1.0, 0.0, 1.0, 1.0]
        uids = ["a", "a", "b", "b"]
        variance = compute_variance_per_prompt(rewards, uids)
        assert variance == 0.125

    def test_single_sample_per_prompt_excluded(self):
        """Prompts with single sample are excluded (can't compute variance)."""
        rewards = [1.0, 0.0, 0.5]  # Three prompts with 1 sample each
        uids = ["a", "b", "c"]
        variance = compute_variance_per_prompt(rewards, uids)
        assert variance == 0.0  # No prompts with 2+ samples

    def test_empty_input(self):
        """Empty input returns 0.0."""
        assert compute_variance_per_prompt([], []) == 0.0

    def test_saturated_environment(self):
        """Saturated env (all 1.0) has zero variance = no learning signal."""
        rewards = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        uids = ["a", "a", "b", "b", "c", "c"]
        variance = compute_variance_per_prompt(rewards, uids)
        assert variance == 0.0  # All prompts have zero variance

    def test_mixed_variance_across_prompts(self):
        """Some prompts have variance, others don't."""
        # Prompt "a": [1.0, 0.0] -> var = 0.25
        # Prompt "b": [0.5, 0.5] -> var = 0.0
        # Prompt "c": [1.0, 0.5] -> var = 0.0625
        # Mean = (0.25 + 0.0 + 0.0625) / 3 = 0.104166...
        rewards = [1.0, 0.0, 0.5, 0.5, 1.0, 0.5]
        uids = ["a", "a", "b", "b", "c", "c"]
        variance = compute_variance_per_prompt(rewards, uids)
        assert abs(variance - 0.10416666666666667) < 1e-10


class TestComputeRewardMetrics:
    def test_basic_metrics(self):
        """Test basic metric calculation with mixed rewards."""
        rewards = [1.0, 0.0, 0.5, 0.0]
        uids = ["a", "a", "b", "b"]
        metrics = compute_reward_metrics(rewards, uids, n_samples_per_prompt=2)

        assert metrics["pass_at_2"] == 1.0  # both uids have positive
        assert metrics["mean_positive_reward"] == 0.75  # (1+0.5)/2
        # variance: uid "a" [1,0] var=0.25, uid "b" [0.5,0] var=0.0625 -> mean=0.15625
        assert abs(metrics["variance_per_prompt"] - 0.15625) < 1e-10

    def test_no_positive_rewards(self):
        """All zero rewards."""
        rewards = [0.0, 0.0]
        uids = ["a", "b"]
        metrics = compute_reward_metrics(rewards, uids, n_samples_per_prompt=1)

        assert metrics["pass_at_1"] == 0.0
        assert metrics["mean_positive_reward"] == 0.0
        assert metrics["variance_per_prompt"] == 0.0  # single sample per uid

    def test_all_positive_rewards(self):
        """All positive rewards with variance."""
        rewards = [1.0, 0.5, 0.75, 0.25]
        uids = ["a", "a", "b", "b"]
        metrics = compute_reward_metrics(rewards, uids, n_samples_per_prompt=2)

        assert metrics["pass_at_2"] == 1.0  # all pass
        assert metrics["mean_positive_reward"] == 0.625  # (1+0.5+0.75+0.25)/4
        # variance: uid "a" [1,0.5] var=0.0625, uid "b" [0.75,0.25] var=0.0625 -> mean=0.0625
        assert abs(metrics["variance_per_prompt"] - 0.0625) < 1e-10

    def test_empty_input(self):
        """Empty input returns zeros."""
        metrics = compute_reward_metrics([], [], n_samples_per_prompt=1)

        assert metrics["pass_at_1"] == 0.0
        assert metrics["mean_positive_reward"] == 0.0
        assert metrics["variance_per_prompt"] == 0.0

    def test_different_n_samples(self):
        """Test that n_samples_per_prompt is reflected in key name."""
        rewards = [1.0, 0.0]
        uids = ["a", "a"]

        metrics_4 = compute_reward_metrics(rewards, uids, n_samples_per_prompt=4)
        assert "pass_at_4" in metrics_4
        assert "variance_per_prompt" in metrics_4

        metrics_8 = compute_reward_metrics(rewards, uids, n_samples_per_prompt=8)
        assert "pass_at_8" in metrics_8
        assert "variance_per_prompt" in metrics_8


class TestComputePerGroupMetrics:
    def test_per_env_grouping(self):
        """Test grouping by environment."""
        rewards = [1.0, 0.0, 0.5, 0.0]
        uids = ["task1", "task1", "task2", "task2"]
        groups = ["github", "github", "booking", "booking"]

        metrics = compute_per_group_metrics(rewards, uids, groups, n_samples_per_prompt=2, prefix="reward")

        # Check all expected keys exist
        assert "reward/github/variance_per_prompt" in metrics
        assert "reward/github/pass_at_2" in metrics
        assert "reward/github/mean_positive_reward" in metrics
        assert "reward/booking/variance_per_prompt" in metrics
        assert "reward/booking/pass_at_2" in metrics
        assert "reward/booking/mean_positive_reward" in metrics

        # Check values
        assert metrics["reward/github/pass_at_2"] == 1.0  # task1 passes
        assert metrics["reward/github/variance_per_prompt"] == 0.25  # [1,0] var=0.25
        assert metrics["reward/booking/pass_at_2"] == 1.0  # task2 passes
        assert metrics["reward/booking/variance_per_prompt"] == 0.0625  # [0.5,0] var=0.0625

    def test_eval_prefix(self):
        """Test with eval prefix instead of reward."""
        rewards = [1.0, 0.0]
        uids = ["a", "a"]
        groups = ["env1", "env1"]

        metrics = compute_per_group_metrics(rewards, uids, groups, n_samples_per_prompt=2, prefix="eval")

        assert "eval/env1/variance_per_prompt" in metrics
        assert "eval/env1/pass_at_2" in metrics
        assert "eval/env1/mean_positive_reward" in metrics

    def test_group_with_slash_sanitized(self):
        """Test that group names with slashes are sanitized."""
        rewards = [1.0, 0.0]
        uids = ["a", "a"]
        groups = ["foo/bar", "foo/bar"]

        metrics = compute_per_group_metrics(rewards, uids, groups, n_samples_per_prompt=2, prefix="reward")

        assert "reward/foo_bar/variance_per_prompt" in metrics

    def test_multiple_groups_independent(self):
        """Each group's metrics are independent."""
        # github: all pass with variance, booking: none pass
        rewards = [1.0, 0.5, 0.0, 0.0]
        uids = ["t1", "t1", "t2", "t2"]
        groups = ["github", "github", "booking", "booking"]

        metrics = compute_per_group_metrics(rewards, uids, groups, n_samples_per_prompt=2, prefix="reward")

        assert metrics["reward/github/pass_at_2"] == 1.0
        assert metrics["reward/github/variance_per_prompt"] == 0.0625  # [1,0.5] var=0.0625
        assert metrics["reward/booking/pass_at_2"] == 0.0
        assert metrics["reward/booking/variance_per_prompt"] == 0.0  # [0,0] var=0

    def test_none_group_becomes_unknown(self):
        """None group values become 'unknown'."""
        rewards = [1.0, 0.0]
        uids = ["a", "a"]
        groups = [None, None]

        metrics = compute_per_group_metrics(rewards, uids, groups, n_samples_per_prompt=2, prefix="reward")

        assert "reward/unknown/variance_per_prompt" in metrics

    def test_empty_input(self):
        """Empty input returns empty dict."""
        metrics = compute_per_group_metrics([], [], [], n_samples_per_prompt=1, prefix="reward")
        assert metrics == {}

    def test_single_group_all_data(self):
        """All data in single group."""
        rewards = [1.0, 0.0, 0.5, 0.5]
        uids = ["a", "a", "b", "b"]
        groups = ["env1", "env1", "env1", "env1"]

        metrics = compute_per_group_metrics(rewards, uids, groups, n_samples_per_prompt=2, prefix="reward")

        assert len([k for k in metrics.keys() if k.startswith("reward/env1/")]) == 3
        # uid "a" [1,0] var=0.25, uid "b" [0.5,0.5] var=0 -> mean=0.125
        assert abs(metrics["reward/env1/variance_per_prompt"] - 0.125) < 1e-10


class TestIntegrationWithSkyRLPatterns:
    """Test that the functions produce output matching SkyRL's expected patterns."""

    def test_training_metrics_pattern(self):
        """Verify output matches reward/{env}/metric pattern for training."""
        rewards = [1.0, 0.0]
        uids = ["task1", "task1"]
        groups = ["github", "github"]

        metrics = compute_per_group_metrics(rewards, uids, groups, n_samples_per_prompt=2, prefix="reward")

        # These are the exact keys SkyRL trainer expects
        assert "reward/github/variance_per_prompt" in metrics
        assert "reward/github/pass_at_2" in metrics
        assert "reward/github/mean_positive_reward" in metrics

    def test_eval_metrics_pattern(self):
        """Verify output matches eval/{env}/metric pattern for evaluation."""
        rewards = [1.0, 0.0]
        uids = ["task1", "task1"]
        groups = ["github", "github"]

        metrics = compute_per_group_metrics(rewards, uids, groups, n_samples_per_prompt=3, prefix="eval")

        # These are the exact keys SkyRL evaluate.py expects
        assert "eval/github/variance_per_prompt" in metrics
        assert "eval/github/pass_at_3" in metrics
        assert "eval/github/mean_positive_reward" in metrics
