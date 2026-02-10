"""
Tests for worker utility functions.

uv run --isolated --extra dev pytest tests/cpu/workers/test_worker_utils.py
"""

import pytest
from unittest.mock import MagicMock
from skyrl_train.workers.worker_utils import reduce_metrics, all_reduce_metrics


class TestReduceMetrics:
    def test_reduce_metrics_max_suffix(self):
        """Keys ending in _max should use max reduction."""
        metrics = {"is_ratio_max": [1.0, 5.0, 3.0]}
        result = reduce_metrics(metrics)
        assert result["is_ratio_max"] == 5.0

    def test_reduce_metrics_min_suffix(self):
        """Keys ending in _min should use min reduction."""
        metrics = {"is_ratio_min": [1.0, 5.0, 3.0]}
        result = reduce_metrics(metrics)
        assert result["is_ratio_min"] == 1.0

    def test_reduce_metrics_mean_default(self):
        """Keys without _max/_min/_loss suffix should use mean reduction."""
        metrics = {"entropy": [1.0, 2.0, 3.0]}
        result = reduce_metrics(metrics)
        assert result["entropy"] == 2.0  # mean of [1, 2, 3]

    def test_reduce_metrics_loss_sum(self):
        """Keys ending in _loss should use sum reduction."""
        metrics = {"policy_loss": [1.0, 2.0, 3.0]}
        result = reduce_metrics(metrics)
        assert result["policy_loss"] == 6.0  # sum of [1, 2, 3]

    def test_reduce_metrics_mixed(self):
        """Test mixed metric types are reduced correctly."""
        metrics = {
            "is_ratio_max": [1.0, 10.0],
            "is_ratio_min": [0.5, 2.0],
            "policy_loss": [1.0, 3.0],
            "entropy": [1.0, 3.0],
        }
        result = reduce_metrics(metrics)
        assert result["is_ratio_max"] == 10.0
        assert result["is_ratio_min"] == 0.5
        assert result["policy_loss"] == 4.0  # sum
        assert result["entropy"] == 2.0  # mean

    def test_reduce_metrics_single_value(self):
        """Test reduction with single value lists."""
        metrics = {
            "is_ratio_max": [5.0],
            "is_ratio_min": [0.5],
            "policy_loss": [1.5],
        }
        result = reduce_metrics(metrics)
        assert result["is_ratio_max"] == 5.0
        assert result["is_ratio_min"] == 0.5
        assert result["policy_loss"] == 1.5

    def test_reduce_metrics_empty_raises(self):
        """Test that empty list raises assertion error."""
        metrics = {"policy_loss": []}
        with pytest.raises(AssertionError, match="No metrics for key"):
            reduce_metrics(metrics)


class TestAllReduceMetrics:
    def test_all_reduce_metrics_separates_by_suffix(self):
        """Verify metrics are correctly separated by suffix and reduced with correct ops."""
        strategy = MagicMock()

        # Mock all_reduce to return the input dict unchanged but track calls
        def mock_all_reduce(d, op, group=None):
            return {k: v for k, v in d.items()}

        strategy.all_reduce.side_effect = mock_all_reduce

        metrics = {
            "is_ratio_max": 10.0,
            "is_ratio_min": 0.1,
            "policy_loss": 1.5,
            "entropy": 0.5,
        }

        _ = all_reduce_metrics(metrics, strategy)

        # Verify all_reduce was called 4 times (mean, min, max, sum)
        assert strategy.all_reduce.call_count == 4

        # Check that the correct ops were used
        calls = strategy.all_reduce.call_args_list

        # Find which call used which op
        ops_and_keys = []
        for call in calls:
            args, kwargs = call
            data_dict = args[0]
            op = kwargs.get("op") if kwargs else args[1]
            ops_and_keys.append((op, set(data_dict.keys())))

        # Verify mean metrics (entropy only - no suffix)
        mean_call = [c for c in ops_and_keys if c[0] == "mean"][0]
        assert mean_call[1] == {"entropy"}

        # Verify sum metrics (_loss suffix)
        sum_call = [c for c in ops_and_keys if c[0] == "sum"][0]
        assert sum_call[1] == {"policy_loss"}

        # Verify min metrics
        min_call = [c for c in ops_and_keys if c[0] == "min"][0]
        assert min_call[1] == {"is_ratio_min"}

        # Verify max metrics
        max_call = [c for c in ops_and_keys if c[0] == "max"][0]
        assert max_call[1] == {"is_ratio_max"}

    def test_all_reduce_metrics_returns_merged_results(self):
        """Verify results from all reductions are merged correctly."""
        strategy = MagicMock()

        # Mock all_reduce to modify values based on op
        def mock_all_reduce(d, op, group=None):
            if op == "mean":
                return {k: v * 2 for k, v in d.items()}  # Double for mean
            elif op == "min":
                return {k: v / 2 for k, v in d.items()}  # Halve for min
            elif op == "max":
                return {k: v * 3 for k, v in d.items()}  # Triple for max
            elif op == "sum":
                return {k: v * 4 for k, v in d.items()}  # Quadruple for sum
            return d

        strategy.all_reduce.side_effect = mock_all_reduce

        metrics = {
            "is_ratio_max": 10.0,
            "is_ratio_min": 0.1,
            "policy_loss": 1.5,
            "entropy": 0.5,
        }

        result = all_reduce_metrics(metrics, strategy)

        # Check all keys are present
        assert "is_ratio_max" in result
        assert "is_ratio_min" in result
        assert "policy_loss" in result
        assert "entropy" in result

        # Check values were transformed correctly
        assert result["is_ratio_max"] == 30.0  # 10.0 * 3 (max op)
        assert result["is_ratio_min"] == 0.05  # 0.1 / 2 (min op)
        assert result["policy_loss"] == 6.0  # 1.5 * 4 (sum op)
        assert result["entropy"] == 1.0  # 0.5 * 2 (mean op)

    def test_all_reduce_metrics_only_max(self):
        """Test with only _max metrics."""
        strategy = MagicMock()
        strategy.all_reduce.side_effect = lambda d, op, group=None: d

        metrics = {"loss_max": 5.0, "ratio_max": 10.0}

        result = all_reduce_metrics(metrics, strategy)

        assert result == {"loss_max": 5.0, "ratio_max": 10.0}

    def test_all_reduce_metrics_only_min(self):
        """Test with only _min metrics."""
        strategy = MagicMock()
        strategy.all_reduce.side_effect = lambda d, op, group=None: d

        metrics = {"loss_min": 0.1, "ratio_min": 0.01}

        result = all_reduce_metrics(metrics, strategy)

        assert result == {"loss_min": 0.1, "ratio_min": 0.01}

    def test_all_reduce_metrics_only_mean(self):
        """Test with only mean metrics (no _max/_min/_loss suffix)."""
        strategy = MagicMock()
        strategy.all_reduce.side_effect = lambda d, op, group=None: d

        metrics = {"entropy": 0.5, "kl_div": 1.5}

        result = all_reduce_metrics(metrics, strategy)

        assert result == {"entropy": 0.5, "kl_div": 1.5}

    def test_all_reduce_metrics_only_sum(self):
        """Test with only _loss metrics (sum reduction)."""
        strategy = MagicMock()
        strategy.all_reduce.side_effect = lambda d, op, group=None: d

        metrics = {"policy_loss": 1.5, "value_loss": 0.5}

        result = all_reduce_metrics(metrics, strategy)

        assert result == {"policy_loss": 1.5, "value_loss": 0.5}
