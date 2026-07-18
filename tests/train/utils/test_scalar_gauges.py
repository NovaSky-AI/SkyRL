"""
uv run --isolated --extra dev --extra skyrl-train pytest tests/train/utils/test_scalar_gauges.py
"""

from unittest.mock import MagicMock, patch

from skyrl.train.utils.scalar_gauges import ScalarGauges


def test_create_once_and_update():
    created = {}

    def fake_gauge(name, description=None):
        created[name] = MagicMock(description=description)
        return created[name]

    with patch("ray.util.metrics.Gauge", side_effect=fake_gauge):
        g = ScalarGauges()
        g.set("skyrl_gen_buffer_qsize", 3, "queued groups")
        g.set("skyrl_gen_buffer_qsize", 5)
        g.set("skyrl_mini_batch_size", 8)

    # A gauge is created once per name with its first description, and values are coerced to float.
    assert set(created) == {"skyrl_gen_buffer_qsize", "skyrl_mini_batch_size"}
    assert created["skyrl_gen_buffer_qsize"].description == "queued groups"
    created["skyrl_gen_buffer_qsize"].set.assert_called_with(5.0)
    created["skyrl_mini_batch_size"].set.assert_called_with(8.0)


def test_disabled_when_ray_metrics_unavailable():
    with patch("ray.util.metrics.Gauge", side_effect=RuntimeError("ray not initialized")):
        g = ScalarGauges()
        g.set("skyrl_gen_buffer_qsize", 1)  # must not raise
