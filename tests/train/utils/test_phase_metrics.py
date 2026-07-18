"""
uv run --isolated --extra dev --extra skyrl-train pytest tests/train/utils/test_phase_metrics.py
"""

from unittest.mock import MagicMock, patch

from skyrl.train.utils.phase_metrics import PHASES, ScalarGauges, TrainingPhaseGauge


def _make_gauge():
    """Build a TrainingPhaseGauge over a mock ray Gauge; return (obj, {phase: latest value})."""
    values = {}
    mock_gauge = MagicMock()
    mock_gauge.set.side_effect = lambda v, tags: values.__setitem__(tags["phase"], v)
    with patch("ray.util.metrics.Gauge", return_value=mock_gauge):
        obj = TrainingPhaseGauge()
    return obj, values


def _active_phase(values):
    # every phase series exists (seeded at construction) and exactly one is 1.0
    assert set(values) == set(PHASES)
    active = [p for p, v in values.items() if v == 1.0]
    assert len(active) == 1, f"expected exactly one active phase, got {active}"
    return active[0]


def test_construction_defaults_to_generating():
    _, values = _make_gauge()
    assert _active_phase(values) == "generating"


def test_set_phase_marks_exactly_one_active():
    obj, values = _make_gauge()
    obj.set_phase("run_training")
    assert _active_phase(values) == "run_training"
    obj.set_phase("eval")
    assert _active_phase(values) == "eval"


def test_phase_context_manager_restores_prior_phase():
    obj, values = _make_gauge()
    obj.set_phase("run_training")
    with obj.phase("save_checkpoints"):
        assert _active_phase(values) == "save_checkpoints"
    # restored to whatever was active before the block, not hard-coded to a default
    assert _active_phase(values) == "run_training"


def test_unknown_phase_is_rejected():
    obj, values = _make_gauge()
    obj.set_phase("run_training")
    obj.set_phase("bogus_phase")
    assert _active_phase(values) == "run_training"


def test_disabled_when_ray_metrics_unavailable():
    # Gauge() raising must not propagate
    with patch("ray.util.metrics.Gauge", side_effect=RuntimeError("ray not initialized")):
        obj = TrainingPhaseGauge()
    obj.set_phase("run_training")
    with obj.phase("eval"):
        pass


def test_scalar_gauges_create_once_and_update():
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


def test_scalar_gauges_disabled_when_ray_metrics_unavailable():
    with patch("ray.util.metrics.Gauge", side_effect=RuntimeError("ray not initialized")):
        g = ScalarGauges()
        g.set("skyrl_gen_buffer_qsize", 1)  # must not raise
