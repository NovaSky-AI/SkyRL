"""
uv run --isolated --extra dev --extra skyrl-train pytest tests/train/utils/test_phase_metrics.py
"""

from unittest.mock import MagicMock, patch

from skyrl.train.utils.phase_metrics import PHASES, TrainingPhaseGauge


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


def test_disabled_by_flag_constructs_no_gauge():
    with patch("ray.util.metrics.Gauge") as gauge_cls:
        obj = TrainingPhaseGauge(enabled=False)
    gauge_cls.assert_not_called()
    obj.set_phase("run_training")
    with obj.phase("eval"):
        pass
