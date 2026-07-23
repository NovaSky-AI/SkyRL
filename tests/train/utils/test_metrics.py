"""
uv run --isolated --extra dev --extra skyrl-train pytest tests/train/utils/test_metrics.py
"""

from unittest.mock import MagicMock, patch

from skyrl.train.utils.metrics import PHASES, TrainingPhaseGauge


def _make_gauge():
    """Build a TrainingPhaseGauge over a mock ray Gauge; return (obj, {phase: latest value})."""
    values = {}
    mock_gauge = MagicMock()
    mock_gauge.set.side_effect = lambda v, tags: values.__setitem__(tags["phase"], v)
    with patch("ray.util.metrics.Gauge", return_value=mock_gauge):
        obj = TrainingPhaseGauge()
    return obj, values


def _active(values):
    return {p for p, v in values.items() if v == 1.0}


def test_construction_seeds_every_phase_to_zero():
    _, values = _make_gauge()
    assert set(values) == set(PHASES)
    assert _active(values) == set()


def test_phase_marks_one_active_then_clears():
    obj, values = _make_gauge()
    with obj.phase("run_training"):
        assert _active(values) == {"run_training"}
    assert _active(values) == set()


def test_unknown_phase_publishes_nothing():
    obj, values = _make_gauge()
    with obj.phase("bogus_phase"):
        assert _active(values) == set()


def test_timed_phase_records_timing_and_marks_phase():
    obj, values = _make_gauge()
    timings = {}
    with obj.timed_phase("run_training", timings):
        assert _active(values) == {"run_training"}
    assert _active(values) == set()
    assert "run_training" in timings
