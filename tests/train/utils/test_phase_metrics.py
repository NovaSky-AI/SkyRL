"""
uv run --isolated --extra dev --extra skyrl-train pytest tests/train/utils/test_phase_metrics.py
"""

from unittest.mock import MagicMock, patch

from skyrl.train.utils.phase_metrics import PHASES, TrainingPhaseGauge


def _make_gauge():
    """Build a TrainingPhaseGauge backed by a mock ray Gauge; return (obj, mock_gauge)."""
    mock_gauge = MagicMock()
    with patch("ray.util.metrics.Gauge", return_value=mock_gauge):
        obj = TrainingPhaseGauge()
    return obj, mock_gauge


def _active_phase(mock_gauge):
    """Return the single phase set to 1.0 in the most recent full emit (len(PHASES) set calls)."""
    last = mock_gauge.set.call_args_list[-len(PHASES) :]
    active = [c.kwargs["tags"]["phase"] for c in last if c.args[0] == 1.0]
    assert len(active) == 1, f"expected exactly one active phase, got {active}"
    # every phase must be written each emit (so stale phases are cleared to 0.0)
    written = {c.kwargs["tags"]["phase"] for c in last}
    assert written == set(PHASES)
    return active[0]


def test_construction_defaults_to_generating():
    _, g = _make_gauge()
    assert _active_phase(g) == "generating"


def test_set_phase_marks_exactly_one_active():
    obj, g = _make_gauge()
    obj.set_phase("training")
    assert _active_phase(g) == "training"
    obj.set_phase("eval")
    assert _active_phase(g) == "eval"


def test_phase_context_manager_restores_prior_phase():
    obj, g = _make_gauge()
    obj.set_phase("training")
    with obj.phase("checkpoint"):
        assert _active_phase(g) == "checkpoint"
    # restored to whatever was active before the block, not hard-coded to a default
    assert _active_phase(g) == "training"


def test_disabled_when_ray_metrics_unavailable():
    # Gauge construction raising (e.g. Ray not initialized) must degrade to a silent no-op,
    # never propagate, so training is never broken by observability.
    with patch("ray.util.metrics.Gauge", side_effect=RuntimeError("ray not initialized")):
        obj = TrainingPhaseGauge()
    # all calls are safe no-ops
    obj.set_phase("training")
    with obj.phase("eval"):
        pass
