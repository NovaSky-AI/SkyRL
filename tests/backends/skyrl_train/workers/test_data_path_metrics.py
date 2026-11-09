"""uv run --isolated --extra dev --extra skyrl-train pytest tests/backends/skyrl_train/workers/test_data_path_metrics.py"""

from types import SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase
from skyrl.utils import stage_trace


class MaxOverRanks:
    """A strategy whose other ranks report ``others``: ``all_reduce(op="max")`` takes the larger."""

    def __init__(self, others):
        self.others = others
        self.calls = []

    def all_reduce(self, data, op="mean", group=None):
        self.calls.append((op, group))
        return {k: max(v, self.others.get(k, 0.0)) for k, v in data.items()}


@pytest.fixture(autouse=True)
def fresh_totals(monkeypatch):
    monkeypatch.setattr(stage_trace, "_totals", {})


def test_a_calls_microbatch_waits_become_world_max_metrics():
    stage_trace.record("mb.fetch", 0.0, 0.001)
    stage_trace.record("mb.fetch", 0.0, 1.5)
    stage_trace.record("mb.stage_exposed", 0.0, 0.8)
    stage_trace.record("mb.stage", 0.0, 0.4)
    stage_trace.record("mb.stage", 0.0, 0.6)
    stage_trace.record("worker.fb_schedule", 0.0, 100.0)
    strategy = MaxOverRanks({"data_path/mb_stage_exposed_total_s_max": 2.0})

    metrics = PolicyWorkerBase._data_path_metrics(SimpleNamespace(strategy=strategy))

    assert metrics == {
        "data_path/mb_fetch_total_s_max": pytest.approx(1.501),
        "data_path/mb_fetch_slowest_s_max": pytest.approx(1.5),
        "data_path/mb_stage_exposed_total_s_max": pytest.approx(2.0),
        "data_path/mb_stage_exposed_count_max": 1.0,
        "data_path/mb_stage_slowest_s_max": pytest.approx(0.6),
    }
    # Over the whole world: a stalled rank on any pipeline stage holds up the step.
    assert strategy.calls == [("max", None)]
    # Taken: the next call starts from zero; spans outside the microbatch path are left alone.
    assert stage_trace.take_totals("mb.") == {}
    assert "worker.fb_schedule" in stage_trace.take_totals()


def test_a_call_with_no_microbatch_spans_reports_zeros():
    metrics = PolicyWorkerBase._data_path_metrics(SimpleNamespace(strategy=MaxOverRanks({})))
    assert set(metrics.values()) == {0.0}
