from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyrl.backends.skyrl_train.utils import profiler


def test_optimizer_and_profile_processing_have_disjoint_durations(monkeypatch):
    monkeypatch.setattr(profiler, "perf_counter", Mock(side_effect=[10, 12, 12, 72]))
    metrics = {}
    with profiler.measure_phase_seconds(metrics, "optimizer"):
        pass
    with profiler.measure_phase_seconds(metrics, "profile_processing"):
        pass
    assert metrics == {"optimizer": 2, "profile_processing": 60}


def test_failed_phase_does_not_report_completed_duration(monkeypatch):
    clock = Mock(return_value=10)
    monkeypatch.setattr(profiler, "perf_counter", clock)
    metrics = {}
    with pytest.raises(RuntimeError, match="optimizer failed"):
        with profiler.measure_phase_seconds(metrics, "optimizer"):
            raise RuntimeError("optimizer failed")
    assert metrics == {}
    assert clock.call_count == 1


class FakeTimer:
    def __init__(self):
        self.seconds = 100

    def reset(self):
        self.seconds = 0

    def elapsed(self, reset, barrier):
        assert not reset and not barrier
        return self.seconds


class FakeTimers:
    def __init__(self):
        self.phases = {}
        self.levels = {}

    def __call__(self, name, log_level=None):
        if name not in self.phases:
            self.phases[name] = FakeTimer()
        if log_level is not None:
            self.levels[name] = log_level
        return self.phases[name]


def test_schedule_accumulates_two_microbatches_without_reference_subtraction():
    config, timers = SimpleNamespace(timers=None), FakeTimers()
    with profiler.measure_megatron_schedule(config, timers) as report:
        for forward, backward in ((2, 3), (4, 7)):
            config.timers("forward-compute").seconds += forward
            config.timers("backward-compute").seconds += backward
            config.timers("forward-backward").seconds += forward + backward + 1
    assert report == {
        "forward-compute": 6,
        "backward-compute": 10,
        "forward-backward": 18,
    }
    assert config.timers is None
    assert timers.levels == {
        "forward-compute": 2,
        "backward-compute": 2,
        "forward-backward": 1,
    }


def test_failed_schedule_restores_configuration_without_partial_receipt():
    config = SimpleNamespace(timers=None)
    with pytest.raises(RuntimeError, match="backward failed"):
        with profiler.measure_megatron_schedule(config, FakeTimers()) as report:
            raise RuntimeError("backward failed")
    assert config.timers is None and report == {}


def test_existing_schedule_timers_are_not_overwritten():
    existing = object()
    config = SimpleNamespace(timers=existing)
    with pytest.raises(ValueError, match="existing Megatron timers"):
        with profiler.measure_megatron_schedule(config, FakeTimers()):
            pytest.fail("must reject before executing the schedule")
    assert config.timers is existing
