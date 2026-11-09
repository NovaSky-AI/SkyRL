"""uv run --isolated --extra dev pytest tests/utils/test_stage_trace.py"""

import json
import os
import time

import pytest

from skyrl.utils import stage_trace


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """Each test starts untraced, with no totals."""
    monkeypatch.setattr(stage_trace, "_path", None)
    monkeypatch.setattr(stage_trace, "_pid", None)
    monkeypatch.setattr(stage_trace, "_events", [])
    monkeypatch.setattr(stage_trace, "_totals", {})
    monkeypatch.setattr(stage_trace, "SKYRL_STAGE_TRACE_DIR", "")


def test_totals_count_spans_untraced_and_reset_when_taken():
    stage_trace.record("mb.fetch", 0.0, 0.5)
    stage_trace.record("mb.fetch", 1.0, 3.0)
    with stage_trace.span("mb.stage"):
        pass
    stage_trace.record("worker.fb_schedule", 0.0, 9.0)

    taken = stage_trace.take_totals("mb.")

    assert set(taken) == {"mb.fetch", "mb.stage"}
    count, seconds, slowest = taken["mb.fetch"]
    assert (count, seconds, slowest) == (2, pytest.approx(2.5), pytest.approx(2.0))
    assert stage_trace.take_totals("mb.") == {}
    assert set(stage_trace.take_totals()) == {"worker.fb_schedule"}
    assert not stage_trace.enabled()
    assert stage_trace._events == []


def test_a_span_counts_when_its_block_raises():
    with pytest.raises(ValueError):
        with stage_trace.span("mb.collate"):
            raise ValueError
    assert stage_trace.take_totals()["mb.collate"][0] == 1


def test_traced_events_merge_into_one_chrome_trace(tmp_path):
    trace_dir = tmp_path / "traces"
    assert stage_trace.configure(str(trace_dir), component="policy_worker")
    t0 = time.perf_counter()
    with stage_trace.span("mb.fetch", payloads=3) as attrs:
        attrs["ready"] = 3
    stage_trace.record("mb.collate", t0, t0 + 0.25, rows=2)
    stage_trace.instant("trainer.group_buffered", buffered=1)
    stage_trace.flush()
    # A second process's file, with a line its process left half-written.
    other = trace_dir / "other-host-1.jsonl"
    other.write_text(json.dumps({"ph": "X", "name": "mb.stage", "ts": 1.0, "dur": 2.0, "pid": 1}) + '\n{"ph": "X"')

    out = tmp_path / "trace.json"
    assert stage_trace.merge(str(trace_dir), str(out)) == 5

    events = json.loads(out.read_text())["traceEvents"]
    by_name = {e["name"]: e for e in events}
    assert by_name["process_name"]["args"]["name"].startswith("policy_worker ")
    assert by_name["mb.fetch"]["ph"] == "X" and by_name["mb.fetch"]["args"] == {"payloads": 3, "ready": 3}
    assert by_name["mb.collate"]["dur"] == pytest.approx(0.25e6)
    assert by_name["trainer.group_buffered"]["ph"] == "i"
    # Files keep apart on the timeline, whatever pids they carried.
    assert by_name["mb.stage"]["pid"] != by_name["mb.fetch"]["pid"]


def test_the_trace_directory_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_trace, "SKYRL_STAGE_TRACE_DIR", str(tmp_path))
    stage_trace.record("mb.fetch", 0.0, 1.0)
    stage_trace.flush()
    assert os.listdir(tmp_path) == [f"{stage_trace._host}-{os.getpid()}.jsonl"]


def test_timer_records_its_phase():
    from skyrl.train.utils.utils import Timer

    timings = {}
    with Timer("check_payloads", timings):
        pass
    assert stage_trace.take_totals("t.")["t.check_payloads"][0] == 1
    assert "check_payloads" in timings
