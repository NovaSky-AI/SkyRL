"""Timed spans of the rollout-to-gradient data path: per-process totals, and an optional timeline.

Every ``record``/``span`` adds its duration to this process's running totals per span name
(count, seconds, slowest), which ``take_totals`` reads and resets: a worker turns the totals of
one call into that call's metrics, so they cost a dict update whether or not anything is traced.

With ``SKYRL_STAGE_TRACE_DIR`` set (or ``configure`` given a directory), each process also writes
its spans and instants as Chrome trace events, one JSON line each, to ``<dir>/<host>-<pid>.jsonl``
from a background thread. Timestamps are wall-clock microseconds (``time.time``), so events from
different nodes line up to NTP accuracy. ``python -m skyrl.utils.stage_trace <dir> <out.json>``
merges a run's files into one trace for https://ui.perfetto.dev.

``Timer`` records each phase it times as ``t.<name>``, so the trainer's phases frame the timeline.

Tracing never fails the traced work: a process whose trace can't be written or serialized logs a
warning and stops tracing; its totals keep counting.
"""

import atexit
import contextlib
import glob
import os
import socket
import sys
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import orjson

from skyrl.env_vars import SKYRL_STAGE_TRACE_DIR
from skyrl.utils.log import logger

_lock = threading.Lock()
# Span name -> [count, total seconds, slowest seconds], since the last ``take_totals``.
_totals: Dict[str, List[float]] = {}
_events: List[bytes] = []
_path: Optional[str] = None
_pid: Optional[int] = None
_component = ""
_host = socket.gethostname()
_flusher: Optional[threading.Thread] = None
_failed = False  # This process stopped tracing after an error (``_stop``).


def configure(trace_dir: Optional[str] = None, component: Optional[str] = None) -> bool:
    """Write this process's events under ``trace_dir`` (default ``$SKYRL_STAGE_TRACE_DIR``); whether it does.

    ``component`` labels the process on the timeline (e.g. ``policy_worker``).
    """
    global _path, _pid, _component, _flusher
    if component:
        _component = component
    trace_dir = trace_dir or SKYRL_STAGE_TRACE_DIR
    if not trace_dir or _failed:
        return False
    if _path is not None and _pid == os.getpid():
        return True
    _pid = os.getpid()
    try:
        os.makedirs(trace_dir, exist_ok=True)
    except OSError as error:
        _stop(error)
        return False
    _path = os.path.join(trace_dir, f"{_host}-{_pid}.jsonl")
    _emit({"ph": "M", "name": "process_name", "args": {"name": f"{_component or 'skyrl'} {_host}:{_pid}"}})
    if _flusher is None or not _flusher.is_alive():
        _flusher = threading.Thread(target=_flush_loop, name="stage-trace-flush", daemon=True)
        _flusher.start()
        atexit.register(flush)
    return not _failed


def enabled() -> bool:
    """Whether this process writes trace events (configured here, or in a parent before a fork)."""
    if _path is not None and _pid == os.getpid():
        return True
    return bool(SKYRL_STAGE_TRACE_DIR) and configure()


def _emit(event: Dict[str, Any]) -> None:
    event["pid"] = _pid
    event["tid"] = threading.get_ident() & 0xFFFFFF
    try:
        line = orjson.dumps(event, option=orjson.OPT_SERIALIZE_NUMPY, default=str)
    except TypeError as error:  # orjson.JSONEncodeError, e.g. an int past 64 bits
        _stop(error)
        return
    with _lock:
        _events.append(line)


def _stop(error: Exception) -> None:
    """Stop this process's tracing after ``error``, dropping its unwritten events."""
    global _path, _events, _failed
    with _lock:
        if _failed:
            return
        _failed = True
        _path = None
        _events = []
    logger.warning("stage_trace: stopped tracing this process after %r", error)


def record(name: str, t0: float, t1: float, **attrs: Any) -> None:
    """A span from ``time.perf_counter`` readings ``t0`` to ``t1`` taken in this process."""
    seconds = t1 - t0
    with _lock:
        totals = _totals.setdefault(name, [0, 0.0, 0.0])
        totals[0] += 1
        totals[1] += seconds
        totals[2] = max(totals[2], seconds)
    if enabled():
        end_us = time.time() * 1e6 - (time.perf_counter() - t1) * 1e6
        _emit({"ph": "X", "name": name, "ts": end_us - seconds * 1e6, "dur": seconds * 1e6, "args": attrs})


@contextlib.contextmanager
def span(name: str, **attrs: Any) -> Iterator[Dict[str, Any]]:
    """Time the block as ``name``. Entries set on the yielded dict become the span's attributes."""
    t0 = time.perf_counter()
    try:
        yield attrs
    finally:
        record(name, t0, time.perf_counter(), **attrs)


def instant(name: str, **attrs: Any) -> None:
    """A point event on the timeline (not counted in the totals)."""
    if enabled():
        _emit({"ph": "i", "s": "t", "name": name, "ts": time.time() * 1e6, "args": attrs})


def take_totals(prefix: str = "") -> Dict[str, Tuple[int, float, float]]:
    """``(count, seconds, slowest seconds)`` per span name under ``prefix`` since the last take, then reset them."""
    with _lock:
        taken = {name: tuple(_totals.pop(name)) for name in [n for n in _totals if n.startswith(prefix)]}
    return taken


def flush() -> None:
    global _events
    with _lock:
        path, events, _events = _path, _events, []
    if path is None or not events:
        return
    try:
        with open(path, "ab") as f:
            f.write(b"\n".join(events) + b"\n")
    except OSError as error:
        _stop(error)


def _flush_loop() -> None:
    while True:
        time.sleep(1.0)
        try:
            flush()
        except Exception as error:  # noqa: BLE001 - tracing must never take the process down
            _stop(error)


def merge(trace_dir: str, out_path: str) -> int:
    """Merge ``trace_dir``'s event files into one Chrome trace at ``out_path``; returns the event count.

    Each file gets its own process id, since pids repeat across nodes.
    """
    events = []
    for index, path in enumerate(sorted(glob.glob(os.path.join(trace_dir, "*.jsonl")))):
        with open(path, "rb") as f:
            for line in f:
                try:
                    event = orjson.loads(line)
                except orjson.JSONDecodeError:  # a line cut short by a killed process
                    continue
                event["pid"] = index
                events.append(event)
    with open(out_path, "wb") as f:
        f.write(orjson.dumps({"traceEvents": events, "displayTimeUnit": "ms"}))
    return len(events)


if __name__ == "__main__":
    print(f"merged {merge(sys.argv[1], sys.argv[2])} events into {sys.argv[2]}")
