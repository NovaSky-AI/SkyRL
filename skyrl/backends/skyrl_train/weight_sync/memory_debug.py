"""Lightweight memory logging helpers for weight-sync debugging."""

from __future__ import annotations

import ctypes
import gc
import json
import os
import resource
import socket
import threading
import time
from pathlib import Path
from typing import Any

_KIB = 1024


def memory_logging_enabled() -> bool:
    return os.environ.get("SKYRL_DELTA_MEMORY_LOG", "").lower() in {"1", "true", "yes", "on"}


def _parse_kib_file(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, raw_value = line.split(":", 1)
                parts = raw_value.strip().split()
                if not parts:
                    continue
                try:
                    value = int(parts[0])
                except ValueError:
                    continue
                values[key] = value * _KIB if len(parts) > 1 and parts[1] == "kB" else value
    except OSError:
        return {}
    return values


def _count_fds() -> int | None:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return None


def process_memory_snapshot(label: str, **extra: Any) -> dict[str, Any]:
    status = _parse_kib_file(Path("/proc/self/status"))
    smaps = _parse_kib_file(Path("/proc/self/smaps_rollup"))
    meminfo = _parse_kib_file(Path("/proc/meminfo"))
    usage = resource.getrusage(resource.RUSAGE_SELF)

    snapshot: dict[str, Any] = {
        "event": "skyrl_memory",
        "label": label,
        "time": time.time(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "threading_active_count": threading.active_count(),
        "fd_count": _count_fds(),
        "gc_count": list(gc.get_count()),
        "ru_maxrss_bytes": int(usage.ru_maxrss) * _KIB,
    }

    for key in ("VmRSS", "VmHWM", "VmSize", "VmData", "VmSwap", "Threads"):
        if key in status:
            snapshot[f"status_{key}_bytes" if key != "Threads" else "status_threads"] = status[key]
    for key in ("Rss", "Pss", "Private_Clean", "Private_Dirty", "Shared_Clean", "Shared_Dirty", "Anonymous"):
        if key in smaps:
            snapshot[f"smaps_{key}_bytes"] = smaps[key]
    for key in ("MemTotal", "MemAvailable", "MemFree", "Cached", "SwapTotal", "SwapFree"):
        if key in meminfo:
            snapshot[f"node_{key}_bytes"] = meminfo[key]

    snapshot.update(extra)
    return snapshot


def log_memory(logger: Any, label: str, **extra: Any) -> None:
    if not memory_logging_enabled():
        return
    message = "skyrl_memory " + json.dumps(process_memory_snapshot(label, **extra), sort_keys=True)
    try:
        logger.info(message)
    except TypeError:
        logger.info("%s", message)


def trim_process_memory() -> bool:
    """Best-effort release of freed Python/native heap pages back to the OS."""
    gc.collect()
    if os.name != "posix":
        return False
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is None:
            return False
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        return bool(malloc_trim(0))
    except Exception:
        return False
