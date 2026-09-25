"""What `/healthz` and `/metrics` report, assembled from the components.

Each component reports its own counters; this puts them in the one document
operators and the viewer's header read. It is the only consumer of every
component's `stats()` at once, which is why it exists as a class rather than
as a method somewhere that would have to know about all of them anyway.

A standalone viewer has a health document too, shaped the same so the UI's
header needs no special case.
"""

from __future__ import annotations

from typing import Any


class HealthProvider:
    def __init__(
        self,
        *,
        data_plane: Any,
        proxy: Any,
        registry: Any,
        commits: Any,
        active: Any,
        committed: Any,
        mode: str,
        record: str,
        clock_epoch: str,
    ) -> None:
        self._data_plane = data_plane
        # The one proxy this process was built with. Health reports what is
        # running rather than a slot per mode, half of them always zero.
        self._proxy = proxy
        self._registry = registry
        self._commits = commits
        self._active = active
        self._committed = committed
        self._mode = mode
        self._record = record
        self._clock_epoch = clock_epoch

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self._mode,
            "record": self._record,
            "requests_served": self._data_plane.stats()["requests_served"],
            "clock_epoch": self._clock_epoch,
            # How much capture is holding, how far behind the disk is, and
            # what it has refused. The three numbers an outage shows up in.
            "registry": self._registry.stats(),
            "commits": self._commits.stats(),
            "store": {**self._active.stats(), **self._committed.stats()},
            # Whatever the selected proxy counts. In token capture that
            # includes the renderer's phase timings, the loop lag, the trace
            # cache and the wait before a response closes; in text capture,
            # the forward path's counters.
            "capture": self._proxy.stats(),
        }


class ViewerHealth:
    """Shaped like the service's, so the UI header needs no special case."""

    def __init__(self, reader: Any) -> None:
        self._reader = reader

    def stats(self) -> dict[str, Any]:
        info = self._reader.source_info()
        return {
            "mode": "viewer",
            "requests_served": 0,
            "capture": {"capture_errors": 0, "capture_enabled": False},
            "registry": {"hot_trajectories": 0, "recovered": 0, "evicted": 0},
            "commits": {
                "pending_commits": 0,
                "pending_high_water": 0,
                "capacity": 0,
                "oldest_pending_age_s": 0.0,
                "commits": 0,
                "refused": 0,
                "failures": 0,
                "unwritten_gaps": 0,
                "last_error": None,
            },
            **info,
        }
