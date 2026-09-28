"""What the clocks on a set of exchanges say about each other.

`gap_ms` and `overlapping` are not stored on an exchange, because neither is a
property of one exchange: both are answers about a trajectory's other calls.
Deriving them on read is what keeps them correct when a trajectory branches
after the fact, and it is why they live here rather than in the exchange row.
"""

from __future__ import annotations

from typing import Any


def gap_ms(row: dict[str, Any], siblings: list[dict[str, Any]]) -> float | None:
    """The wait before this call, measured from the call it continued from.

    From the *parent* exchange -- the one whose model output this call's
    context continued from -- and not from whichever call happened to arrive
    before it. Arrival order is completion order, and the moment a trajectory
    branches the previous arrival is a sibling.
    """
    parent_id = row.get("parent_output_node_id")
    if parent_id is None or row.get("request_start_at") is None:
        return None
    for candidate in siblings:
        if candidate.get("output_node_id") != parent_id:
            continue
        ended = candidate.get("response_end_at")
        if ended is None:
            continue
        return (row["request_start_at"] - ended).total_seconds() * 1000.0
    return None


def overlapping(row: dict[str, Any], siblings: list[dict[str, Any]]) -> bool:
    """Whether another call in this trajectory was open at the same time."""
    start = row.get("request_start_at")
    if start is None:
        return False
    end = row.get("response_end_at") or start
    for other in siblings:
        if other["id"] == row["id"] or other.get("request_start_at") is None:
            continue
        other_end = other.get("response_end_at") or other["request_start_at"]
        if other["request_start_at"] < end and other_end > start:
            return True
    return False


def with_derived(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every row, with `gap_ms` and `overlapping` added."""
    return [
        {**row, "gap_ms": gap_ms(row, rows), "overlapping": overlapping(row, rows)} for row in rows
    ]


def gap_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The observed waits between consecutive model calls.

    The sign is kept: a negative gap means a call started before its parent
    finished, which is a real observation rather than an error.
    """
    gaps = sorted(row["gap_ms"] for row in rows if row.get("gap_ms") is not None)
    if not gaps:
        return {
            "samples": 0, "min_ms": None, "max_ms": None, "mean_ms": None,
            "p50_ms": None, "p95_ms": None, "p99_ms": None,
            "overlapping": 0, "total_wait_ms": 0.0,
        }

    def percentile(fraction: float) -> float:
        # Linear interpolation, matching `percentile_cont`.
        if len(gaps) == 1:
            return gaps[0]
        position = fraction * (len(gaps) - 1)
        low = int(position)
        high = min(low + 1, len(gaps) - 1)
        return gaps[low] + (gaps[high] - gaps[low]) * (position - low)

    return {
        "samples": len(gaps),
        "min_ms": gaps[0],
        "max_ms": gaps[-1],
        "mean_ms": sum(gaps) / len(gaps),
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "overlapping": sum(1 for gap in gaps if gap < 0),
        "total_wait_ms": sum(gap for gap in gaps if gap >= 0),
    }
