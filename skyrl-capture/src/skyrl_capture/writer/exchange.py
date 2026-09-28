"""One exchange as observed by a capture proxy.

Raw bytes and clock readings, and nothing else. Every derived field --
provider ids, usage, parsed messages, graph nodes -- is computed from these by
`core.derive` after the response has gone to the client, so the request
path itself never parses a provider payload.

It exists as its own type because it is the contract between a proxy, which
knows how to observe a turn, and the commit, which knows what a turn means.
Two proxies fill it and one function reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StreamChunkTiming:
    """One streaming chunk observation.

    ``monotonic_ns`` is a ``time.perf_counter_ns`` reading, so chunk deltas stay
    correct across wall-clock adjustments.
    """

    index: int
    monotonic_ns: int
    byte_count: int


@dataclass(slots=True)
class Exchange:
    """One observed request/response exchange."""

    exchange_id: str
    trajectory_id: str
    project: str
    provider: str
    endpoint_kind: str
    method: str
    path: str
    query: str

    # Clocks. Wall-clock values are unix nanoseconds; monotonic values are
    # ``perf_counter_ns`` readings from this process and are only meaningful as
    # differences within one process lifetime.
    request_start_wall_ns: int
    request_start_mono_ns: int
    first_byte_mono_ns: int | None = None
    response_end_wall_ns: int | None = None
    response_end_mono_ns: int | None = None

    http_status: int | None = None
    streaming: bool = False
    request_headers: list[tuple[str, str]] = field(default_factory=list)
    response_headers: list[tuple[str, str]] = field(default_factory=list)
    request_body: bytes = b""
    response_body: bytes = b""
    # Set when the proxy elected not to keep bodies (sampling mode).
    bodies_omitted: bool = False
    request_byte_count: int = 0
    response_byte_count: int = 0

    chunk_timings: list[StreamChunkTiming] = field(default_factory=list)
    chunk_count: int = 0
    # Transport-level failure (no HTTP status was ever received).
    transport_error: str | None = None
    retry_attempt: int = 0
    source_metadata: dict[str, Any] = field(default_factory=dict)

    # Tokens-only payload, attached by the tokens route. Text-mode exchanges
    # leave this unset.
    tokens: dict[str, Any] | None = None
