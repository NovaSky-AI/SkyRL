"""From an observed exchange to one journal `ExchangeCommitted`.

Parse the bodies, derive the exchange row, plan the graph change against the
trajectory's own graph, and hand back the record. Everything here reads the
hot aggregate and changes nothing on it except the per-trajectory sequence it
claims; the caller applies the record and submits it for persistence.

Doing this in the proxy, after the response has been observed, is what makes
the aggregate complete the moment a turn ends. What is *not* complete at that
moment is the journal, and the two capture modes differ only in whether they
wait for it: text returns and commits behind the response, TITO commits before
the response closes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from skyrl_capture.domain.graph import GraphDelta
from skyrl_capture.domain.hashing import bytes_hash, normalize_json, stable_hash
from skyrl_capture.domain.records import ActiveTrajectory, CapturedExchange, now
from skyrl_capture.persistence import journal
from skyrl_capture.version import DERIVATION_VERSION, SCHEMA_VERSION
from skyrl_capture.writer.exchange import Exchange

# Endpoint kinds that carry message history and participate in the graph.
GRAPHED_KINDS = frozenset({"chat_completions", "responses", "messages"})

# What the token payload keeps in the exchange's metadata: everything but the
# arrays, which are on the nodes.
_TOKEN_ARRAYS = ("nodes", "prompt_token_ids", "prompt_message_indices", "completion_ids", "completion_logprobs")

# Never persisted, whatever the configured allowlist says.
_ALWAYS_DROP_HEADERS = frozenset(
    {"authorization", "x-api-key", "proxy-authorization", "cookie", "set-cookie", "api-key"}
)


def _filter_headers(headers: list[tuple[str, str]], allowlist: tuple[str, ...]) -> dict[str, str]:
    """Keep only allowlisted headers, and never keep a credential."""
    allowed = {name.lower() for name in allowlist}
    return {
        name.lower(): value
        for name, value in headers
        if name.lower() in allowed and name.lower() not in _ALWAYS_DROP_HEADERS
    }


def _extract_provider_ids(headers: list[tuple[str, str]]) -> dict[str, str]:
    """Extract common request identifiers from upstream response headers."""
    found: dict[str, str] = {}
    for name, value in headers:
        lowered = name.lower()
        if lowered in ("x-request-id", "request-id"):
            found["provider_request_id"] = value
        elif lowered == "openai-processing-ms":
            found["processing_ms"] = value
    return found


def _duration_ms(start_ns: int | None, end_ns: int | None) -> float | None:
    if start_ns is None or end_ns is None:
        return None
    return (end_ns - start_ns) / 1_000_000.0


def _stream_summary(
    timings: list[tuple[int, int, int]],
    *,
    request_start_mono_ns: int,
    response_end_mono_ns: int | None,
) -> dict[str, Any]:
    """Summarize streaming cadence from per-chunk arrival times."""
    if not timings:
        return {"chunk_count": 0}
    first = timings[0]
    last = timings[-1]
    deltas = [
        (timings[index][1] - timings[index - 1][1]) / 1_000_000.0
        for index in range(1, len(timings))
    ]
    summary: dict[str, Any] = {
        "chunk_count": len(timings),
        "first_chunk_ms": (first[1] - request_start_mono_ns) / 1_000_000.0,
        "last_chunk_ms": (last[1] - request_start_mono_ns) / 1_000_000.0,
        "total_bytes": sum(item[2] for item in timings),
        "mean_inter_chunk_ms": (sum(deltas) / len(deltas)) if deltas else None,
        "max_inter_chunk_ms": max(deltas) if deltas else None,
    }
    if response_end_mono_ns is not None and len(timings) > 1:
        span_ms = (last[1] - first[1]) / 1_000_000.0
        summary["chunks_per_second"] = (
            (len(timings) - 1) / (span_ms / 1000.0) if span_ms > 0 else None
        )
    return summary


def _wall(nanoseconds: int | None) -> datetime | None:
    return datetime.fromtimestamp(nanoseconds / 1e9, tz=UTC) if nanoseconds else None


def derive_exchange(
    active: ActiveTrajectory,
    observed: Exchange,
    *,
    protocol: Any,
    header_allowlist: tuple[str, ...],
    delivery_confirmed: bool = True,
) -> journal.ExchangeCommitted:
    """The journal record for one observed exchange.

    ``delivery_confirmed`` is false only on the TITO path, where the exchange
    is made durable *before* the response closes and a second record confirms
    the send afterwards. Text capture commits after the response has gone, so
    delivery is already a fact by the time this runs.
    """
    trajectory_id = active.id
    exchange_id = observed.exchange_id
    tokens_payload = observed.tokens

    # Reading the bodies belongs to the protocol adapter -- the one resolved at
    # startup, handed in rather than looked up per response. A provider with a
    # novel shape implements `extract` and touches nothing here. What comes
    # back is provider-native messages; identity is applied below, once, the
    # same way for every provider.
    parsed = protocol.extract(
        endpoint_kind=observed.endpoint_kind,
        request_body=observed.request_body,
        response_body=observed.response_body,
        streaming=observed.streaming,
        status=observed.http_status,
    )
    tools = normalize_json(parsed.tools) if parsed.tools is not None else None
    tools_hash = stable_hash(tools if tools is not None else [])

    request_start_at = _wall(observed.request_start_wall_ns)
    response_end_at = _wall(observed.response_end_wall_ns)
    request_start_mono = observed.request_start_mono_ns
    first_byte_mono = observed.first_byte_mono_ns
    response_end_mono = observed.response_end_mono_ns
    chunk_timings = [(c.index, c.monotonic_ns, c.byte_count) for c in observed.chunk_timings]

    header_ids = _extract_provider_ids(observed.response_headers)
    summary = _stream_summary(
        chunk_timings,
        request_start_mono_ns=request_start_mono or 0,
        response_end_mono_ns=response_end_mono,
    )
    if parsed.stream_summary:
        summary.update(parsed.stream_summary)

    # "Late" means the turn's own clock reading is after finish was requested.
    # The gate reads live status, so a request arriving after finish is refused
    # outright; what is left is the microsecond race where a request passes the
    # gate and finish lands before its clock is read. Marked, never refused: the
    # turn happened, and the trajectory should say so.
    finish_requested_at = active.finish_requested_at
    late = finish_requested_at is not None and request_start_at > finish_requested_at

    previous_exchange_id = None
    if parsed.previous_response_id:
        previous_exchange_id = active.exchange_by_response_id(parsed.previous_response_id)

    # -- the graph change this exchange makes --------------------------------------
    graph = active.graph
    delta: GraphDelta | None = None
    source_metadata = {**observed.source_metadata, "parse_errors": parsed.errors}
    model = parsed.model or (tokens_payload or {}).get("model")
    if tokens_payload and tokens_payload.get("nodes") is not None:
        # Token nodes were computed and verified by the proxy, which owns the
        # renderer. The graph takes exactly that structure -- re-deriving it
        # here could disagree with the tokens inference actually received.
        delta = graph.plan_token_append(
            exchange_id=exchange_id, tokens=tokens_payload, output_ended_at=response_end_at
        )
        source_metadata["tokens"] = {k: v for k, v in tokens_payload.items() if k not in _TOKEN_ARRAYS}
    elif observed.endpoint_kind in GRAPHED_KINDS and (parsed.input_messages or parsed.output_message):
        delta = graph.plan_text_append(
            exchange_id=exchange_id,
            request_messages=parsed.input_messages,
            output_message=parsed.output_message,
            tools_hash=tools_hash,
            model=parsed.model,
            output_started_at=(
                _wall(observed.request_start_wall_ns + (first_byte_mono - request_start_mono))
                if first_byte_mono and request_start_mono
                else None
            ),
            output_ended_at=response_end_at,
        )

    sequence = active.next_sequence()
    row = {
        # Stamped here rather than carried on the record: they describe this
        # build's reading of the exchange, not a fact about the turn.
        "schema_version": SCHEMA_VERSION,
        "derivation_version": DERIVATION_VERSION,
        "id": exchange_id,
        "trajectory_id": trajectory_id,
        "project": observed.project,
        "sequence": sequence,
        "provider": observed.provider,
        "model": model,
        "endpoint_kind": observed.endpoint_kind,
        "method": observed.method,
        "path": observed.path,
        "query": observed.query or "",
        "request_start_at": request_start_at,
        "response_end_at": response_end_at,
        "request_headers": _filter_headers(observed.request_headers, header_allowlist),
        "response_headers": _filter_headers(observed.response_headers, header_allowlist),
        "http_status": observed.http_status,
        "streaming": observed.streaming,
        "transport_error": observed.transport_error,
        "provider_error": parsed.provider_error,
        "retry_attempt": observed.retry_attempt,
        "completion_reason": parsed.finish_reason or (tokens_payload or {}).get("stop_reason"),
        "clock_epoch": observed.source_metadata.get("clock_epoch"),
        "request_start_mono_ns": request_start_mono,
        "first_byte_mono_ns": first_byte_mono,
        "response_end_mono_ns": response_end_mono,
        "ttft_ms": _duration_ms(request_start_mono, first_byte_mono),
        "duration_ms": _duration_ms(request_start_mono, response_end_mono),
        "chunk_count": observed.chunk_count,
        "stream_summary": summary,
        "usage": parsed.usage,
        # What was asked for is first class: a training row cannot be
        # reproduced from a grab-bag field nobody thinks to read.
        "sampling": parsed.parameters,
        "max_output_tokens": parsed.max_output_tokens,
        "tools": tools,
        "tools_hash": tools_hash,
        "request_byte_count": observed.request_byte_count,
        "response_byte_count": observed.response_byte_count,
        "provider_request_id": header_ids.get("provider_request_id"),
        "provider_response_id": parsed.provider_response_id,
        "previous_response_id": parsed.previous_response_id,
        "source_metadata": source_metadata,
        "previous_exchange_id": previous_exchange_id,
        "late": late,
        "bodies_omitted": observed.bodies_omitted,
        # Whether the log holds this exchange's bodies. The live state never
        # does; see `ExchangeCommitted`.
        "has_payload": bool(observed.request_body or observed.response_body),
        # No graph change: an ungraphed endpoint, or a poisoned turn.
        "input_prefix_node_id": None,
        "input_node_ids": [],
        "input_leaf_node_id": None,
        "output_node_id": None,
        "parent_output_node_id": None,
        "is_duplicate_retry": False,
        **(delta.association() if delta is not None else {}),
    }
    chunks = (
        {"frames": parsed.stream_frames, "timings": [list(item) for item in chunk_timings]}
        if parsed.stream_frames
        else None
    )
    return journal.ExchangeCommitted(
        exchange=CapturedExchange(
            id=exchange_id,
            sequence=sequence,
            row=row,
            delivery_confirmed=delivery_confirmed,
            request_body=observed.request_body,
            response_body=observed.response_body,
            chunks=chunks,
            tokens=(
                {k: v for k, v in tokens_payload.items() if k != "nodes"}
                if tokens_payload
                else None
            ),
        ),
        graph=delta,
        at=now(),
        # Diagnostic only. Two turns of one trajectory may send byte-identical
        # requests on purpose -- that is resampling -- so this must never be
        # used to deduplicate, and nothing reads it to decide anything.
        request_fingerprint=bytes_hash(observed.request_body),
    )
