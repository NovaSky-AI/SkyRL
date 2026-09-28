"""Text-mode capture: forward one request, record what was observed.

This is the only code in the system that runs once per ordinary inference
request, so it is a hand-written ASGI handler rather than a framework route.
The whole capture contribution to a request is:

* four ``perf_counter_ns`` reads,
* appending already-available byte objects to a list,
* and, after the last byte has gone to the client, one derivation and one
  queued journal append.

No JSON parsing of provider payloads, no hashing, no storage on the way to the
client. Once the response has been sent, the exchange is parsed, its graph
change planned, and the journal append queued -- CPU work on this task, disk
work on another. The client already has every byte by then, and nothing it
sees depends on the append landing.

A capture failure is caught and counted; it never changes a response the
upstream already produced successfully. Queue saturation, a parse failure and
a dead disk all end in the same place: a gap marked on the trajectory, which
its final record carries. That is the fail-open half of the product contract,
and it is the deliberate opposite of what token capture does.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from skyrl_capture.config import ProxyConfig, TextUpstream
from skyrl_capture.domain.records import ActiveTrajectory
from skyrl_capture.ids import exchange_id
from skyrl_capture.transport import headers
from skyrl_capture.transport.http import TransportError, UpstreamTransport
from skyrl_capture.transport.response import send_response
from skyrl_capture.writer.commits import CommitCoordinator
from skyrl_capture.writer.derive import derive_exchange
from skyrl_capture.writer.exchange import Exchange, StreamChunkTiming

logger = logging.getLogger(__name__)

Scope = MutableMapping[str, Any]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

_JSON_HEADERS = [(b"content-type", b"application/json")]

# A request that did not ask for a stream gets the buffered forward path, which
# skips the incremental read/write machinery entirely. Detected by substring
# rather than by parsing the body: parsing JSON on the request path is exactly
# what this design avoids, and a false positive is harmless because the
# streaming path is correct for any response.
_STREAM_HINTS = (b'"stream":true', b'"stream": true')


def wants_stream(body: bytes) -> bool:
    return any(hint in body for hint in _STREAM_HINTS)


class TextProxy:
    def __init__(
        self,
        *,
        transport: UpstreamTransport,
        upstream: TextUpstream,
        proxy: ProxyConfig,
        commits: CommitCoordinator,
        header_allowlist: tuple[str, ...],
        clock_epoch: str,
    ) -> None:
        # The data plane depends on the small transport protocol, not on any
        # particular HTTP client, so the client can be replaced without editing
        # the hot path.
        self._transport = transport
        # One upstream, fixed at startup: its protocol, its URL and its
        # credential are read here once instead of per request.
        self._upstream = upstream
        # Everything provider-specific, resolved once. The forward path asks it
        # where to send and how to authenticate; the commit path, afterwards,
        # asks it what the bytes meant.
        self._protocol = upstream.protocol
        self._upstream_credential = upstream.api_key
        self._commits = commits
        self._header_allowlist = header_allowlist
        self._clock_epoch = clock_epoch
        self._capture_enabled = proxy.capture_enabled
        self._capture_chunks = proxy.capture_stream_chunks
        self._max_chunk_records = proxy.max_stream_chunk_records
        self._sample_rate = proxy.payload_sample_rate
        self.capture_errors = 0
        self.capture_refused = 0

    async def handle(
        self,
        *,
        scope: Scope,
        send: Send,
        active: ActiveTrajectory,
        suffix: str,
        body: bytes,
        request_start_wall: int,
        request_start_mono: int,
    ) -> None:
        endpoint_kind = self._protocol.endpoint_kind(suffix)
        raw_headers: list[tuple[bytes, bytes]] = scope["headers"]
        query = scope.get("query_string", b"").decode("latin-1")
        method: str = scope["method"]

        protocol = self._protocol
        url = protocol.upstream_url(self._upstream.url, suffix, query)
        upstream_headers = headers.prepare_headers(
            raw_headers, protocol=protocol, credential=self._upstream_credential
        )

        chunks: list[bytes] = []
        chunk_timings: list[StreamChunkTiming] = []
        first_byte_mono: int | None = None
        status: int | None = None
        response_headers: list[tuple[bytes, bytes]] = []
        transport_error: str | None = None
        response_bytes = 0
        chunk_count = 0
        streaming = False

        expects_stream = wants_stream(body)
        try:
            response = await self._transport.send(
                method, url, upstream_headers, body, stream=expects_stream
            )
        except TransportError as error:
            transport_error = str(error)
            status = 502
            payload = protocol.error_body(502, f"upstream request failed: {error}")
            await send_response(send, 502, _JSON_HEADERS, payload)
        else:
            try:
                status = response.status_code
                response_headers = headers.filter_response_headers(response.raw_headers())
                streaming = "event-stream" in response.header("content-type")

                if not expects_stream and not streaming:
                    # Buffered forward: one read, one write, no incremental
                    # plumbing. The common case for chat completions.
                    body_bytes = await response.read()
                    now = time.perf_counter_ns()
                    first_byte_mono = now
                    response_bytes = len(body_bytes)
                    if body_bytes:
                        chunks.append(body_bytes)
                        chunk_count = 1
                        if self._capture_chunks:
                            chunk_timings.append(StreamChunkTiming(0, now, response_bytes))
                    await send_response(send, status, response_headers, body_bytes)
                else:
                    await send(
                        {
                            "type": "http.response.start",
                            "status": status,
                            "headers": response_headers,
                        }
                    )
                    capture_chunks = self._capture_chunks
                    max_records = self._max_chunk_records
                    async for chunk in response.aiter_raw():
                        if not chunk:
                            continue
                        now = time.perf_counter_ns()
                        if first_byte_mono is None:
                            first_byte_mono = now
                        chunks.append(chunk)
                        response_bytes += len(chunk)
                        if capture_chunks and chunk_count < max_records:
                            chunk_timings.append(StreamChunkTiming(chunk_count, now, len(chunk)))
                        chunk_count += 1
                        await send({"type": "http.response.body", "body": chunk, "more_body": True})
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
            except TransportError as error:
                # The upstream failed mid-stream. The client already has a
                # status line, so the stream is terminated and the failure is
                # recorded.
                transport_error = str(error)
                await send({"type": "http.response.body", "body": b"", "more_body": False})
            finally:
                await response.aclose()

        response_end_mono = time.perf_counter_ns()
        response_end_wall = time.time_ns()

        if not self._capture_enabled:
            return
        try:
            keep_payload = self._keep_payload(active)
            observed = Exchange(
                exchange_id=exchange_id(),
                trajectory_id=active.id,
                project=active.header.project,
                provider=self._upstream.type,
                endpoint_kind=endpoint_kind,
                method=method,
                path=suffix,
                query=query,
                request_start_wall_ns=request_start_wall,
                request_start_mono_ns=request_start_mono,
                first_byte_mono_ns=first_byte_mono,
                response_end_wall_ns=response_end_wall,
                response_end_mono_ns=response_end_mono,
                http_status=status,
                streaming=streaming,
                request_headers=decode_headers(raw_headers),
                response_headers=decode_headers(response_headers),
                request_body=body if keep_payload else b"",
                response_body=b"".join(chunks) if keep_payload else b"",
                bodies_omitted=not keep_payload,
                request_byte_count=len(body),
                response_byte_count=response_bytes,
                chunk_timings=chunk_timings,
                chunk_count=chunk_count,
                transport_error=transport_error,
                retry_attempt=retry_attempt(raw_headers),
                source_metadata={"clock_epoch": self._clock_epoch},
            )
            record = derive_exchange(
                active,
                observed,
                protocol=self._protocol,
                header_allowlist=self._header_allowlist,
            )
            # The aggregate takes the exchange now, so the next turn matches
            # against a graph that already contains this one. Persistence
            # happens behind the response, ordered per trajectory.
            active.add_exchange(record.exchange, record.graph, record.at)
            if self._commits.submit(active, record) is None:
                # The bound is reached. Refusing the work and saying so is the
                # fail-open answer; blocking here would make capture decide how
                # fast inference may run.
                self.capture_refused += 1
                self._commits.record_gap(
                    active, reason="commit queue full; exchange not persisted"
                )
        except Exception as error:  # pragma: no cover - capture must never raise
            self.capture_errors += 1
            self._commits.record_gap(active, reason=f"{type(error).__name__}: {error}")
            logger.warning("capture failed for trajectory %s: %s", active.id, error)

    def error_body(self, status: int, message: str) -> bytes:
        """An error in this upstream's shape, for the data plane to send."""
        return self._protocol.error_body(status, message)

    def _keep_payload(self, active: ActiveTrajectory) -> bool:
        """Full capture is the default; sampling is opt-in, per trajectory."""
        if active.header.bodies != "sampled":
            return True
        return random.random() < self._sample_rate

    def stats(self) -> dict[str, Any]:
        return {
            "capture_errors": self.capture_errors,
            "capture_refused": self.capture_refused,
            "capture_enabled": self._capture_enabled,
        }


def retry_attempt(headers: list[tuple[bytes, bytes]]) -> int:
    """Provider SDKs advertise their retry count; preserve it when present."""
    for name, value in headers:
        if name == b"x-stainless-retry-count":
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


def decode_headers(headers: list[tuple[bytes, bytes]]) -> list[tuple[str, str]]:
    return [(name.decode("latin-1"), value.decode("latin-1")) for name, value in headers]
