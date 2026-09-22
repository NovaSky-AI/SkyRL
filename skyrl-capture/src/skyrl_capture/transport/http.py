"""The upstream transport: a minimal HTTP/1.1 client for the forward path.

The data plane needs very little from an HTTP client: send a request with
already-built headers and an already-serialized body, then read the response
either whole or chunk by chunk. It does not need redirects, cookies, auth
flows, content negotiation, event hooks, or response decoding -- the proxy is
transparent, and it pins ``accept-encoding: identity`` precisely so nothing has
to decode anything.

So this is that, and only that. There used to be an httpx implementation
beside it, selected by `UPSTREAM_TRANSPORT`. Forwarding through httpx reached
about 28% of the no-proxy ceiling -- the overhead inside the client library
rather than in capture -- and under concurrent token traffic it collapsed
rather than plateaued, so nothing would have chosen it; what it cost instead
was a second production path through the one piece of code that runs per
inference request, and a configuration matrix to test both under.

What it supports is what a transparent proxy needs, and it is worth being
exact about the limits, because anything else raises `TransportError` rather
than being handled quietly:

* HTTP/1.1 over TCP or TLS, one request per connection at a time;
* keep-alive pooling per origin;
* ``content-length``, ``chunked``, and read-until-close response bodies;
* no redirects, no content decoding, no cookies, no HTTP/2.
"""

from __future__ import annotations

import asyncio
import ssl
from collections import deque
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

from skyrl_capture.config import ProxyConfig

_MAX_HEADER_BYTES = 256 * 1024
_READ_CHUNK = 65536


class TransportError(Exception):
    """Any failure reaching or reading from the upstream."""


class _IdleConnectionClosed(TransportError):
    """A pooled connection the server had already closed.

    The one failure that is safe to retry: the request never reached anything.
    See `UpstreamTransport.send`.
    """


def _default_ssl_context(config: ProxyConfig) -> ssl.SSLContext:
    """Verifying TLS context, optionally trusting an extra CA bundle.

    Certificate verification is never disabled: the proxy holds the upstream
    credential, so an unverified connection is exactly the case where it could
    be handed to the wrong server. A self-hosted upstream behind a private CA
    sets ``UPSTREAM_CA_BUNDLE`` instead.
    """
    context = ssl.create_default_context()
    if config.upstream_ca_bundle:
        context.load_verify_locations(cafile=config.upstream_ca_bundle)
    return context


class _Connection:
    """One pooled HTTP/1.1 connection."""

    __slots__ = ("reader", "writer", "origin", "in_use", "broken", "reused")

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, origin: tuple) -> None:
        self.reader = reader
        self.writer = writer
        self.origin = origin
        self.in_use = True
        self.broken = False
        # Whether this connection came out of the pool rather than being dialed
        # for this request. It is what decides whether a failure may be retried.
        self.reused = False

    def close(self) -> None:
        self.broken = True
        try:
            self.writer.close()
        except Exception:
            pass

    @property
    def reusable(self) -> bool:
        return not self.broken and not self.writer.is_closing() and not self.reader.at_eof()


class UpstreamResponse:
    """A response being read from a pooled connection."""

    __slots__ = (
        "status_code",
        "_headers",
        "_lookup",
        "_connection",
        "_pool",
        "_length",
        "_chunked",
        "_keep_alive",
        "_consumed",
        "_closed",
        "_prefetched",
        "_read_timeout",
    )

    def __init__(
        self,
        status_code: int,
        headers: list[tuple[bytes, bytes]],
        connection: _Connection,
        pool: UpstreamTransport,
        *,
        length: int | None,
        chunked: bool,
        keep_alive: bool,
        read_timeout: float,
        prefetched: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self._headers = headers
        self._lookup = {name.lower(): value for name, value in headers}
        self._connection = connection
        self._pool = pool
        self._length = length
        self._chunked = chunked
        self._keep_alive = keep_alive
        self._consumed = False
        self._closed = False
        self._prefetched = prefetched
        self._read_timeout = read_timeout

    def raw_headers(self) -> list[tuple[bytes, bytes]]:
        return self._headers

    def header(self, name: str) -> str:
        return self._lookup.get(name.lower().encode(), b"").decode("latin-1")

    async def read(self) -> bytes:
        parts = [chunk async for chunk in self.aiter_raw()]
        return b"".join(parts)

    async def _recv(self, reader: asyncio.StreamReader, size: int) -> bytes:
        """One read, bounded by the idle timeout.

        The bound is between reads rather than over the whole body: a long
        generation streams for minutes and is healthy the whole time, while an
        upstream that has stopped sending is not. Without this a stalled stream
        held the request open for ever -- the header read was bounded and
        nothing after it was.
        """
        try:
            return await asyncio.wait_for(reader.read(size), timeout=self._read_timeout)
        except TimeoutError as error:
            raise TransportError(
                f"upstream sent nothing for {self._read_timeout:g}s while its response was open"
            ) from error

    async def aiter_raw(self) -> AsyncIterator[bytes]:
        if self._consumed:
            return
        self._consumed = True
        reader = self._connection.reader
        try:
            if self._chunked:
                async for chunk in self._iter_chunked(reader):
                    yield chunk
            elif self._length is not None:
                remaining = self._length
                if self._prefetched:
                    take = self._prefetched[:remaining]
                    remaining -= len(take)
                    if take:
                        yield take
                while remaining > 0:
                    data = await self._recv(reader, min(_READ_CHUNK, remaining))
                    if not data:
                        raise TransportError("upstream closed before content-length was satisfied")
                    remaining -= len(data)
                    yield data
            else:
                # No length and not chunked: the body runs to connection close,
                # so the connection cannot be reused afterwards.
                self._keep_alive = False
                if self._prefetched:
                    yield self._prefetched
                while True:
                    data = await self._recv(reader, _READ_CHUNK)
                    if not data:
                        break
                    yield data
        except (TransportError, asyncio.IncompleteReadError, ConnectionError, OSError) as error:
            self._connection.broken = True
            if isinstance(error, TransportError):
                raise
            raise TransportError(f"reading upstream response failed: {error}") from error
        finally:
            await self.aclose()

    async def _iter_chunked(self, reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
        """Decode chunked transfer-encoding.

        The proxy re-frames the body for its own client, so chunk boundaries are
        decoded here rather than forwarded verbatim. Timing is preserved: each
        upstream chunk is yielded as it arrives, which is what streaming chunk
        timing measures.
        """
        buffered = self._prefetched
        self._prefetched = b""

        async def read_line() -> bytes:
            nonlocal buffered
            while b"\r\n" not in buffered:
                data = await self._recv(reader, _READ_CHUNK)
                if not data:
                    raise TransportError("upstream closed mid-chunk")
                buffered += data
            line, _, buffered = buffered.partition(b"\r\n")
            return line

        async def read_exactly(count: int) -> bytes:
            nonlocal buffered
            while len(buffered) < count:
                data = await self._recv(reader, max(_READ_CHUNK, count - len(buffered)))
                if not data:
                    raise TransportError("upstream closed mid-chunk")
                buffered += data
            out, buffered = buffered[:count], buffered[count:]
            return out

        while True:
            line = await read_line()
            size_text = line.split(b";", 1)[0].strip()
            try:
                size = int(size_text, 16)
            except ValueError as error:
                raise TransportError(f"malformed chunk size {size_text!r}") from error
            if size == 0:
                # Trailers, then the terminating blank line.
                while True:
                    trailer = await read_line()
                    if not trailer:
                        break
                return
            payload = await read_exactly(size)
            await read_exactly(2)  # the CRLF after the chunk
            yield payload

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        connection = self._connection
        if self._keep_alive and self._consumed and connection.reusable:
            self._pool.release(connection)
        else:
            connection.close()


class UpstreamTransport:
    """The forward path's HTTP client."""

    def __init__(self, config: ProxyConfig, *, ssl_context: ssl.SSLContext | None = None) -> None:
        self._config = config
        self._pools: dict[tuple, deque[_Connection]] = {}
        self._ssl_context = ssl_context or _default_ssl_context(config)
        self._max_idle_per_origin = max(1, config.upstream_idle_connections)
        self._connect_timeout = config.upstream_connect_timeout
        self._read_timeout = config.upstream_read_timeout

    # -- pooling -----------------------------------------------------------
    #
    # There is no cap on connections in flight, deliberately. A cap would make
    # capture queue inference requests behind each other, which is the one
    # thing the design forbids: the upstream decides its own concurrency, and
    # this is a proxy in front of it rather than a scheduler for it. What is
    # bounded is how many *idle* connections are kept per origin, which is a
    # memory and file-descriptor question rather than a throughput one.
    def release(self, connection: _Connection) -> None:
        connection.in_use = False
        pool = self._pools.setdefault(connection.origin, deque())
        if len(pool) < self._max_idle_per_origin:
            pool.append(connection)
        else:
            connection.close()

    async def _acquire(self, origin: tuple) -> _Connection:
        pool = self._pools.get(origin)
        while pool:
            candidate = pool.popleft()
            if candidate.reusable:
                candidate.in_use = True
                candidate.reused = True
                return candidate
            candidate.close()

        scheme, host, port = origin
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    host,
                    port,
                    ssl=self._ssl_context if scheme == "https" else None,
                    server_hostname=host if scheme == "https" else None,
                    limit=_READ_CHUNK,
                ),
                timeout=self._connect_timeout,
            )
        except TimeoutError as error:
            raise TransportError(f"connecting to {host}:{port} timed out") from error
        except (OSError, ssl.SSLError) as error:
            raise TransportError(f"connecting to {host}:{port} failed: {error}") from error
        return _Connection(reader, writer, origin)

    # -- request path ------------------------------------------------------
    async def send(
        self,
        method: str,
        url: str,
        headers: list[tuple[bytes, bytes]],
        body: bytes,
        *,
        stream: bool,
    ) -> UpstreamResponse:
        parts = urlsplit(url)
        scheme = parts.scheme or "http"
        if scheme not in ("http", "https"):
            raise TransportError(f"unsupported scheme {scheme!r}")
        host = parts.hostname or ""
        port = parts.port or (443 if scheme == "https" else 80)
        origin = (scheme, host, port)
        target = parts.path or "/"
        if parts.query:
            target = f"{target}?{parts.query}"

        authority = host if parts.port is None else f"{host}:{port}"
        request_lines = [f"{method} {target} HTTP/1.1".encode(), b"host: " + authority.encode()]
        for name, value in headers:
            request_lines.append(name + b": " + value)
        request_lines.append(b"content-length: " + str(len(body)).encode())
        request_lines.append(b"")
        request_lines.append(b"")
        head = b"\r\n".join(request_lines)

        # At most one retry, and only for the race this pool creates: a server
        # may close an idle keep-alive connection at any moment, and a request
        # written into one that closed never reached anything.
        #
        # Nothing else is retried, however it fails. Every request through here
        # is an inference call: an upstream that read the request and then died
        # may already have generated, and a retry would generate again and bill
        # again. So a *fresh* connection is never retried, and a reused one is
        # retried only when the response ended before its first byte -- which
        # is what a closed idle connection looks like and what a half-finished
        # generation does not.
        for attempt in (0, 1):
            connection = await self._acquire(origin)
            try:
                connection.writer.write(head + body if body else head)
                await connection.writer.drain()
                return await self._read_head(connection)
            except (TransportError, ConnectionError, OSError) as error:
                retryable = connection.reused and isinstance(error, _IdleConnectionClosed)
                connection.close()
                if attempt == 0 and retryable:
                    continue
                raise TransportError(f"upstream request failed: {error}") from error
        raise TransportError("upstream request failed")

    async def _read_head(self, connection: _Connection) -> UpstreamResponse:
        reader = connection.reader
        try:
            buffered = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), timeout=self._read_timeout
            )
        except asyncio.LimitOverrunError as error:
            raise TransportError("response headers exceeded the read buffer") from error
        except asyncio.IncompleteReadError as error:
            if not error.partial:
                # Nothing at all came back. On a reused connection that is the
                # server having closed it before our request arrived, which
                # `send` may retry; on a fresh one it is a real failure.
                raise _IdleConnectionClosed(
                    "upstream closed the connection without sending a response"
                ) from error
            raise TransportError("upstream closed before sending response headers") from error
        except TimeoutError as error:
            raise TransportError("upstream response timed out") from error
        if len(buffered) > _MAX_HEADER_BYTES:
            raise TransportError("response headers too large")

        head, _, remainder = buffered.partition(b"\r\n\r\n")
        lines = head.split(b"\r\n")
        status_line = lines[0]
        pieces = status_line.split(b" ", 2)
        if len(pieces) < 2 or not pieces[0].startswith(b"HTTP/1."):
            raise TransportError(f"malformed status line {status_line!r}")
        try:
            status_code = int(pieces[1])
        except ValueError as error:
            raise TransportError(f"malformed status code in {status_line!r}") from error

        headers: list[tuple[bytes, bytes]] = []
        length: int | None = None
        chunked = False
        keep_alive = True
        for line in lines[1:]:
            name, separator, value = line.partition(b":")
            if not separator:
                continue
            value = value.strip()
            lowered = name.lower()
            headers.append((name, value))
            if lowered == b"content-length":
                try:
                    length = int(value)
                except ValueError as error:
                    raise TransportError(f"malformed content-length {value!r}") from error
            elif lowered == b"transfer-encoding":
                if b"chunked" in value.lower():
                    chunked = True
            elif lowered == b"connection" and value.lower() == b"close":
                keep_alive = False

        if status_code == 100:
            raise TransportError("unexpected 1xx informational response")
        # A body-less response has no content to read regardless of headers.
        if status_code in (204, 304):
            length, chunked = 0, False

        return UpstreamResponse(
            status_code,
            headers,
            connection,
            self,
            length=length,
            chunked=chunked,
            keep_alive=keep_alive,
            read_timeout=self._read_timeout,
            prefetched=remainder,
        )

    async def aclose(self) -> None:
        for pool in self._pools.values():
            while pool:
                pool.popleft().close()
        self._pools.clear()

    def stats(self) -> dict[str, int]:
        return {
            "idle_connections": sum(len(pool) for pool in self._pools.values()),
            "origins": len(self._pools),
        }
