"""The upstream transport.

There is one, and every capture, streaming and token test in the suite runs
through it over real sockets. What is left for this file is the HTTP/1.1
detail a minimal client has to get right on its own, and the three places
where getting it wrong is expensive rather than merely broken: retrying an
inference that may already have run, hanging on a stalled stream, and holding
connections it should have dropped.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest
import pytest_asyncio

from skyrl_capture.config import ProxyConfig
from skyrl_capture.transport.http import TransportError, UpstreamTransport


# -- HTTP/1.1 details ------------------------------------------------------
class RawServer:
    """A socket server that replies with fixed bytes, for HTTP/1.1 details.

    Keep-alive means a handler sits blocked reading the next request, so
    teardown order matters: the client's connections are closed first (which
    gives each handler EOF), then the handlers are cancelled, then the server.
    Using ``async with server`` instead deadlocks, because ``wait_closed``
    waits for handlers that are still blocked on a read.
    """

    def __init__(self, response: bytes, *, close_after: bool = False) -> None:
        self._response = response
        self._close_after = close_after
        self.requests: list[bytes] = []
        self._server: asyncio.AbstractServer | None = None
        self._handlers: set[asyncio.Task] = set()
        self.port = 0

    async def start(self) -> RawServer:
        self._server = await asyncio.start_server(self._track, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def _track(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)
        try:
            await self._handle(reader, writer)
        finally:
            if task is not None:
                self._handlers.discard(task)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while True:
            try:
                head = await reader.readuntil(b"\r\n\r\n")
            except (asyncio.IncompleteReadError, ConnectionError, asyncio.CancelledError):
                break
            length = 0
            for line in head.split(b"\r\n"):
                name, separator, value = line.partition(b":")
                if separator and name.lower() == b"content-length":
                    length = int(value.strip())
            if length:
                try:
                    await reader.readexactly(length)
                except (asyncio.IncompleteReadError, ConnectionError):
                    break
            self.requests.append(head)
            try:
                writer.write(self._response)
                await writer.drain()
            except (ConnectionError, OSError):
                break
            if self._close_after:
                break
        with contextlib.suppress(Exception):
            writer.close()

    async def stop(self) -> None:
        for task in list(self._handlers):
            task.cancel()
        if self._handlers:
            await asyncio.gather(*self._handlers, return_exceptions=True)
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._server.wait_closed(), timeout=5)

    def url(self, path: str = "/") -> str:
        return f"http://127.0.0.1:{self.port}{path}"


def chunked(*payloads: bytes, trailers: bytes = b"") -> bytes:
    """Frame payloads as chunked transfer-encoding.

    Sizes are computed rather than written by hand: a hand-written hex length
    that disagrees with its payload produces a decode failure that looks like a
    transport bug.
    """
    parts = []
    for payload in payloads:
        parts.append(f"{len(payload):x}".encode() + b"\r\n" + payload + b"\r\n")
    parts.append(b"0\r\n" + trailers + b"\r\n")
    return b"".join(parts)


@pytest.fixture
def lean() -> UpstreamTransport:
    return UpstreamTransport(ProxyConfig())


@pytest_asyncio.fixture
async def raw_server(lean):
    """Start a raw server, and tear both it and the transport down safely."""
    servers: list[RawServer] = []

    async def start(response: bytes, *, close_after: bool = False) -> RawServer:
        server = await RawServer(response, close_after=close_after).start()
        servers.append(server)
        return server

    try:
        yield start
    finally:
        # Transport first: closing its sockets releases the blocked handlers.
        await lean.aclose()
        for server in servers:
            await server.stop()


async def test_lean_reads_a_content_length_body(lean, raw_server):
    server = await raw_server(
        b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 13\r\n\r\n"
        b'{"ok": true}\n'
    )
    response = await lean.send(
        "POST", server.url("/v1/x"), [(b"content-type", b"application/json")],
        b'{"a":1}', stream=False,
    )
    assert response.status_code == 200
    assert response.header("content-type") == "application/json"
    assert await response.read() == b'{"ok": true}\n'
    await response.aclose()
    # The request line, host header, and content-length are all synthesized.
    assert server.requests[0].startswith(b"POST /v1/x HTTP/1.1\r\n")
    assert b"host: 127.0.0.1" in server.requests[0]
    assert b"content-length: 7" in server.requests[0]


async def test_lean_decodes_chunked_and_preserves_boundaries(lean, raw_server):
    """Chunk boundaries are what streaming chunk timing measures."""
    server = await raw_server(
        b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
        b"transfer-encoding: chunked\r\n\r\n"
        + chunked(b"data: a", b"data: b", b"data: [DONE]\n\ntrailing!!")
    )
    response = await lean.send(
        "POST", server.url("/v1/chat/completions"), [], b"{}", stream=True
    )
    chunks = [chunk async for chunk in response.aiter_raw()]
    assert chunks == [b"data: a", b"data: b", b"data: [DONE]\n\ntrailing!!"]


async def test_lean_handles_chunked_with_trailers(lean, raw_server):
    server = await raw_server(
        b"HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n"
        + chunked(b"hello", trailers=b"x-trailer: value\r\n")
    )
    response = await lean.send("GET", server.url(), [], b"", stream=True)
    assert b"".join([chunk async for chunk in response.aiter_raw()]) == b"hello"


async def test_lean_reuses_a_pooled_connection(lean, raw_server):
    server = await raw_server(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nhi")
    for _ in range(3):
        response = await lean.send("GET", server.url(), [], b"", stream=False)
        assert await response.read() == b"hi"
        await response.aclose()
    assert lean.stats()["idle_connections"] == 1
    assert len(server.requests) == 3, "all three requests went down one connection"


async def test_lean_retries_once_when_a_pooled_connection_was_closed(lean, raw_server):
    """A server may close an idle keep-alive connection at any time.

    That race is indistinguishable from a real failure until the write or read
    fails, so one retry on a fresh connection is required for correctness, not
    just resilience.
    """
    server = await raw_server(
        b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok", close_after=True
    )
    first = await lean.send("GET", server.url(), [], b"", stream=False)
    assert await first.read() == b"ok"
    await first.aclose()
    # The server hung up; the next send must still succeed.
    second = await lean.send("GET", server.url(), [], b"", stream=False)
    assert await second.read() == b"ok"
    await second.aclose()
    assert len(server.requests) == 2


async def test_lean_does_not_reuse_a_connection_marked_close(lean, raw_server):
    server = await raw_server(
        b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nby"
    )
    response = await lean.send("GET", server.url(), [], b"", stream=False)
    assert await response.read() == b"by"
    await response.aclose()
    assert lean.stats()["idle_connections"] == 0


async def test_lean_treats_204_as_bodyless(lean, raw_server):
    server = await raw_server(b"HTTP/1.1 204 No Content\r\ncontent-length: 5\r\n\r\n")
    response = await lean.send("DELETE", server.url("/x"), [], b"", stream=False)
    assert response.status_code == 204
    assert await response.read() == b""
    await response.aclose()


async def test_lean_surfaces_a_truncated_body_as_an_error(lean, raw_server):
    """A short body must fail rather than be recorded as a complete response."""
    server = await raw_server(
        b"HTTP/1.1 200 OK\r\ncontent-length: 100\r\n\r\ntoo short", close_after=True
    )
    response = await lean.send("GET", server.url(), [], b"", stream=False)
    with pytest.raises(TransportError, match="content-length"):
        await response.read()


async def test_lean_rejects_a_malformed_status_line(lean, raw_server):
    server = await raw_server(
        b"NOT-HTTP 200\r\ncontent-length: 0\r\n\r\n", close_after=True
    )
    with pytest.raises(TransportError, match="malformed status line"):
        await lean.send("GET", server.url(), [], b"", stream=False)


async def test_lean_reports_a_refused_connection(lean):
    from conftest import free_port

    port = free_port()
    with pytest.raises(TransportError, match="failed|timed out"):
        await lean.send("GET", f"http://127.0.0.1:{port}/", [], b"", stream=False)
    await lean.aclose()


async def test_lean_carries_the_query_string_and_pools_per_origin(lean, raw_server):
    server = await raw_server(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n")
    response = await lean.send(
        "GET", server.url("/v1/models?limit=2&after=x"), [], b"", stream=False
    )
    await response.read()
    await response.aclose()
    assert server.requests[0].startswith(b"GET /v1/models?limit=2&after=x HTTP/1.1")
    assert lean.stats()["origins"] == 1


async def test_lean_rejects_an_unsupported_scheme(lean):
    with pytest.raises(TransportError, match="unsupported scheme"):
        await lean.send("GET", "ftp://example.com/x", [], b"", stream=False)
    await lean.aclose()





async def test_lean_handles_a_chunk_larger_than_the_read_buffer(lean, raw_server):
    """A chunk bigger than one socket read must be reassembled, not truncated."""
    payload = b"x" * (200 * 1024)
    server = await raw_server(
        b"HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n" + chunked(payload, b"tail")
    )
    response = await lean.send("GET", server.url(), [], b"", stream=True)
    chunks = [chunk async for chunk in response.aiter_raw()]
    assert chunks == [payload, b"tail"]


async def test_lean_rejects_a_malformed_chunk_size(lean, raw_server):
    server = await raw_server(
        b"HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\nzz\r\nbody\r\n0\r\n\r\n",
        close_after=True,
    )
    response = await lean.send("GET", server.url(), [], b"", stream=True)
    with pytest.raises(TransportError, match="malformed chunk size"):
        [chunk async for chunk in response.aiter_raw()]


async def test_lean_accepts_chunk_extensions(lean, raw_server):
    """A chunk size may carry `;ext=value`, which is not part of the length."""
    server = await raw_server(
        b"HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n"
        b"5;name=value\r\nhello\r\n0\r\n\r\n"
    )
    response = await lean.send("GET", server.url(), [], b"", stream=True)
    assert b"".join([chunk async for chunk in response.aiter_raw()]) == b"hello"


# -- TLS -------------------------------------------------------------------
def self_signed(directory):
    """Generate a throwaway certificate for 127.0.0.1."""
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    certificate_path = directory / "cert.pem"
    key_path = directory / "key.pem"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return certificate_path, key_path


async def test_lean_speaks_tls_and_verifies_the_certificate(tmp_path):
    """The forward path must work over HTTPS, and must verify.

    The proxy holds the upstream credential, so an unverified connection is
    exactly the case where it could be handed to the wrong server. This checks
    both halves: a trusted CA bundle works, and an untrusted certificate is
    refused.
    """
    import ssl as ssl_module

    certificate_path, key_path = self_signed(tmp_path)

    server_context = ssl_module.SSLContext(ssl_module.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(certfile=str(certificate_path), keyfile=str(key_path))

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        with contextlib.suppress(Exception):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-length: 5\r\n\r\nhello")
            await writer.drain()
        with contextlib.suppress(Exception):
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0, ssl=server_context)
    port = server.sockets[0].getsockname()[1]

    # Trusting the bundle: the request succeeds over TLS.
    trusting = UpstreamTransport(ProxyConfig(upstream_ca_bundle=str(certificate_path)))
    try:
        response = await trusting.send(
            "GET", f"https://127.0.0.1:{port}/", [], b"", stream=False
        )
        assert response.status_code == 200
        assert await response.read() == b"hello"
        await response.aclose()
    finally:
        await trusting.aclose()

    # Default trust store: the self-signed certificate must be refused.
    strict = UpstreamTransport(ProxyConfig())
    try:
        with pytest.raises(TransportError, match="failed|certificate"):
            await strict.send("GET", f"https://127.0.0.1:{port}/", [], b"", stream=False)
    finally:
        await strict.aclose()
        server.close()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(server.wait_closed(), timeout=5)


def test_certificate_verification_cannot_be_disabled_by_configuration():
    """There is deliberately no "insecure" switch for upstream TLS."""
    import inspect

    from skyrl_capture.config import ProxyConfig as Config
    from skyrl_capture.transport import headers, http

    fields = set(Config.__dataclass_fields__)
    assert not {"upstream_verify", "upstream_insecure", "verify_ssl"} & fields
    for module in (http, headers):
        source = inspect.getsource(module)
        assert "CERT_NONE" not in source
        assert "verify=False" not in source


# -- the three things that are expensive to get wrong ------------------------
class SilentServer:
    """A server that reads a request and then does something unhelpful."""

    def __init__(self, *, reply_after: float | None = None, hang: bool = False) -> None:
        self._reply_after = reply_after
        self._hang = hang
        self.requests = 0
        self._server: asyncio.AbstractServer | None = None
        self._handlers: set[asyncio.Task] = set()
        self.port = 0

    async def start(self) -> SilentServer:
        self._server = await asyncio.start_server(self._track, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def _track(self, reader, writer) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            length = 0
            for line in head.split(b"\r\n"):
                name, separator, value = line.partition(b":")
                if separator and name.lower() == b"content-length":
                    length = int(value.strip())
            if length:
                await reader.readexactly(length)
            self.requests += 1
            if self._hang:
                # Headers, then a stream that never produces another byte.
                writer.write(b"HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n")
                await writer.drain()
                await asyncio.sleep(60)
            else:
                # Read the request, then die without answering -- which is what
                # an upstream that already generated and then crashed looks like.
                writer.close()
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.CancelledError):
            pass
        finally:
            if task is not None:
                self._handlers.discard(task)

    async def stop(self) -> None:
        for task in list(self._handlers):
            task.cancel()
        if self._handlers:
            await asyncio.gather(*self._handlers, return_exceptions=True)
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._server.wait_closed(), timeout=5)

    def url(self, path: str = "/") -> str:
        return f"http://127.0.0.1:{self.port}{path}"


async def test_a_request_that_may_have_been_generated_is_never_retried(lean):
    """The expensive mistake this transport must not make.

    Every request through here is an inference call. An upstream that read the
    request and then failed may already have generated -- so retrying would
    generate twice, bill twice, and hand back whichever answer arrived second.
    A fresh connection is never retried, however it fails.
    """
    server = await SilentServer().start()
    try:
        with pytest.raises(TransportError):
            await lean.send("POST", server.url(), [], b'{"prompt": "expensive"}', stream=False)
        assert server.requests == 1, "the upstream saw the request exactly once"
    finally:
        await lean.aclose()
        await server.stop()


async def test_only_a_reused_connection_that_answered_nothing_is_retried(lean, raw_server):
    """The one retry that is safe, and the reason the pool can exist.

    A server may close an idle keep-alive connection at any moment. A request
    written into one that closed reached nothing, so it can be sent again --
    and that is the only case, which is why the previous test holds.
    """
    server = await raw_server(
        b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok", close_after=True
    )
    first = await lean.send("POST", server.url(), [], b"{}", stream=False)
    assert await first.read() == b"ok"
    await first.aclose()

    second = await lean.send("POST", server.url(), [], b"{}", stream=False)
    assert await second.read() == b"ok"
    await second.aclose()
    assert len(server.requests) == 2, "once down the dead connection, once down a fresh one"


async def test_a_stalled_stream_times_out_rather_than_hanging():
    """The read timeout covers the body, not just the headers.

    It used to bound only the header read, so an upstream that sent headers and
    then stopped held the request open for ever -- and a streaming response is
    exactly where that happens. The bound is between reads rather than over the
    whole body, because a long generation is slow on purpose and a silent one
    is not.
    """
    transport = UpstreamTransport(ProxyConfig(upstream_read_timeout=0.2))
    server = await SilentServer(hang=True).start()
    try:
        response = await transport.send("POST", server.url(), [], b"{}", stream=True)
        with pytest.raises(TransportError, match="sent nothing for"):
            async for _chunk in response.aiter_raw():
                pass
    finally:
        await transport.aclose()
        await server.stop()


async def test_idle_connections_are_bounded_and_requests_in_flight_are_not(raw_server):
    """The pool bounds what is kept, not what is running.

    Capping requests in flight would make capture queue inference behind
    itself, which is the one thing the design forbids. What is bounded is idle
    connections per origin: a file-descriptor question, not a throughput one.
    """
    transport = UpstreamTransport(ProxyConfig(upstream_idle_connections=1))
    server = await raw_server(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nhi")
    try:
        # Four at once: all four are served, so nothing queued behind the cap.
        responses = [
            await transport.send("POST", server.url(), [], b"{}", stream=True) for _ in range(4)
        ]
        assert len(server.requests) == 4
        for response in responses:
            assert await response.read() == b"hi"
            await response.aclose()
        # And only one is kept afterwards.
        assert transport.stats()["idle_connections"] == 1
    finally:
        await transport.aclose()
