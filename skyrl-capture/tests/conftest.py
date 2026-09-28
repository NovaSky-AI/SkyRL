"""Test fixtures.

Everything runs locally and offline: there is nothing to provision -- the state
is a map the runtime owns -- and the providers are the in-repo mock upstreams.
No API keys, no network egress, no model downloads.

The stack runs over **real sockets through uvicorn**, not an in-process ASGI
transport. That costs about 50ms per test and is worth it: ``httpx``'s
``ASGITransport`` coalesces a response body into a single chunk, so streaming
chunk timing -- one of the things this system exists to measure -- cannot be
exercised through it at all. Running the real server also means tests cover the
actual HTTP framing, header handling, and chunked transfer that production
uses.

Everything else is wired exactly as production is: the same ``Runtime``, the
same data plane, the same stores writing the same record directory.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import tempfile
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest_asyncio
import uvicorn
from support.mock_upstream import MockUpstream

from skyrl_capture.application import CaptureApplication
from skyrl_capture.config import Config, ProxyConfig, RecordConfig, TextUpstream, TitoUpstream
from skyrl_capture.ids import trajectory_id as new_trajectory_id
from skyrl_capture.runtime import Runtime


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class LiveServer:
    """A uvicorn server running on a real port for the duration of a test."""

    def __init__(self, app: Any, port: int, *, lifespan: str = "off") -> None:
        self._config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            access_log=False,
            # "on" for the capture application, so the fixture drives the same
            # lifespan production does -- `build_runtime`, `Runtime.start`, and
            # `Runtime.stop` on the way out. The mock upstream ignores the
            # lifespan scope entirely, so it stays "off" or uvicorn waits for a
            # reply that never comes.
            lifespan=lifespan,
        )
        self._server = uvicorn.Server(self._config)
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._server.serve())
        for _ in range(400):
            if self._server.started:
                return
            if self._task.done():
                await self._task
                raise RuntimeError("server exited during startup")
            await asyncio.sleep(0.01)
        raise TimeoutError(f"server on port {self.port} did not start")

    async def stop(self) -> None:
        self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._task, timeout=15)
            self._task = None


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    """Fail a run in which nothing actually ran.

    A suite that skips every test still exits 0, which reads as "passing" to a
    human and to CI. That is how four bugs stayed invisible: the full path was
    untested *and* the run that should have said so was green.

    Set ``CAPTURE_ALLOW_ALL_SKIPPED=1`` when selecting a subset that is legitimately
    all-skipped, e.g. the real-tokenizer tests without a cached tokenizer.
    """
    if os.environ.get("CAPTURE_ALLOW_ALL_SKIPPED"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:  # pragma: no cover - no terminal plugin
        return
    ran = sum(len(reporter.stats.get(key, [])) for key in ("passed", "failed", "error"))
    skipped = len(reporter.stats.get("skipped", []))
    if ran == 0 and skipped > 0:
        reporter.write_line(
            f"ERROR: all {skipped} tests were skipped, so this run proved nothing. "
            "Set CAPTURE_ALLOW_ALL_SKIPPED=1 if that is what you meant.",
            red=True,
        )
        session.exitstatus = 1


class Stack:
    """A running capture service plus clients for both planes."""

    def __init__(
        self,
        runtime: Runtime,
        application: CaptureApplication,
        upstream: MockUpstream,
        client: httpx.AsyncClient,
        base_url: str,
    ) -> None:
        self.runtime = runtime
        self.application = application
        self.upstream = upstream
        self.client = client
        self.base_url = base_url.rstrip("/")

    # Both planes share one port, exactly as they do in the local profile.
    @property
    def control(self) -> httpx.AsyncClient:
        return self.client

    @property
    def registry(self):
        """The trajectories this process is hot for."""
        return self.runtime.registry

    def aggregate(self, trajectory_id: str):
        """One hot aggregate, or `None` once it has been finished and evicted."""
        return self.runtime.registry.hot(trajectory_id)

    def record(self, trajectory_id: str):
        """The committed record, or `None` while the trajectory is still open."""
        return self.runtime.committed.get_sync(trajectory_id)

    def node_rows(self, trajectory_id: str) -> list[dict]:
        active = self.aggregate(trajectory_id)
        if active is not None:
            return [node.public() for node in active.graph.ordered()]
        found = self.record(trajectory_id)
        return [node.public() for node in found.nodes] if found else []

    def exchange_rows(self, trajectory_id: str) -> list[dict]:
        active = self.aggregate(trajectory_id)
        if active is not None:
            return active.exchange_rows()
        found = self.record(trajectory_id)
        return [exchange.row for exchange in found.exchanges] if found else []

    def journal(self, trajectory_id: str):
        """This trajectory's journal, replayed off the disk as a replacement
        process would read it. Nothing here reaches into hot memory."""
        from skyrl_capture.persistence.active import read_journal, rebuild

        path = self.runtime.active.path_for(trajectory_id)
        return rebuild(read_journal(path).records, recovered=False)

    def journal_records(self, trajectory_id: str) -> list[Any]:
        from skyrl_capture.persistence.active import read_journal

        return read_journal(self.runtime.active.path_for(trajectory_id)).records

    def aggregates(self) -> list[Any]:
        """Every trajectory this process is hot for."""
        return [
            found
            for identifier in self.registry.hot_ids()
            if (found := self.aggregate(identifier)) is not None
        ]

    def records(self) -> list[Any]:
        """Every committed record in this stack's directory."""
        store = self.runtime.committed
        return [
            found
            for identifier in store.committed_ids()
            if (found := store.get_sync(identifier)) is not None
        ]

    def get_exchange(self, exchange_id: str) -> dict[str, Any] | None:
        """One derived exchange row, wherever the trajectory holding it is."""
        for active in self.aggregates():
            for exchange in active.exchanges:
                if exchange.id == exchange_id:
                    return exchange.row
        for record in self.records():
            for exchange in record.exchanges:
                if exchange.id == exchange_id:
                    return exchange.row
        return None

    async def stored_bodies(self, exchange_id: str) -> dict[str, Any] | None:
        """What the record holds for one exchange: raw request and response
        bytes, stream chunks, and the token payload.

        Read off the journal -- or the committed record, once there is one --
        after a settle, because that is where captured bodies live. The hot
        aggregate sheds them as soon as they are durable.

        Tests that check what was *stored* read it here rather than through a
        route. A public endpoint whose only caller is the test suite is surface
        the product carries for nothing.
        """
        from skyrl_capture.persistence.active import payload_of

        await self.settle()
        for record in self.records():
            for exchange in record.exchanges:
                if exchange.id == exchange_id:
                    return payload_of(exchange)
        for identifier in self.registry.hot_ids():
            replayed = self.journal(identifier)
            if replayed is None:
                continue
            for exchange in replayed.exchanges:
                if exchange.id == exchange_id:
                    return payload_of(exchange)
        return None

    async def stored_object(self, exchange_id: str, part: str = "request") -> Any:
        """One stored body, decoded as JSON."""
        import orjson

        bodies = await self.stored_bodies(exchange_id)
        if bodies is None or not bodies.get(part):
            return None
        return orjson.loads(bodies[part])

    async def node_payload(self, node_id: str) -> Any:
        """A node's message delta -- and in token mode its token arrays."""
        for active in self.aggregates():
            node = active.graph.nodes.get(node_id)
            if node is not None:
                return node.payload
        for record in self.records():
            node = next((item for item in record.nodes if item.id == node_id), None)
            if node is not None:
                return node.payload
        return None

    @property
    def data(self) -> httpx.AsyncClient:
        return self.client

    # -- control plane -----------------------------------------------------
    # There is no control credential, so these are plain requests.
    async def post(self, path: str, payload: Any = None, **kwargs: Any) -> httpx.Response:
        return await self.client.post(f"{self.base_url}{path}", json=payload, **kwargs)

    async def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return await self.client.get(f"{self.base_url}{path}", **kwargs)

    async def patch(self, path: str, payload: Any = None) -> httpx.Response:
        return await self.client.patch(f"{self.base_url}{path}", json=payload)

    async def delete(self, path: str) -> httpx.Response:
        return await self.client.delete(f"{self.base_url}{path}")

    # -- helpers -----------------------------------------------------------
    upstream_url: str = "http://127.0.0.1:0"

    async def create_trajectory(
        self, *, project: str = "test-project", **fields: Any
    ) -> dict[str, Any]:
        # The id is the caller's, generated before the request goes out, which
        # is what makes a repeat of this call idempotent.
        payload: dict[str, Any] = {
            "project": project,
            "trajectory_id": fields.pop("trajectory_id", None) or new_trajectory_id(),
        }
        payload.update(fields)
        response = await self.post("/v1/trajectories", payload)
        assert response.status_code == 201, response.text
        return response.json()

    async def chat(
        self,
        created: dict[str, Any],
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        model: str = "mock-model",
        headers: dict[str, str] | None = None,
        **body: Any,
    ) -> httpx.Response:
        """Send an OpenAI chat request through the capture route."""
        # No credential: capture authenticates nothing on the way in. The
        # header is here because an OpenAI client sends one, and proving it is
        # stripped rather than recorded is part of what these tests check.
        request_headers = {"authorization": "Bearer whatever-the-client-had"}
        if headers:
            request_headers.update(headers)
        payload = {"model": model, "messages": messages, "stream": stream, **body}
        url = f"{created['base_url']}/chat/completions"
        if stream:
            return await self._stream(url, payload, request_headers)
        return await self.client.post(url, json=payload, headers=request_headers)

    async def _stream(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        """Read a streaming response, recording how many chunks actually arrived."""
        chunks: list[bytes] = []
        async with self.client.stream("POST", url, json=payload, headers=headers) as response:
            async for chunk in response.aiter_raw():
                if chunk:
                    chunks.append(chunk)
            response._content = b"".join(chunks)  # noqa: SLF001 - test convenience
            response.client_chunks = len(chunks)  # type: ignore[attr-defined]
            return response

    async def messages(
        self, created: dict[str, Any], messages: list[dict[str, Any]], *, stream: bool = False, **body: Any
    ) -> httpx.Response:
        """Send an Anthropic Messages request through the capture route."""
        payload = {"model": "mock-claude", "messages": messages, "max_tokens": 256, "stream": stream, **body}
        url = f"{created['base_url']}/v1/messages"
        headers = {"x-api-key": "whatever-the-client-had"}
        if stream:
            return await self._stream(url, payload, headers)
        return await self.client.post(url, json=payload, headers=headers)

    async def settle(self) -> None:
        """Wait until everything submitted so far is durable in the record."""
        await self.runtime.commands.drain()

    async def refresh(self) -> None:
        """Make the viewer's index current. The API refreshes on its own every
        couple of seconds; a test is not waiting two seconds for it."""
        await self.settle()
        if self.runtime.reader is not None:
            await self.runtime.reader.refresh()

    async def token_trace(self, trajectory_id: str):
        """The live token trace for one trajectory.

        Reached through one helper so a test that needs to reach inside token
        capture -- to provoke an attribution failure, say -- names the seam
        once, and the seam can move without every such test moving with it.
        """
        return await self.runtime.proxy.sessions.trace_for(trajectory_id)

    async def run(self, run_id: str) -> dict[str, Any] | None:
        """One run, out of the listing -- there is no per-run route.

        A run's entry already carries the counts and `steps`, so picking it out
        of `/v1/runs` is what a caller does, and `None` is what "no such run"
        looks like from there.
        """
        response = await self.get("/v1/runs")
        assert response.status_code == 200, response.text
        return next((row for row in response.json()["data"] if row["id"] == run_id), None)

    async def exchanges(self, trajectory_id: str) -> list[dict[str, Any]]:
        await self.refresh()
        response = await self.get(f"/v1/trajectories/{trajectory_id}/exchanges?limit=500")
        assert response.status_code == 200, response.text
        return response.json()["data"]

    async def graph(self, trajectory_id: str) -> dict[str, Any]:
        await self.refresh()
        response = await self.get(f"/v1/trajectories/{trajectory_id}/graph")
        assert response.status_code == 200, response.text
        return response.json()

    async def finish(self, trajectory_id: str, **payload: Any) -> httpx.Response:
        response = await self.post(f"/v1/trajectories/{trajectory_id}/finish", payload)
        # A finished trajectory moves from `active/` to `committed/`, and a
        # listing that has not rescanned would still show the journal's view.
        if response.status_code == 200 and self.runtime.reader is not None:
            await self.runtime.reader.refresh()
        return response

    async def run_export(self, **fields: Any) -> dict[str, Any]:
        await self.refresh()
        response = await self.post("/v1/exports", fields)
        assert response.status_code == 202, response.text
        job = response.json()
        for _ in range(100):
            await self.runtime.exports.drain()
            status = (await self.get(f"/v1/exports/{job['id']}")).json()
            if status["status"] in ("ready", "failed"):
                return status
            await asyncio.sleep(0.05)
        raise AssertionError(f"export {job['id']} never finished: {status}")

    async def export_lines(self, **fields: Any) -> list[dict[str, Any]]:
        """Create an export, run it, and decode its JSONL records."""
        import orjson

        from skyrl_capture.compression import decompress

        job = await self.run_export(**fields)
        assert job["status"] == "ready", job
        return [orjson.loads(line) for line in decompress(self.artifact(job)).splitlines() if line.strip()]

    def artifact(self, job: dict[str, Any]) -> bytes:
        """One finished export's bytes, off the disk it was written to."""
        from pathlib import Path

        return Path(job["output_uri"].removeprefix("file://")).read_bytes()


async def _build_stack(
    *,
    tokens: bool = False,
    upstream_for: Any = None,
    config_overrides: dict[str, Any] | None = None,
) -> tuple[Stack, list[Any]]:
    """Bring up a mock upstream and a capture service on real ports.

    ``tokens`` decides the whole process's mode, because the upstream is
    startup configuration -- there is no second target to point a trajectory
    at. ``upstream_for`` is the general form: a callable given the mock
    upstream's base URL that returns the ``UpstreamConfig`` to serve, for a
    test that needs a provider these two shorthands do not cover.
    """
    # Persistence is not optional, so every stack has a record directory.
    record_directory = Path(tempfile.mkdtemp(prefix="capture-record-")) / "traces"

    upstream = MockUpstream(reply="mock reply text", chunks=3)
    upstream_server = LiveServer(upstream, free_port())
    await upstream_server.start()

    capture_port = free_port()
    upstream_config = upstream_for(upstream_server.url) if upstream_for else (
        TitoUpstream(
            type="tokens",
            url=f"{upstream_server.url}/generate",
            model="mock-tokens-model",
            tokenizer="builtin",
            api_key="upstream-secret",
            max_model_len=4096,
        )
        if tokens
        else TextUpstream(
            type="openai", url=f"{upstream_server.url}/v1", api_key="upstream-secret"
        )
    )
    defaults: dict[str, Any] = {
        "upstream": upstream_config,
        "record_dir": record_directory,
        # `fsync="never"` for speed. Appends still complete in order and a
        # commit still has to finish before a TITO response closes, so every
        # ordering these tests assert on holds; what is not exercised is
        # survival of a machine losing power, which no test can exercise
        # anyway. A test that cares about the fsync itself asks for it.
        "record": RecordConfig(fsync="never"),
        "proxy": ProxyConfig(
            host="127.0.0.1",
            port=capture_port,
            public_url=f"http://127.0.0.1:{capture_port}",
        ),
    }
    config = Config(**{**defaults, **(config_overrides or {})})

    # The production path: the application builds and starts the runtime in
    # lifespan startup, and stops it in lifespan shutdown. Nothing here builds
    # a runtime and hands it in, because nothing in production can.
    application = CaptureApplication(config)
    capture_server = LiveServer(application, capture_port, lifespan="on")
    await capture_server.start()
    runtime = application.runtime

    client = httpx.AsyncClient(timeout=30.0)
    stack = Stack(runtime, application, upstream, client, capture_server.url)
    stack.upstream_url = upstream_server.url  # type: ignore[attr-defined]
    return stack, [client, capture_server, upstream_server]


async def _teardown(resources: list[Any]) -> None:
    client, capture_server, upstream_server = resources
    await client.aclose()
    # Stopping the server runs lifespan shutdown, which stops the runtime.
    await capture_server.stop()
    await upstream_server.stop()


@pytest_asyncio.fixture
async def stack() -> AsyncIterator[Stack]:
    built, resources = await _build_stack(tokens=False)
    try:
        yield built
    finally:
        await _teardown(resources)


@pytest_asyncio.fixture
async def stack_builder():
    """Build a stack with configuration overrides, torn down automatically.

    Exposed as a fixture rather than an importable helper so no test module has
    to import from another; a cross-module ``from tests.conftest import ...``
    depends on the repository root being on ``sys.path``, which holds locally
    and not in CI.
    """
    created: list[list[Any]] = []

    async def build(*, upstream_for: Any = None, tokens: bool = False, **config_overrides: Any) -> Stack:
        built, resources = await _build_stack(
            tokens=tokens, upstream_for=upstream_for, config_overrides=config_overrides
        )
        created.append(resources)
        return built

    try:
        yield build
    finally:
        for resources in created:
            await _teardown(resources)


@pytest_asyncio.fixture
async def tokens_stack() -> AsyncIterator[Stack]:
    """A capture process configured against a token-in/token-out engine."""
    built, resources = await _build_stack(tokens=True)
    try:
        yield built
    finally:
        await _teardown(resources)


def unused(*_args: Any) -> None:
    """Silence linters for intentionally unused fixture parameters."""
    return None


# Keep the replace import referenced for callers that build variant configs.
__all__ = ["Stack", "replace", "unused"]


def record_app(root):
    """The `/v1` a record directory serves on its own: the same viewer app a
    capture replica mounts. What `skyrl-capture view --record` builds."""
    from skyrl_capture.control_plane.viewer import record_viewer_app

    return record_viewer_app(str(root))


class _NoExchanges:
    exchanges: list[Any] = []


_NO_EXCHANGES = _NoExchanges()
