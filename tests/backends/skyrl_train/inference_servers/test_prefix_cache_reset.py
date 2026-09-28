"""Check both cache-reset clients against vLLM's native HTTP route."""

import asyncio
import socket
import threading
import time
from contextlib import ExitStack
from functools import partial
from types import SimpleNamespace

import aiohttp
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

pytest.importorskip("vllm")

from vllm.entrypoints.serve.dev.cache.api_router import attach_router

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (
    VLLMServerActor,
)
from skyrl.backends.skyrl_train.weight_sync.control_plane import SkyrlWeightSyncClient

pytestmark = pytest.mark.vllm


class CacheEngine:
    def __init__(self):
        self.calls = []
        self.requests = []
        self.success = True
        self.reply = None
        self.delay = 0
        self.completed = False

    async def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        self.calls.append((reset_running_requests, reset_connector))
        await asyncio.sleep(self.delay)
        self.completed = True
        return self.success


def make_app():
    app = FastAPI()
    attach_router(app)
    VLLMServerActor._add_custom_endpoints(app, None, SimpleNamespace())

    @app.middleware("http")
    async def record_reset(request, call_next):
        engine = app.state.engine_client
        engine.requests.append((dict(request.query_params), await request.body()))
        response = await call_next(request)
        return engine.reply if engine.reply is not None else response

    return app


@pytest.fixture(scope="module")
def cache_servers():
    with ExitStack() as stack:
        backends = []
        servers = []
        threads = []
        try:
            for _ in range(2):
                app = make_app()
                sock = stack.enter_context(socket.socket())
                sock.bind(("127.0.0.1", 0))
                server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
                thread = threading.Thread(target=partial(server.run, sockets=[sock]), daemon=True)
                thread.start()
                servers.append(server)
                threads.append(thread)
                backends.append(SimpleNamespace(app=app, url=f"http://127.0.0.1:{sock.getsockname()[1]}"))
            deadline = time.monotonic() + 10
            while not all(server.started for server in servers) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert all(server.started for server in servers)
            yield backends
        finally:
            for server in servers:
                server.should_exit = True
            for thread in threads:
                thread.join(timeout=10)
                assert not thread.is_alive()


@pytest.fixture
def backends(cache_servers):
    for backend in cache_servers:
        backend.app.state.engine_client = CacheEngine()
    return cache_servers


async def reset(backends, transport, **kwargs):
    urls = [backend.url for backend in backends]
    if transport == "async":
        client = RemoteInferenceClient(proxy_url=urls[0], server_urls=urls, data_parallel_size=1)
        try:
            return await client.reset_prefix_cache(**kwargs)
        finally:
            await client.teardown()
    client = SkyrlWeightSyncClient(urls)
    try:
        return await asyncio.to_thread(client.reset_prefix_cache, **kwargs)
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["async", "sync"])
@pytest.mark.parametrize("flag", [False, True])
async def test_native_reset_query_and_result(backends, transport, flag):
    result = await reset(backends, transport, reset_running_requests=flag)
    for backend in backends:
        engine = backend.app.state.engine_client
        assert engine.calls == [(flag, False)]
        assert engine.requests == [({"reset_running_requests": str(flag).lower()}, b"")]
    if transport == "async":
        assert result == {backend.url: {"status": 200, "body": {"success": True}} for backend in backends}
    else:
        assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("transport,expected", [("async", False), ("sync", True)])
async def test_reset_defaults(backends, transport, expected):
    await reset(backends, transport)
    assert all(backend.app.state.engine_client.calls == [(expected, False)] for backend in backends)


def test_only_native_reset_route(backends):
    for backend in backends:
        routes = [route for route in backend.app.routes if getattr(route, "path", None) == "/reset_prefix_cache"]
        assert len(routes) == 1


@pytest.mark.asyncio
async def test_sync_invalid_json_names_server(backends):
    backends[0].app.state.engine_client.reply = Response(b"not-json", media_type="application/json")
    with pytest.raises(RuntimeError, match=f"{backends[0].url}: invalid JSON response") as error:
        await reset(backends, "sync", reset_running_requests=True)
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["async", "sync"])
@pytest.mark.parametrize("failed_server", [0, 1])
async def test_native_rejection_is_fatal(backends, transport, failed_server):
    backends[failed_server].app.state.engine_client.success = False
    with pytest.raises(RuntimeError, match=backends[failed_server].url):
        await reset(backends, transport, reset_running_requests=True)
    assert all(backend.app.state.engine_client.completed for backend in backends)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["async", "sync"])
@pytest.mark.parametrize("body", [{}, {"success": 1}, {"success": "true"}, None, []])
async def test_invalid_acknowledgment_is_fatal(backends, transport, body):
    backends[0].app.state.engine_client.reply = JSONResponse(body)
    with pytest.raises(RuntimeError, match=backends[0].url):
        await reset(backends, transport, reset_running_requests=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["async", "sync"])
@pytest.mark.parametrize("failure", ["http", "json", "empty"])
async def test_failed_reset_waits_for_other_servers(backends, transport, failure):
    if failure == "http":
        response = JSONResponse({"detail": "reset rejected"}, status_code=503)
        errors = (RuntimeError, aiohttp.ClientResponseError)
    else:
        response = Response(b"not-json" if failure == "json" else b"", media_type="application/json")
        errors = (ValueError, RuntimeError)
    backends[0].app.state.engine_client.reply = response
    backends[1].app.state.engine_client.delay = 0.05
    with pytest.raises(errors):
        await reset(backends, transport, reset_running_requests=True)
    assert all(backend.app.state.engine_client.completed for backend in backends)
