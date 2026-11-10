"""Control-plane requests whose connection drops before a response: idempotent endpoints are retried.

A real aiohttp server closes the connection of the first N requests without answering (what a client sees when
it reuses a pooled keep-alive connection the server already closed), then answers normally.
"""

import aiohttp
import pytest
from aiohttp import web

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    CONTROL_PLANE_DISCONNECT_RETRIES,
    IDEMPOTENT_CONTROL_ENDPOINTS,
    RemoteInferenceClient,
)


class DroppingServer:
    """Answers ``{"status": "ok"}`` on every path after dropping the first ``drops`` connections."""

    def __init__(self, drops: int):
        self.drops = drops
        self.calls = []

    async def handle(self, request: web.Request) -> web.StreamResponse:
        self.calls.append(request.path)
        if self.drops > 0:
            self.drops -= 1
            request.transport.close()  # no response: the client sees ServerDisconnectedError
            return web.Response()
        return web.json_response({"status": "ok"})


async def _start(server: DroppingServer):
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", server.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def _client(url: str) -> RemoteInferenceClient:
    return RemoteInferenceClient(proxy_url=url, server_urls=[url], model_name="test-model", data_parallel_size=1)


@pytest.mark.asyncio
async def test_resume_retried_after_disconnect():
    server = DroppingServer(drops=CONTROL_PLANE_DISCONNECT_RETRIES)
    runner, url = await _start(server)
    client = _client(url)
    try:
        result = await client.resume()
        assert result[url]["status"] == 200
        assert server.calls == ["/resume"] * (CONTROL_PLANE_DISCONNECT_RETRIES + 1)
    finally:
        await client.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_retries_are_bounded():
    server = DroppingServer(drops=CONTROL_PLANE_DISCONNECT_RETRIES + 1)
    runner, url = await _start(server)
    client = _client(url)
    try:
        with pytest.raises(aiohttp.ClientConnectionError):
            await client.resume()
        assert server.calls == ["/resume"] * (CONTROL_PLANE_DISCONNECT_RETRIES + 1)
    finally:
        await client.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_non_idempotent_endpoint_not_retried():
    assert "/fetch_weights" not in IDEMPOTENT_CONTROL_ENDPOINTS
    server = DroppingServer(drops=1)
    runner, url = await _start(server)
    client = _client(url)
    try:
        with pytest.raises(aiohttp.ClientConnectionError):
            await client._call_server(url, "/fetch_weights", {})
        assert server.calls == ["/fetch_weights"]
    finally:
        await client.aclose()
        await runner.cleanup()
