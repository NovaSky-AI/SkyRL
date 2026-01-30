"""Tests for InferenceRouter."""

import asyncio
import threading
import time
from typing import List

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from skyrl_train.inference_servers.common import get_open_port
from skyrl_train.inference_servers.router import InferenceRouter


def create_mock_server(server_id: int) -> FastAPI:
    app = FastAPI()

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def catch_all(path: str):
        return {"server_id": server_id, "path": f"/{path}"}

    return app


def start_server(port: int, server_id: int) -> uvicorn.Server:
    """Start a mock server, return the server instance for cleanup."""
    app = create_mock_server(server_id)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    def run():
        asyncio.run(server.serve())

    threading.Thread(target=run, daemon=True).start()
    return server


def wait_ready(url: str, timeout: float = 5.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
                return True
        except httpx.RequestError:
            time.sleep(0.1)
    return False


@pytest.fixture(scope="module")
def env():
    """Start mock servers and router, clean up after tests."""
    servers: List[uvicorn.Server] = []

    # Start mock servers
    ports = [get_open_port(), get_open_port()]
    router_port = get_open_port()
    urls = [f"http://127.0.0.1:{p}" for p in ports]

    for i, port in enumerate(ports):
        servers.append(start_server(port, server_id=i))
    for url in urls:
        assert wait_ready(url)

    # Start router using public API
    # (use 0.0.0.0 so get_node_ip() health check works)
    router = InferenceRouter(urls, host="0.0.0.0", port=router_port)
    router_url = router.start()

    yield router_url

    # Cleanup
    router.shutdown()
    for server in servers:
        server.should_exit = True
    time.sleep(0.5)


def test_round_robin(env):
    """Requests without session distribute across servers."""
    # Use /test (data plane route) instead of /health (control plane route)
    server_ids = {httpx.get(f"{env}/test").json()["server_id"] for _ in range(4)}
    assert len(server_ids) == 2


def test_session_affinity(env):
    """Same X-Session-ID routes to same server."""
    headers = {"X-Session-ID": "sticky"}
    # Use /test (data plane route) instead of /health (control plane route)
    ids = [httpx.get(f"{env}/test", headers=headers).json()["server_id"] for _ in range(3)]
    assert len(set(ids)) == 1


def test_list_servers(env):
    """/servers returns all server URLs."""
    resp = httpx.get(f"{env}/servers")
    assert resp.status_code == 200 and len(resp.json()["servers"]) == 2
