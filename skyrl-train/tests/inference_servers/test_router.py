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

    # Start router
    router = InferenceRouter(urls, host="127.0.0.1", port=router_port)
    router._client = httpx.AsyncClient(timeout=httpx.Timeout(None))
    router._app = router._build_app()

    router_config = uvicorn.Config(
        router._app, host="127.0.0.1", port=router_port, log_level="error"
    )
    router_server = uvicorn.Server(router_config)
    servers.append(router_server)

    def run_router():
        asyncio.run(router_server.serve())

    threading.Thread(target=run_router, daemon=True).start()

    router_url = f"http://127.0.0.1:{router_port}"
    assert wait_ready(router_url)

    yield router_url

    # Cleanup: signal all servers to shutdown
    for server in servers:
        server.should_exit = True
    time.sleep(0.5)  # Give servers time to shutdown


def test_round_robin(env):
    """Requests without session distribute across servers."""
    server_ids = {httpx.get(f"{env}/health").json()["server_id"] for _ in range(4)}
    assert len(server_ids) == 2


def test_session_affinity(env):
    """Same X-Session-ID routes to same server."""
    headers = {"X-Session-ID": "sticky"}
    ids = [httpx.get(f"{env}/health", headers=headers).json()["server_id"] for _ in range(3)]
    assert len(set(ids)) == 1


def test_control_plane_fanout(env):
    """Control plane routes fan out to all servers."""
    resp = httpx.post(f"{env}/sleep", json={})
    assert resp.status_code == 200 and resp.json()["status"] == "ok"


def test_list_servers(env):
    """/servers returns all server URLs."""
    resp = httpx.get(f"{env}/servers")
    assert resp.status_code == 200 and len(resp.json()["servers"]) == 2
