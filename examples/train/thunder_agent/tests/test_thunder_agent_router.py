"""Tests for ThunderAgentRouter."""

import asyncio
import concurrent.futures
import importlib
import threading
import time
from typing import List

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

pytest.importorskip("ThunderAgent.config", reason="ThunderAgent not installed")

BackendState = importlib.import_module("ThunderAgent.backend.state").BackendState
SGLangMetricsClient = importlib.import_module("ThunderAgent.backend.sglang_metrics").SGLangMetricsClient
get_open_port = importlib.import_module("skyrl.backends.skyrl_train.inference_servers.common").get_open_port
ThunderAgentRouter = importlib.import_module("examples.train.thunder_agent.skyrl_integration.router").ThunderAgentRouter
ThunderAgentRemoteInferenceClient = importlib.import_module(
    "examples.train.thunder_agent.skyrl_integration.remote_inference_client"
).ThunderAgentRemoteInferenceClient


def create_mock_server(server_id: int) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        await request.json()
        return JSONResponse(
            {
                "server_id": server_id,
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "choices": [
                    {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop", "index": 0}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

    @app.post("/inference/v1/generate")
    async def inference_generate(request: Request):
        payload = await request.json()
        extra_body = payload.get("extra_body", {})
        return JSONResponse(
            {
                "server_id": server_id,
                "saw_program_id": "program_id" in payload,
                "saw_extra_body_program_id": isinstance(extra_body, dict) and "program_id" in extra_body,
                "choices": [{"token_ids": [100, 200, 300], "finish_reason": "stop"}],
            }
        )

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def catch_all(path: str):
        return {"server_id": server_id, "path": f"/{path}"}

    return app


def start_server(port: int, server_id: int) -> uvicorn.Server:
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
    """Start mock servers and ThunderAgentRouter, clean up after tests."""
    servers: List[uvicorn.Server] = []

    ports = [get_open_port(), get_open_port()]
    router_port = get_open_port()
    urls = [f"http://127.0.0.1:{p}" for p in ports]

    for i, port in enumerate(ports):
        servers.append(start_server(port, server_id=i))
    for url in urls:
        assert wait_ready(url), f"Mock server at {url} failed to start"

    router = ThunderAgentRouter(
        urls,
        host="0.0.0.0",
        port=router_port,
        router_mode="default",
        backend_type="vllm",
    )
    router_url = router.start()

    yield router_url

    router.shutdown()
    for server in servers:
        server.should_exit = True
    time.sleep(0.5)


# --------------------------------------------------------------------------
# Router Parity
# --------------------------------------------------------------------------


def test_round_robin(env):
    """Catch-all requests without session distribute across servers."""
    server_ids = {httpx.get(f"{env}/test", timeout=5.0).json()["server_id"] for _ in range(4)}
    assert len(server_ids) == 2


def test_session_affinity(env):
    """Same X-Session-ID routes to same backend."""
    headers = {"X-Session-ID": "sticky-test"}
    ids = [httpx.get(f"{env}/test", headers=headers, timeout=5.0).json()["server_id"] for _ in range(3)]
    assert len(set(ids)) == 1


def test_list_servers(env):
    """/servers returns all backend URLs."""
    resp = httpx.get(f"{env}/servers", timeout=5.0)
    assert resp.status_code == 200
    assert len(resp.json()["servers"]) == 2


def test_catch_all_proxy(env):
    """Non-scheduled endpoints proxy directly to backends via catch-all."""
    resp = httpx.get(f"{env}/tokenize", timeout=5.0)
    assert resp.status_code == 200
    data = resp.json()
    assert "server_id" in data
    assert data["path"] == "/tokenize"


def test_start_shutdown_lifecycle():
    """Router starts and stops cleanly."""
    port = get_open_port()
    mock_port = get_open_port()
    mock_url = f"http://127.0.0.1:{mock_port}"

    mock_server = start_server(mock_port, server_id=99)
    assert wait_ready(mock_url)

    router = ThunderAgentRouter(
        [mock_url],
        host="0.0.0.0",
        port=port,
        router_mode="default",
        backend_type="vllm",
    )
    router_url = router.start()
    assert "http" in router_url

    resp = httpx.get(f"{router_url}/health", timeout=5.0)
    assert resp.status_code == 200

    router.shutdown()
    mock_server.should_exit = True
    time.sleep(0.5)


# --------------------------------------------------------------------------
# ThunderAgent HTTP API
# --------------------------------------------------------------------------


def test_chat_completions_proxied(env):
    """POST /v1/chat/completions routes through ThunderAgent and reaches backend."""
    resp = httpx.post(
        f"{env}/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        timeout=10.0,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert data["usage"]["total_tokens"] == 15


def test_inference_generate_tracked(env):
    """/inference/v1/generate routes through ThunderAgent program tracking."""
    resp = httpx.post(
        f"{env}/inference/v1/generate",
        json={"token_ids": [1, 2, 3], "sampling_params": {}, "model": "test", "program_id": "gen-prog-1"},
        timeout=10.0,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert data["choices"][0]["token_ids"] == [100, 200, 300]

    prog_resp = httpx.get(f"{env}/programs", timeout=5.0)
    assert prog_resp.status_code == 200
    assert "gen-prog-1" in prog_resp.json()


def test_program_id_in_body(env):
    """program_id in request body is used as the ThunderAgent program identifier."""
    httpx.post(
        f"{env}/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "program_id": "body-prog-42"},
        timeout=10.0,
    )
    prog_resp = httpx.get(f"{env}/programs", timeout=5.0)
    assert "body-prog-42" in prog_resp.json()


def test_session_id_header_fallback_used_as_program_id(env):
    """X-Session-ID is used as the program identifier when the body omits program_id."""
    session_id = "header-prog-7"
    httpx.post(
        f"{env}/inference/v1/generate",
        json={"token_ids": [1, 2, 3], "sampling_params": {}, "model": "test"},
        headers={"X-Session-ID": session_id},
        timeout=10.0,
    )
    prog_resp = httpx.get(f"{env}/programs", timeout=5.0)
    assert session_id in prog_resp.json()


def test_inference_generate_strips_program_metadata(env):
    """ThunderAgent metadata is not forwarded to the vLLM generation endpoint."""
    resp = httpx.post(
        f"{env}/inference/v1/generate",
        json={
            "token_ids": [1, 2, 3],
            "sampling_params": {},
            "model": "test",
            "program_id": "strip-prog-1",
            "extra_body": {"program_id": "strip-prog-extra"},
        },
        timeout=10.0,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["saw_program_id"] is False
    assert data["saw_extra_body_program_id"] is False


def test_programs_endpoint(env):
    """/programs returns ThunderAgent program state."""
    httpx.post(
        f"{env}/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "hello"}], "program_id": "test-prog-1"},
        timeout=10.0,
    )
    resp = httpx.get(f"{env}/programs", timeout=5.0)
    assert resp.status_code == 200
    assert "test-prog-1" in resp.json()


def test_health(env):
    """/health returns combined status with program stats."""
    resp = httpx.get(f"{env}/health", timeout=5.0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "backends" in data
    assert "programs_count" in data


def test_chat_completions_completion_updates_program_status(env):
    """Successful chat completions transition the program out of REASONING."""
    prog_id = "completion-status-test"
    httpx.post(
        f"{env}/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "program_id": prog_id},
        timeout=10.0,
    )
    prog_resp = httpx.get(f"{env}/programs", timeout=5.0)
    programs = prog_resp.json()
    assert prog_id in programs
    assert programs[prog_id]["status"] == "acting"


# --------------------------------------------------------------------------
# Weight Sync Coordination
# --------------------------------------------------------------------------


def test_weight_sync_begin_end(env):
    """POST /weight_sync/begin and /weight_sync/end toggle weight sync mode."""
    resp = httpx.post(f"{env}/weight_sync/begin", json={}, timeout=5.0)
    assert resp.status_code == 200
    assert resp.json()["weight_sync_active"] is True

    resp = httpx.post(f"{env}/weight_sync/end", json={}, timeout=5.0)
    assert resp.status_code == 200
    assert resp.json()["weight_sync_active"] is False


def test_weight_sync_idempotent_begin(env):
    """Calling begin twice is safe (idempotent)."""
    httpx.post(f"{env}/weight_sync/begin", json={}, timeout=5.0)
    resp = httpx.post(f"{env}/weight_sync/begin", json={}, timeout=5.0)
    assert resp.status_code == 200

    httpx.post(f"{env}/weight_sync/end", json={}, timeout=5.0)


def test_weight_sync_idempotent_end(env):
    """Calling end without begin is safe (idempotent)."""
    resp = httpx.post(f"{env}/weight_sync/end", json={}, timeout=5.0)
    assert resp.status_code == 200


def test_weight_sync_blocks_requests(env):
    """During weight sync, /inference/v1/generate requests are held until sync ends."""
    resp = httpx.post(f"{env}/weight_sync/begin", json={}, timeout=5.0)
    assert resp.status_code == 200

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            httpx.post,
            f"{env}/inference/v1/generate",
            json={"token_ids": [1, 2, 3], "sampling_params": {}, "model": "test", "program_id": "sync-test-1"},
            timeout=10.0,
        )

        time.sleep(0.5)
        assert not future.done(), "Request should be blocked during weight sync"

        httpx.post(f"{env}/weight_sync/end", json={}, timeout=5.0)

        result = future.result(timeout=5.0)
        assert result.status_code == 200
        assert "choices" in result.json()


@pytest.mark.asyncio
async def test_remote_client_release_program(env):
    """ThunderAgentRemoteInferenceClient can explicitly release tracked programs."""
    servers = httpx.get(f"{env}/servers", timeout=5.0).json()["servers"]
    client = ThunderAgentRemoteInferenceClient(proxy_url=env, server_urls=servers, model_name="test")
    program_id = "client-release-1"

    await client.chat_completion(
        {
            "json": {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "session_id": program_id,
            }
        }
    )
    assert program_id in httpx.get(f"{env}/programs", timeout=5.0).json()

    await client.release_program(program_id)
    assert program_id not in httpx.get(f"{env}/programs", timeout=5.0).json()
    await client.teardown()


@pytest.mark.asyncio
async def test_remote_client_pause_resume_wraps_weight_sync(env):
    """ThunderAgentRemoteInferenceClient pause/resume brackets ThunderAgent weight sync."""
    servers = httpx.get(f"{env}/servers", timeout=5.0).json()["servers"]
    client = ThunderAgentRemoteInferenceClient(proxy_url=env, server_urls=servers, model_name="test")

    await client.pause_generation()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            httpx.post,
            f"{env}/inference/v1/generate",
            json={"token_ids": [1, 2, 3], "sampling_params": {}, "model": "test"},
            headers={"X-Session-ID": "client-sync-test"},
            timeout=10.0,
        )

        time.sleep(0.5)
        assert not future.done(), "Request should be blocked while the client keeps ThunderAgent in weight sync"

        await client.resume_generation()

        result = future.result(timeout=5.0)
        assert result.status_code == 200
        assert "choices" in result.json()

    await client.teardown()


# --------------------------------------------------------------------------
# Backend URL Regressions
# --------------------------------------------------------------------------


def test_backend_state_completions_url_handles_root_and_v1_bases():
    """BackendState should not duplicate /v1 when the backend base already includes it."""
    assert BackendState("http://127.0.0.1:8000").completions_url == "http://127.0.0.1:8000/v1/chat/completions"
    assert BackendState("http://127.0.0.1:8000/v1").completions_url == "http://127.0.0.1:8000/v1/chat/completions"
    assert BackendState("http://127.0.0.1:8000/v1/").completions_url == "http://127.0.0.1:8000/v1/chat/completions"


def test_sglang_metrics_client_uses_root_level_probe_endpoints():
    """SGLang metrics and capacity probes stay at root endpoints even for /v1 backend URLs."""
    client = SGLangMetricsClient("http://127.0.0.1:8000/v1/")
    assert client.metrics_url == "http://127.0.0.1:8000/metrics"
    assert client.server_info_url == "http://127.0.0.1:8000/get_server_info"
