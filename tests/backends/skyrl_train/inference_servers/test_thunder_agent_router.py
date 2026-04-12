"""Tests for ThunderAgentRouter."""

import asyncio
import concurrent.futures
import importlib
import threading
import time
from types import SimpleNamespace
from typing import List

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

pytest.importorskip("examples.train.thunder_agent.thunderagent.config", reason="ThunderAgent deps missing")

program_module = importlib.import_module("examples.train.thunder_agent.thunderagent.program.state")
Program = program_module.Program
ProgramStatus = program_module.ProgramStatus
BackendState = importlib.import_module("examples.train.thunder_agent.thunderagent.backend.state").BackendState
SGLangMetricsClient = importlib.import_module(
    "examples.train.thunder_agent.thunderagent.backend.sglang_metrics"
).SGLangMetricsClient
MultiBackendRouter = importlib.import_module(
    "examples.train.thunder_agent.thunderagent.scheduler.router"
).MultiBackendRouter
get_open_port = importlib.import_module("skyrl.backends.skyrl_train.inference_servers.common").get_open_port
ThunderAgentRouter = importlib.import_module("examples.train.thunder_agent.thunder_agent_router").ThunderAgentRouter
ThunderAgentRemoteInferenceClient = importlib.import_module(
    "examples.train.thunder_agent.ta_remote_inference_client"
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
        await request.json()
        return JSONResponse(
            {
                "server_id": server_id,
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


# --------------------------------------------------------------------------
# Scheduler Internals
# --------------------------------------------------------------------------


def _build_scheduler_router(*, capacity: int):
    router = MultiBackendRouter(
        ["http://backend-0"],
        scheduling_enabled=True,
        backend_type="vllm",
    )
    backend_url = next(iter(router.backends))
    backend = router.backends[backend_url]
    backend.metrics_client.cache_config = SimpleNamespace(total_tokens_capacity=capacity)
    backend.metrics_client.healthy = True
    return router, backend_url


def _add_program(
    router,
    backend_url: str,
    *,
    program_id: str,
    total_tokens: int,
    status: ProgramStatus = ProgramStatus.REASONING,
) -> Program:
    program = Program(
        program_id=program_id,
        backend_url=backend_url,
        status=status,
    )
    program.total_tokens = total_tokens
    router.programs[program_id] = program
    router.backends[backend_url].register_program(program_id, program)
    return program


@pytest.mark.asyncio
async def test_pause_until_safe_stops_once_marked_tokens_cover_overflow():
    router, backend_url = _build_scheduler_router(capacity=1000)
    backend = router.backends[backend_url]

    smallest = _add_program(router, backend_url, program_id="prog-small", total_tokens=250)
    middle = _add_program(router, backend_url, program_id="prog-mid", total_tokens=300)
    largest = _add_program(router, backend_url, program_id="prog-large", total_tokens=350)

    assert backend.remaining_capacity() == -200

    await router._pause_until_safe(backend)

    assert smallest.marked_for_pause is True
    assert middle.marked_for_pause is False
    assert largest.marked_for_pause is False
    assert backend.future_paused_tokens == 250
    assert backend.marked_for_pause_count == 1
    assert backend.remaining_capacity(include_future_release=True) >= 0


@pytest.mark.asyncio
async def test_pause_until_safe_counts_future_buffer_release_for_zero_token_programs():
    router, backend_url = _build_scheduler_router(capacity=250)
    backend = router.backends[backend_url]

    first = _add_program(router, backend_url, program_id="prog-a", total_tokens=0)
    second = _add_program(router, backend_url, program_id="prog-b", total_tokens=0)
    third = _add_program(router, backend_url, program_id="prog-c", total_tokens=0)

    assert backend.remaining_capacity() == -50

    await router._pause_until_safe(backend)

    assert first.marked_for_pause is True
    assert second.marked_for_pause is False
    assert third.marked_for_pause is False
    assert backend.future_paused_tokens == 0
    assert backend.marked_for_pause_count == 1
    assert backend.remaining_capacity(include_future_release=True) >= 0
