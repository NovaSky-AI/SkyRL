"""Minimal smoke tests for ThunderAgentRouter."""

import asyncio
import threading
import time
from typing import Dict, List

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

pytest.importorskip("ThunderAgent.config", reason="ThunderAgent package not installed")

from examples.train.thunder_agent.thunder_agent_router import ThunderAgentRouter
from skyrl.backends.skyrl_train.inference_servers.common import get_open_port
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient


@pytest.fixture(scope="session", autouse=True)
def ray_init():
    """Override the backend test suite's global Ray fixture for pure HTTP smoke tests."""
    yield


def create_mock_server(server_id: int) -> FastAPI:
    """Create a minimal vLLM-like backend for router smoke tests."""
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok", "server_id": server_id}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(
            'vllm:cache_config_info{block_size="16",num_gpu_blocks="1024"} 1.0\n'
            "vllm:num_requests_running{} 0\n"
            "vllm:num_requests_waiting{} 0\n"
            "vllm:kv_cache_usage_perc{} 0\n"
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        await request.json()
        return JSONResponse(
            {
                "server_id": server_id,
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

    @app.post("/v1/completions")
    async def completions(request: Request):
        await request.json()
        return JSONResponse(
            {
                "server_id": server_id,
                "id": "cmpl-mock",
                "object": "text_completion",
                "choices": [{"text": "hello", "finish_reason": "stop", "index": 0}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
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

    @app.post("/tokenize")
    async def tokenize(request: Request):
        payload = await request.json()
        prompt = payload.get("prompt", "")
        return JSONResponse({"server_id": server_id, "tokens": [len(prompt)]})

    @app.post("/detokenize")
    async def detokenize(request: Request):
        payload = await request.json()
        tokens = payload.get("tokens", [])
        return JSONResponse({"server_id": server_id, "prompt": ",".join(str(tok) for tok in tokens)})

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
    """Start mock backends and ThunderAgentRouter, then clean up."""
    servers: List[uvicorn.Server] = []
    backend_ports = [get_open_port(), get_open_port()]
    backend_urls = [f"http://127.0.0.1:{port}" for port in backend_ports]
    router_port = get_open_port()

    for server_id, port in enumerate(backend_ports):
        servers.append(start_server(port, server_id=server_id))
    for url in backend_urls:
        assert wait_ready(url), f"Mock server at {url} failed to start"

    router = ThunderAgentRouter(
        backend_urls,
        host="0.0.0.0",
        port=router_port,
        router_mode="default",
        backend_type="vllm",
    )
    router_url = router.start()

    yield {"router": router, "router_url": router_url, "backend_urls": backend_urls}

    router.shutdown()
    for server in servers:
        server.should_exit = True
    time.sleep(0.5)


def _load_programs(router_url: str) -> Dict[str, Dict]:
    response = httpx.get(f"{router_url}/programs", timeout=5.0)
    assert response.status_code == 200
    return response.json()


def test_router_lifecycle_and_health(env):
    response = httpx.get(f"{env['router_url']}/health", timeout=5.0)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert len(payload["backends"]) == 2


def test_chat_completion_tracks_program(env):
    async def run():
        client = RemoteInferenceClient(
            proxy_url=env["router_url"],
            server_urls=env["backend_urls"],
            model_name="test-model",
        )
        try:
            response = await client.chat_completion(
                {
                    "json": {
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "session_id": "chat-prog-1",
                    }
                }
            )
            assert response["usage"]["total_tokens"] == 15
        finally:
            await client.teardown()

    asyncio.run(run())
    programs = _load_programs(env["router_url"])
    assert "chat-prog-1" in programs


def test_completion_tracks_program(env):
    async def run():
        client = RemoteInferenceClient(
            proxy_url=env["router_url"],
            server_urls=env["backend_urls"],
            model_name="test-model",
        )
        try:
            response = await client.completion(
                {
                    "json": {
                        "model": "test-model",
                        "prompt": "hello",
                        "session_id": "completion-prog-1",
                    }
                }
            )
            assert response["usage"]["total_tokens"] == 12
        finally:
            await client.teardown()

    asyncio.run(run())
    programs = _load_programs(env["router_url"])
    assert "completion-prog-1" in programs


def test_generate_tracks_program_and_detokenizes(env):
    async def run():
        client = RemoteInferenceClient(
            proxy_url=env["router_url"],
            server_urls=env["backend_urls"],
            model_name="test-model",
        )
        try:
            output = await client.generate(
                {
                    "prompt_token_ids": [[1, 2, 3]],
                    "sampling_params": {},
                    "session_ids": ["generate-prog-1"],
                }
            )
            assert output["response_ids"][0] == [100, 200, 300]
            assert output["responses"][0] == "100,200,300"
        finally:
            await client.teardown()

    asyncio.run(run())
    programs = _load_programs(env["router_url"])
    assert "generate-prog-1" in programs


def test_tokenize_and_detokenize_proxy(env):
    tokenize_response = httpx.post(
        f"{env['router_url']}/tokenize",
        json={"model": "test-model", "prompt": "abcd", "add_special_tokens": True},
        timeout=5.0,
    )
    assert tokenize_response.status_code == 200
    assert tokenize_response.json()["tokens"] == [4]

    detokenize_response = httpx.post(
        f"{env['router_url']}/detokenize",
        json={"model": "test-model", "tokens": [7, 8]},
        timeout=5.0,
    )
    assert detokenize_response.status_code == 200
    assert detokenize_response.json()["prompt"] == "7,8"
