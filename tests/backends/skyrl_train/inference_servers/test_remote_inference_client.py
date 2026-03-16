"""Tests for RemoteInferenceClient."""

import asyncio
import pickle
import threading
import time
from typing import List

import httpx
import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Request

from skyrl.backends.skyrl_train.inference_servers.common import get_open_port
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    PauseMode,
    RemoteInferenceClient,
)


def create_mock_vllm_server(server_id: int) -> FastAPI:
    """Create a mock vLLM server with standard endpoints."""
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/get_world_size")
    async def get_world_size():
        return {"world_size": 2}  # Simulate TP=2

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        prompts = body.get("prompt", [])
        n_prompts = len(prompts) if isinstance(prompts, list) else 1
        return {
            "choices": [
                {"index": i, "text": f"Response {i} from server {server_id}", "finish_reason": "stop"}
                for i in range(n_prompts)
            ]
        }

    @app.post("/inference/v1/generate")
    async def generate(request: Request):
        body = await request.json()  # Consume body
        num_prompts = len(body.get("token_ids", []))

        return {
            "choices": [
                {"request_id": "dummy", "token_ids": [i, i + 1, i + 2], "finish_reason": "stop"}
                for i in range(num_prompts)
            ]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return {"choices": [{"message": {"content": f"Chat from server {server_id}"}}]}

    @app.post("/tokenize")
    async def tokenize(request: Request):
        return {"tokens": [1, 2, 3]}

    @app.post("/detokenize")
    async def detokenize(request: Request):
        return {"prompt": "hello world"}

    # Control plane endpoints
    @app.post("/pause")
    async def pause(request: Request, mode: str = "abort", clear_cache: str = "true"):
        return {"status": "paused", "server_id": server_id, "mode": mode, "clear_cache": clear_cache}

    @app.post("/resume")
    async def resume():
        return {"status": "resumed", "server_id": server_id}

    @app.get("/is_paused")
    async def is_paused():
        # Mock always returns not paused for basic tests
        return {"is_paused": False}

    @app.post("/sleep")
    async def sleep(request: Request):
        return {"status": "sleeping", "server_id": server_id}

    @app.post("/wake_up")
    async def wake_up():
        return {"status": "awake", "server_id": server_id}

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache(request: Request):
        return {"status": "cache_reset", "server_id": server_id}

    @app.post("/init_weight_transfer_engine")
    async def init_weight_transfer_engine(request: Request):
        return {"status": "ok", "server_id": server_id}

    @app.post("/update_weights")
    async def update_weights(request: Request):
        return {"status": "ok", "server_id": server_id}

    return app


def start_server(port: int, server_id: int) -> uvicorn.Server:
    """Start a mock server, return the server instance."""
    app = create_mock_vllm_server(server_id)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="error")
    server = uvicorn.Server(config)

    def run():
        asyncio.run(server.serve())

    threading.Thread(target=run, daemon=True).start()
    return server


def wait_ready(url: str, timeout: float = 5.0) -> bool:
    """Wait for server to become healthy."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
                return True
        except httpx.RequestError:
            time.sleep(0.1)
    return False


@pytest.fixture(scope="module")
def mock_servers():
    """Start mock vLLM servers, return proxy_url and server_urls."""
    servers: List[uvicorn.Server] = []
    ports = [get_open_port(), get_open_port()]
    server_urls = [f"http://127.0.0.1:{p}" for p in ports]

    for i, port in enumerate(ports):
        servers.append(start_server(port, server_id=i))

    for url in server_urls:
        assert wait_ready(url), f"Server {url} failed to start"

    # proxy_url defaults to first server; can be replaced with router URL later
    yield {"proxy_url": server_urls[0], "server_urls": server_urls}

    # Cleanup
    for server in servers:
        server.should_exit = True
    time.sleep(0.3)


@pytest_asyncio.fixture
async def client(mock_servers):
    """Create a RemoteInferenceClient for data/control plane tests."""
    client = RemoteInferenceClient(
        proxy_url=mock_servers["proxy_url"],
        server_urls=mock_servers["server_urls"],
    )
    yield client
    await client.teardown()


class TestRemoteInferenceClientInit:
    """Test client initialization and serialization."""

    def test_serialization(self, mock_servers):
        """Client can be pickled and unpickled."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="test-model",
        )

        # Pickle and unpickle
        pickled = pickle.dumps(client)
        restored = pickle.loads(pickled)

        assert restored.proxy_url == client.proxy_url
        assert restored.server_urls == client.server_urls
        assert restored.model_name == client.model_name
        # Session should be None after unpickling
        assert restored._session is None


class TestDataPlane:
    """Test data plane methods."""

    @pytest.mark.asyncio
    async def test_generate(self, client):
        """Test generate method."""
        input_batch = {
            "prompt_token_ids": [[1, 2, 3], [4, 5, 6]],
            "sampling_params": {"max_tokens": 100},
        }
        result = await client.generate(input_batch)

        assert "responses" in result
        assert "stop_reasons" in result
        assert len(result["responses"]) == 2
        assert all(r == "stop" for r in result["stop_reasons"])
        # response_ids are tokenized from the response
        assert len(result["response_ids"]) == 2

    @pytest.mark.asyncio
    async def test_generate_with_session_id(self, client):
        """Test generate with session ID for consistent routing."""
        input_batch = {
            "prompt_token_ids": [[1, 2, 3]],
            "session_ids": ["test-session"],
        }
        result = await client.generate(input_batch)
        assert len(result["responses"]) == 1

    @pytest.mark.asyncio
    async def test_chat_completion(self, client):
        """Test chat completion method."""
        request_payload = {
            "json": {
                "model": "test",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            "headers": {},
        }
        result = await client.chat_completion(request_payload)
        assert "choices" in result

    @pytest.mark.asyncio
    async def test_completion(self, client):
        """Test completion method."""
        request_payload = {
            "json": {"model": "test", "prompt": "Hello"},
            "headers": {},
        }
        result = await client.completion(request_payload)
        assert "choices" in result

    @pytest.mark.asyncio
    async def test_tokenize(self, client):
        """Test tokenize method."""
        result = await client.tokenize(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [1, 2, 3]  # Mock response

    @pytest.mark.asyncio
    async def test_detokenize(self, client):
        """Test detokenize method."""
        result = await client.detokenize([[1, 2, 3], [4, 5, 6]])
        assert len(result) == 2
        assert result[0] == "hello world"  # Mock response


class TestControlPlane:
    """Test control plane methods (fan-out to all servers)."""

    @pytest.mark.asyncio
    async def test_pause_keep_mode(self, client):
        """Test pause with KEEP mode (default) sends mode=keep and clear_cache=false."""
        result = await client.pause(mode=PauseMode.KEEP)
        assert len(result) == 2
        for url, response in result.items():
            assert response["status"] == 200
            assert response["body"]["status"] == "paused"
            assert response["body"]["mode"] == "keep"
            assert response["body"]["clear_cache"] == "false"

    @pytest.mark.asyncio
    async def test_pause_abort_mode(self, client):
        """Test pause with ABORT mode fans out to all servers with mode=abort."""
        result = await client.pause(mode=PauseMode.ABORT)
        assert len(result) == 2
        for url, response in result.items():
            assert response["status"] == 200
            assert response["body"]["status"] == "paused"
            assert response["body"]["mode"] == "abort"

    @pytest.mark.asyncio
    async def test_pause_wait_mode(self, client):
        """Test pause with WAIT mode fans out to all servers with mode=wait."""
        result = await client.pause(mode=PauseMode.WAIT)
        assert len(result) == 2
        for url, response in result.items():
            assert response["status"] == 200
            assert response["body"]["mode"] == "wait"

    @pytest.mark.asyncio
    async def test_pause_generation_uses_keep_mode(self, client):
        """Test that pause_generation() alias uses KEEP mode."""
        result = await client.pause_generation()
        assert len(result) == 2
        for url, response in result.items():
            assert response["status"] == 200
            assert response["body"]["mode"] == "keep"
            assert response["body"]["clear_cache"] == "false"

    @pytest.mark.asyncio
    async def test_resume(self, client):
        """Test resume fans out to all servers."""
        await client.pause()

        result = await client.resume()
        assert len(result) == 2
        for url, response in result.items():
            assert response["status"] == 200

    @pytest.mark.asyncio
    async def test_sleep(self, client):
        """Test sleep fans out to all servers."""
        result = await client.sleep(level=2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_wake_up(self, client):
        """Test wake_up fans out to all servers."""
        result = await client.wake_up()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_reset_prefix_cache(self, client):
        """Test reset_prefix_cache fans out to all servers."""
        result = await client.reset_prefix_cache()
        assert len(result) == 2


class TestWeightSync:
    """Test weight sync methods."""

    @pytest.mark.asyncio
    async def test_init_weight_update_communicator(self, client):
        """Test init_weight_update_communicator expands init_info and fans out to all servers."""

        class MockInitInfo:
            """Lightweight mock satisfying the for_servers / to_api_payload protocol."""

            def for_servers(self, world_size_per_server, num_servers):
                return [self] * num_servers

            def to_api_payload(self):
                return {"master_address": "127.0.0.1", "master_port": 29500, "rank_offset": 1, "world_size": 5}

        result = await client.init_weight_update_communicator(MockInitInfo())
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_update_named_weights(self, client):
        """Test update_weights fans out to all servers."""
        update_info = {
            "names": ["layer.weight"],
            "dtype_names": ["bfloat16"],
            "shapes": [[1024, 1024]],
            "packed": True,
        }
        result = await client.update_named_weights(update_info)
        assert len(result) == 2


class TestServerInfo:
    """Test server info and world_size."""

    @pytest.mark.asyncio
    async def test_get_world_size(self, client):
        """Test world_size fetching and caching."""
        # First call fetches from all servers and sums
        total_world_size, world_size_per_server = await client.get_world_size()
        # Each mock server reports world_size=2, we have 2 servers = 4
        assert total_world_size == 4
        assert world_size_per_server == 2

        # Second call returns cached value
        total_world_size2, _ = await client.get_world_size()
        assert total_world_size2 == 4


class TestContextManager:
    """Test async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_servers):
        """Test using client as async context manager."""

        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
        )

        async with client:
            result = await client.resume()
            assert len(result) == 2

        # Session should be closed after exiting context
        assert client._session is None or client._session.closed
