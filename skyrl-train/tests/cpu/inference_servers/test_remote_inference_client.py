"""Tests for RemoteInferenceClient."""

import asyncio
import pickle
import threading
import time
from typing import Any, Dict, List

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request

from skyrl_train.inference_servers.common import get_open_port
from skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient, PauseMode


def create_mock_vllm_server(server_id: int) -> FastAPI:
    """Create a mock vLLM server with standard endpoints."""
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/get_server_info")
    async def get_server_info():
        return {
            "ip": "127.0.0.1",
            "port": 8000 + server_id,
            "url": f"http://127.0.0.1:{8000 + server_id}",
            "server_idx": server_id,
            "world_size": 2,  # Simulate TP=2
        }

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
    async def pause(request: Request):
        return {"status": "paused", "server_id": server_id}

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

    @app.post("/init_weight_transfer")
    async def init_weight_transfer(request: Request):
        return {"status": "ok", "server_id": server_id}

    @app.post("/update_weights")
    async def update_weights(request: Request):
        return {"status": "ok", "server_id": server_id}

    @app.post("/finalize_weight_update")
    async def finalize_weight_update(request: Request):
        return {"status": "ok", "server_id": server_id}

    return app


def start_server(port: int, server_id: int) -> uvicorn.Server:
    """Start a mock server, return the server instance."""
    app = create_mock_vllm_server(server_id)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
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
    """Start mock vLLM servers."""
    servers: List[uvicorn.Server] = []
    ports = [get_open_port(), get_open_port()]
    urls = [f"http://127.0.0.1:{p}" for p in ports]

    for i, port in enumerate(ports):
        servers.append(start_server(port, server_id=i))

    for url in urls:
        assert wait_ready(url), f"Server {url} failed to start"

    yield urls

    # Cleanup
    for server in servers:
        server.should_exit = True
    time.sleep(0.3)


class TestRemoteInferenceClientInit:
    """Test client initialization and serialization."""

    def test_init(self, mock_servers):
        """Client initializes with proxy_url and server_urls."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )
        assert client.proxy_url == mock_servers[0]
        assert client.server_urls == mock_servers

    def test_serialization(self, mock_servers):
        """Client can be pickled and unpickled."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
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

    def test_from_server_group(self, mock_servers):
        """Test factory method from_server_group."""
        client = RemoteInferenceClient.from_server_group(
            server_urls=mock_servers,
            model_name="test-model",
        )
        assert client.proxy_url == mock_servers[0]  # Defaults to first server
        assert client.server_urls == mock_servers

    def test_from_server_group_with_router(self, mock_servers):
        """Test factory method from_server_group with router URL."""
        router_url = "http://router:8080"
        client = RemoteInferenceClient.from_server_group(
            server_urls=mock_servers,
            router_url=router_url,
            model_name="test-model",
        )
        assert client.proxy_url == router_url
        assert client.server_urls == mock_servers

    def test_from_router(self, mock_servers):
        """Test factory method from_router."""
        router_url = mock_servers[0]
        client = RemoteInferenceClient.from_router(
            router_url=router_url,
            model_name="test-model",
        )
        assert client.proxy_url == router_url
        assert client.server_urls == [router_url]


class TestDataPlane:
    """Test data plane methods."""

    @pytest.mark.asyncio
    async def test_generate(self, mock_servers):
        """Test generate method."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            input_batch = {
                "prompt_token_ids": [[1, 2, 3], [4, 5, 6]],
                "sampling_params": {"max_tokens": 100},
            }
            result = await client.generate(input_batch)

            assert "responses" in result
            assert "stop_reasons" in result
            assert len(result["responses"]) == 2
            assert all(r == "stop" for r in result["stop_reasons"])
            # response_ids are empty (use tokenize() if needed)
            assert all(len(ids) == 0 for ids in result["response_ids"])
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_generate_with_session_id(self, mock_servers):
        """Test generate with session ID for consistent routing."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            input_batch = {
                "prompt_token_ids": [[1, 2, 3]],
                "session_ids": ["test-session"],
            }
            result = await client.generate(input_batch)
            assert len(result["responses"]) == 1
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_chat_completion(self, mock_servers):
        """Test chat completion method."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            request_payload = {
                "json": {
                    "model": "test",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                "headers": {},
            }
            result = await client.chat_completion(request_payload)
            assert "choices" in result
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_completion(self, mock_servers):
        """Test completion method."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            request_payload = {
                "json": {"model": "test", "prompt": "Hello"},
                "headers": {},
            }
            result = await client.completion(request_payload)
            assert "choices" in result
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_tokenize(self, mock_servers):
        """Test tokenize method."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            result = await client.tokenize(["hello", "world"])
            assert len(result) == 2
            assert result[0] == [1, 2, 3]  # Mock response
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_detokenize(self, mock_servers):
        """Test detokenize method."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            result = await client.detokenize([[1, 2, 3], [4, 5, 6]])
            assert len(result) == 2
            assert result[0] == "hello world"  # Mock response
        finally:
            await client.teardown()


class TestControlPlane:
    """Test control plane methods (fan-out to all servers)."""

    @pytest.mark.asyncio
    async def test_pause_abort_mode(self, mock_servers):
        """Test pause with ABORT mode (default) fans out to all servers."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            result = await client.pause(mode=PauseMode.ABORT)
            assert len(result) == 2
            for url, response in result.items():
                assert response["status"] == 200
                assert response["body"]["status"] == "paused"
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_pause_finish_mode(self, mock_servers):
        """Test pause with FINISH mode fans out to all servers."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            result = await client.pause(mode=PauseMode.FINISH)
            assert len(result) == 2
            for url, response in result.items():
                assert response["status"] == 200
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_pause_keep_mode_not_supported(self, mock_servers):
        """Test pause with KEEP mode raises NotImplementedError."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            with pytest.raises(NotImplementedError, match="KEEP is not yet supported"):
                await client.pause(mode=PauseMode.KEEP)
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_resume(self, mock_servers):
        """Test resume fans out to all servers."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            # Pause first
            await client.pause()

            # Resume
            result = await client.resume()
            assert len(result) == 2
            for url, response in result.items():
                assert response["status"] == 200
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_sleep(self, mock_servers):
        """Test sleep fans out to all servers."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            result = await client.sleep(level=2)
            assert len(result) == 2
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_wake_up(self, mock_servers):
        """Test wake_up fans out to all servers."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            result = await client.wake_up()
            assert len(result) == 2
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_reset_prefix_cache(self, mock_servers):
        """Test reset_prefix_cache fans out to all servers."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            result = await client.reset_prefix_cache()
            assert len(result) == 2
        finally:
            await client.teardown()


class TestWeightSync:
    """Test weight sync methods."""

    @pytest.mark.asyncio
    async def test_init_weight_transfer(self, mock_servers):
        """Test init_weight_transfer fans out to all servers."""
        from skyrl_train.weight_sync import BroadcastInitInfo

        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            init_info = BroadcastInitInfo(
                master_addr="127.0.0.1",
                master_port=29500,
                rank_offset=1,
                world_size=5,
                group_name="test",
                backend="nccl",
                model_dtype_str="torch.bfloat16",
            )
            result = await client.init_weight_transfer(init_info)
            assert len(result) == 2
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_update_weights(self, mock_servers):
        """Test update_weights fans out to all servers."""
        from skyrl_train.weight_sync import BroadcastWeightUpdateRequest

        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            request = BroadcastWeightUpdateRequest(
                names=["layer.weight"],
                dtypes=["torch.bfloat16"],
                shapes=[[1024, 1024]],
            )
            result = await client.update_weights(request)
            assert len(result) == 2
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_finalize_weight_update(self, mock_servers):
        """Test finalize_weight_update fans out to all servers."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            result = await client.finalize_weight_update()
            assert len(result) == 2
        finally:
            await client.teardown()


class TestServerInfo:
    """Test server info and world_size."""

    @pytest.mark.asyncio
    async def test_get_world_size(self, mock_servers):
        """Test world_size fetching and caching."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        )

        try:
            # First call fetches from all servers and sums
            world_size = await client.get_world_size()
            # Each mock server reports world_size=2, we have 2 servers = 4
            assert world_size == 4

            # Second call returns cached value
            world_size2 = await client.get_world_size()
            assert world_size2 == 4
        finally:
            await client.teardown()


class TestContextManager:
    """Test async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_servers):
        """Test using client as async context manager."""
        async with RemoteInferenceClient(
            proxy_url=mock_servers[0],
            server_urls=mock_servers,
        ) as client:
            result = await client.resume()
            assert len(result) == 2

        # Session should be closed after exiting context
        assert client._session is None or client._session.closed


class TestRetryOnAbort:
    """Test retry on abort functionality."""

    @pytest.fixture
    def abort_mock_server(self):
        """Create a mock server that returns abort on first call, then stop."""
        app = FastAPI()
        call_count = {"completions": 0}

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.post("/v1/completions")
        async def completions(request: Request):
            call_count["completions"] += 1
            body = await request.json()

            # First call returns abort with partial response
            if call_count["completions"] == 1:
                return {
                    "choices": [
                        {"index": 0, "text": "Partial ", "finish_reason": "abort"}
                    ]
                }
            # Second call returns complete response
            else:
                return {
                    "choices": [
                        {"index": 0, "text": "response complete", "finish_reason": "stop"}
                    ]
                }

        @app.post("/tokenize")
        async def tokenize(request: Request):
            body = await request.json()
            prompt = body.get("prompt", "")
            # Simple tokenization: one token per word
            tokens = [hash(word) % 10000 for word in prompt.split()]
            return {"tokens": tokens}

        @app.get("/get_server_info")
        async def get_server_info():
            return {"world_size": 1}

        @app.get("/is_paused")
        async def is_paused():
            # Not paused - allows retry to proceed immediately
            return {"is_paused": False}

        # Start server in background thread
        port = get_open_port()
        config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Wait for server to be ready
        for _ in range(100):
            try:
                httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.1)
                break
            except Exception:
                time.sleep(0.05)

        yield f"http://127.0.0.1:{port}", call_count

        server.should_exit = True
        thread.join(timeout=1)

    @pytest.mark.asyncio
    async def test_retry_on_abort(self, abort_mock_server):
        """Test that retry on abort is always active (built-in behavior)."""
        url, call_count = abort_mock_server
        client = RemoteInferenceClient(
            proxy_url=url,
            server_urls=[url],
        )

        try:
            result = await client.generate({
                "prompt_token_ids": [[1, 2, 3]],
                "sampling_params": {"max_tokens": 100},
            })

            # Should get complete response after retry
            assert result["stop_reasons"][0] == "stop"
            assert result["responses"][0] == "Partial response complete"
            assert call_count["completions"] == 2
            # Should have response_ids from tokenization
            assert len(result["response_ids"][0]) > 0
        finally:
            await client.teardown()
