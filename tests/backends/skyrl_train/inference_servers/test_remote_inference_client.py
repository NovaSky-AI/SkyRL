"""Tests for RemoteInferenceClient."""

import asyncio
import pickle
import threading
import time
from typing import List, Optional

import httpx
import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Query, Request

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

    @app.post("/skyrl/v1/generate")
    @app.post("/inference/v1/generate")
    async def generate(request: Request):
        body = await request.json()  # Consume body
        sp = body.get("sampling_params", {})
        input_token_ids = body.get("token_ids", [])
        n = sp.get("n", 1)
        # If logprobs is explicitly set (sample path), use n for num_choices.
        # Otherwise (generate path), use len(token_ids) for per-prompt responses.
        if "logprobs" in sp:
            num_choices = n
        else:
            num_choices = 1

        response: dict = {
            "choices": [
                {
                    "request_id": "dummy",
                    "token_ids": [i, i + 1, i + 2],
                    "finish_reason": "stop",
                    "logprobs": {"content": [{"logprob": -0.1 * (i + 1)}]},
                }
                for i in range(num_choices)
            ]
        }

        # Mock prompt_logprobs when requested via sampling_params
        pl = sp.get("prompt_logprobs")
        # vLLM returns k or k+1 logprobs per position (extra entry when
        # the prompt token falls outside the top-k).
        if pl is not None and input_token_ids:
            prompt_logprobs = [None]  # position 0: no prior context
            for idx in range(1, len(input_token_ids)):
                position_dict = {
                    str(input_token_ids[idx]): {
                        "logprob": -0.5 * idx,
                        "rank": 1,
                        "decoded_token": None,
                    }
                }
                # If topk > 0, add extra entries
                if pl > 0:
                    for extra in range(pl):
                        fake_token_id = 9000 + idx * 10 + extra
                        position_dict[str(fake_token_id)] = {
                            "logprob": -1.0 * idx - 0.1 * extra,
                            "rank": extra + 2,
                            "decoded_token": None,
                        }
                prompt_logprobs.append(position_dict)
            response["prompt_logprobs"] = prompt_logprobs

        return response

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return {"choices": [{"message": {"content": f"Chat from server {server_id}"}}]}

    @app.post("/v1/chat/completions/render")
    async def render_chat_completion(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        has_multimodal = any(
            isinstance(msg.get("content"), list) and any(c.get("type") == "image_url" for c in msg["content"])
            for msg in messages
        )
        features = None
        if has_multimodal:
            features = {
                "mm_hashes": {"image": ["fd2700fd096d55f04c20dbfdc903f7e65421e282"]},
                "mm_placeholders": {"image": [{"offset": 4, "length": 100}]},
            }
        return {
            "request_id": f"chatcmpl-mock-{server_id}",
            "token_ids": list(range(110)) if has_multimodal else list(range(10)),
            "features": features,
            "sampling_params": {"temperature": 0.7, "max_tokens": 100},
            "model": body.get("model", "test"),
            "stream": body.get("stream", False),
            "stream_options": body.get("stream_options"),
            "cache_salt": None,
            "priority": 0,
            "kv_transfer_params": None,
        }

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
    async def sleep(level: int = 2, tags: Optional[List[str]] = Query(None)):
        return {"status": "sleeping", "server_id": server_id, "level": level, "tags": tags}

    @app.post("/wake_up")
    async def wake_up(tags: Optional[List[str]] = Query(None)):
        return {"status": "awake", "server_id": server_id, "tags": tags}

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
        for url, response in result.items():
            assert response["body"]["level"] == 2
            assert response["body"]["tags"] is None

    @pytest.mark.asyncio
    async def test_sleep_with_tags(self, client):
        """Test sleep with tags produces correct repeated query params."""
        result = await client.sleep(level=1, tags=["weights", "kv_cache"])
        assert len(result) == 2
        for url, response in result.items():
            assert response["body"]["level"] == 1
            assert response["body"]["tags"] == ["weights", "kv_cache"]

    @pytest.mark.asyncio
    async def test_wake_up(self, client):
        """Test wake_up fans out to all servers."""
        result = await client.wake_up()
        assert len(result) == 2
        for url, response in result.items():
            assert response["body"]["tags"] is None

    @pytest.mark.asyncio
    async def test_wake_up_with_tags(self, client):
        """Test wake_up with tags produces correct repeated query params."""
        result = await client.wake_up(tags=["weights"])
        assert len(result) == 2
        for url, response in result.items():
            assert response["body"]["tags"] == ["weights"]

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


class TestSample:
    """Test sample() method (Tinker API)."""

    @pytest.mark.asyncio
    async def test_sample(self, client):
        """Test sample with n=1 returns correct structure and prompt_logprobs."""
        request_payload = {
            "json": {
                "prompt": {"chunks": [{"tokens": [10, 20, 30]}]},
                "num_samples": 1,
                "sampling_params": {"temperature": 0.7, "max_tokens": 64},
                "include_prompt_logprobs": True,
            }
        }
        result = await client.sample(request_payload)

        assert result["type"] == "sample"
        assert len(result["sequences"]) == 1

        seq = result["sequences"][0]
        assert seq["tokens"] == [0, 1, 2]
        assert seq["logprobs"] == [-0.1]
        assert seq["stop_reason"] == "stop"

        # prompt_logprobs: one float per prompt token, position 0 is None
        pl = result["prompt_logprobs"]
        assert pl is not None
        assert len(pl) == 3
        assert pl[0] is None
        assert pl[1] == pytest.approx(-0.5)
        assert pl[2] == pytest.approx(-1.0)
        # topk not requested
        assert result["topk_prompt_logprobs"] is None

    @pytest.mark.asyncio
    async def test_sample_n2(self, client):
        """Test sample with n=2 returns two sequences and prompt_logprobs."""
        request_payload = {
            "json": {
                "prompt": {"chunks": [{"tokens": [1, 2]}, {"tokens": [3]}]},
                "num_samples": 2,
                "sampling_params": {"temperature": 1.0, "max_tokens": 32},
                "include_prompt_logprobs": True,
            }
        }
        result = await client.sample(request_payload)

        assert len(result["sequences"]) == 2
        assert result["sequences"][0]["tokens"] == [0, 1, 2]
        assert result["sequences"][1]["tokens"] == [1, 2, 3]
        assert result["sequences"][0]["logprobs"] == [-0.1]
        assert result["sequences"][1]["logprobs"] == [-0.2]

        # prompt_logprobs shared across choices
        pl = result["prompt_logprobs"]
        assert pl is not None
        assert len(pl) == 3
        assert pl[0] is None

    @pytest.mark.asyncio
    async def test_sample_topk_prompt_logprobs(self, client):
        """Test topk_prompt_logprobs returns both prompt_logprobs and topk tuples."""
        request_payload = {
            "json": {
                "prompt": {"chunks": [{"tokens": [10, 20, 30]}]},
                "num_samples": 1,
                "sampling_params": {"temperature": 0.7, "max_tokens": 64},
                "include_prompt_logprobs": True,
                "topk_prompt_logprobs": 2,
            }
        }
        result = await client.sample(request_payload)

        pl = result["prompt_logprobs"]
        assert pl is not None
        assert len(pl) == 3
        assert pl[0] is None
        assert pl[1] == pytest.approx(-0.5)
        assert pl[2] == pytest.approx(-1.0)

        topk = result["topk_prompt_logprobs"]
        assert topk is not None
        assert len(topk) == 3
        assert topk[0] is None
        # Exactly top-k (2) entries per position, sorted by logprob descending
        assert len(topk[1]) == 2
        assert len(topk[2]) == 2
        # Position 1: top-2 are token 20 at -0.5 and 9010 at -1.0 (9011 at -1.1 is dropped)
        topk1 = dict(topk[1])
        assert topk1[20] == pytest.approx(-0.5)
        assert topk1[9010] == pytest.approx(-1.0)

    @pytest.mark.asyncio
    async def test_sample_topk_without_include_returns_none(self, client):
        """topk_prompt_logprobs alone does not return prompt logprobs when include_prompt_logprobs is False."""
        request_payload = {
            "json": {
                "prompt": {"chunks": [{"tokens": [10, 20, 30]}]},
                "num_samples": 1,
                "sampling_params": {"temperature": 0.7, "max_tokens": 64},
                "topk_prompt_logprobs": 2,
            }
        }
        result = await client.sample(request_payload)

        assert result["prompt_logprobs"] is None
        assert result["topk_prompt_logprobs"] is None


class TestRenderChatCompletion:
    """Test render_chat_completion method (multimodal and text-only)."""

    @pytest.mark.asyncio
    async def test_render_chat_completion_basic(self, client):
        """Text-only render returns correct top-level fields and features is None."""
        request_payload = {
            "json": {
                "model": "test",
                "messages": [{"role": "user", "content": "Hello, who are you?"}],
            },
        }
        result = await client.render_chat_completion(request_payload)

        assert result["request_id"] == "chatcmpl-mock-0"
        assert result["token_ids"] == list(range(10))
        assert result["sampling_params"] == {"temperature": 0.7, "max_tokens": 100}
        assert result["model"] == "test"
        assert result["features"] is None
        assert result["stream"] is False
        assert result["stream_options"] is None
        assert result["cache_salt"] is None
        assert result["priority"] == 0
        assert result["kv_transfer_params"] is None

    @pytest.mark.asyncio
    async def test_render_chat_completion_multimodal(self, client):
        """Multimodal render returns features with mm_hashes and mm_placeholders."""
        request_payload = {
            "json": {
                "model": "test-vlm",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"},
                            },
                            {"type": "text", "text": "What is in this image?"},
                        ],
                    }
                ],
            },
        }
        result = await client.render_chat_completion(request_payload)

        assert result["request_id"] == "chatcmpl-mock-0"
        assert result["token_ids"] == list(range(110))
        assert result["sampling_params"] == {"temperature": 0.7, "max_tokens": 100}
        assert result["model"] == "test-vlm"
        assert result["stream"] is False
        assert result["stream_options"] is None
        assert result["cache_salt"] is None
        assert result["priority"] == 0
        assert result["kv_transfer_params"] is None

        assert result["features"] == {
            "mm_hashes": {"image": ["fd2700fd096d55f04c20dbfdc903f7e65421e282"]},
            "mm_placeholders": {"image": [{"offset": 4, "length": 100}]},
        }


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
