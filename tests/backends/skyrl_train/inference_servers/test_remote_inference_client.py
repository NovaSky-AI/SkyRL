"""Tests for RemoteInferenceClient."""

import asyncio
import pickle
import threading
import time
from typing import Dict, List, Optional

import aiohttp
import httpx
import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from skyrl.backends.skyrl_train.inference_servers.common import get_open_port
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    SKYRL_LORA_ADAPTER_NAME,
    PauseMode,
    RemoteInferenceClient,
)


def create_mock_vllm_server(server_id: int) -> FastAPI:
    """Create a mock vLLM server with standard endpoints."""
    app = FastAPI()
    app.state.last_generate_features = None
    app.state.last_generate_model = None
    app.state.last_chat_model = None
    app.state.last_completion_model = None
    app.state.last_render_model = None
    # Per-server LoRA registry: lora_name -> lora_path
    app.state.lora_registry = {}
    # Scripted-response support for testing sample_with_retry. Tests can
    # POST to /test/script_generate with a list of partial choice payloads
    # (each merged into the next /inference/v1/generate response) so the
    # retry loop can be driven through abort → stop transitions.
    app.state.generate_script: List[Dict] = []
    app.state.generate_payloads: List[Dict] = []
    # Tracks the last per-LoRA abort call made to this server, for
    # /skyrl/v1/abort_lora_requests.
    app.state.last_abort_lora_name = None
    app.state.abort_lora_call_count = 0

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/test/last_generate_features")
    async def get_last_generate_features():
        return {"features": app.state.last_generate_features}

    @app.get("/test/last_models")
    async def get_last_models():
        return {
            "generate": app.state.last_generate_model,
            "chat": app.state.last_chat_model,
            "completion": app.state.last_completion_model,
            "render": app.state.last_render_model,
        }

    @app.get("/test/lora_registry")
    async def get_lora_registry():
        return {"registry": dict(app.state.lora_registry)}

    @app.get("/get_world_size")
    async def get_world_size():
        return {"world_size": 2}  # Simulate TP=2

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        app.state.last_completion_model = body.get("model")
        prompts = body.get("prompt", [])
        n_prompts = len(prompts) if isinstance(prompts, list) else 1
        return {
            "choices": [
                {"index": i, "text": f"Response {i} from server {server_id}", "finish_reason": "stop"}
                for i in range(n_prompts)
            ],
            "model": body.get("model"),
        }

    @app.post("/skyrl/v1/generate")
    @app.post("/inference/v1/generate")
    async def generate(request: Request):
        body = await request.json()  # Consume body
        sp = body.get("sampling_params", {})
        input_token_ids = body.get("token_ids", [])
        app.state.last_generate_model = body.get("model")
        # Record incoming payload for sample_with_retry tests to inspect.
        app.state.generate_payloads.append(body)
        n = sp.get("n", 1)
        # If logprobs is explicitly set (sample path), use n for num_choices.
        # Otherwise (generate path), use len(token_ids) for per-prompt responses.
        if "logprobs" in sp:
            num_choices = n
        else:
            num_choices = 1

        # Scripted response path: tests can pre-load a list of choice
        # overrides into app.state.generate_script. Each call pops the
        # front entry and merges it into the default choice. The list is
        # drained in FIFO order; once empty we fall back to the default
        # "stop" response below.
        if app.state.generate_script:
            override = app.state.generate_script.pop(0)
            default_choice = {
                "request_id": "dummy",
                "token_ids": override.get("token_ids", []),
                "finish_reason": override.get("finish_reason", "stop"),
                "logprobs": override.get(
                    "logprobs",
                    {"content": [{"logprob": -0.1 * (i + 1)} for i in range(len(override.get("token_ids", [])))]},
                ),
            }
            return {"choices": [default_choice for _ in range(num_choices)]}

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

        features = body.get("features")
        app.state.last_generate_features = features
        if features is not None:
            response["features"] = features

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
        body = await request.json()
        app.state.last_chat_model = body.get("model")
        return {
            "choices": [{"message": {"content": f"Chat from server {server_id}"}}],
            "model": body.get("model"),
        }

    @app.post("/v1/chat/completions/render")
    async def render_chat_completion(request: Request):
        body = await request.json()
        app.state.last_render_model = body.get("model")
        messages = body.get("messages", [])

        # Count image_url parts across all messages.
        num_images = sum(
            1
            for msg in messages
            if isinstance(msg.get("content"), list)
            for c in msg["content"]
            if c.get("type") == "image_url"
        )

        features = None
        if num_images > 0:
            # Each image gets 100 placeholder tokens.  Token IDs are laid out as:
            # [0..3] preamble, then 100 tokens per image, then [N..N+5] suffix.
            placeholder_size = 100
            preamble_len = 4
            total_len = preamble_len + num_images * placeholder_size + 6

            mm_hashes = []
            mm_placeholders = []
            kwargs_items = []
            for i in range(num_images):
                offset = preamble_len + i * placeholder_size
                mm_hashes.append(f"hash-{i}")
                mm_placeholders.append({"offset": offset, "length": placeholder_size})
                kwargs_items.append(f"mock-encoded-tensor-{i}")

            features = {
                "mm_hashes": {"image": mm_hashes},
                "mm_placeholders": {"image": mm_placeholders},
                "kwargs_data": {"image": kwargs_items},
            }
        else:
            total_len = 10

        return {
            "request_id": f"chatcmpl-mock-{server_id}",
            "token_ids": list(range(total_len)),
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

    @app.post("/skyrl/v1/abort_lora_requests")
    async def abort_lora_requests(request: Request):
        body = await request.json()
        lora_name = body.get("lora_name")
        if not lora_name:
            return JSONResponse(status_code=400, content={"error": "'lora_name' required"})
        app.state.last_abort_lora_name = lora_name
        app.state.abort_lora_call_count += 1
        return {"status": "ok", "aborted": [], "count": 0, "server_id": server_id}

    @app.post("/test/script_generate")
    async def script_generate(request: Request):
        """Test helper: pre-load scripted responses for /inference/v1/generate."""
        body = await request.json()
        app.state.generate_script = list(body.get("script", []))
        app.state.generate_payloads = []
        return {"status": "ok"}

    @app.get("/test/abort_lora_state")
    async def abort_lora_state():
        return {
            "last_abort_lora_name": app.state.last_abort_lora_name,
            "abort_lora_call_count": app.state.abort_lora_call_count,
        }

    @app.get("/test/generate_payloads")
    async def get_generate_payloads():
        return {"payloads": app.state.generate_payloads}

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

    @app.post("/skyrl/v1/load_lora_adapter")
    async def load_lora_adapter(request: Request):
        body = await request.json()
        lora_name = body.get("lora_name")
        lora_path = body.get("lora_path")
        if lora_name is None or lora_path is None:
            return JSONResponse(
                status_code=400,
                content={"object": "error", "message": "missing lora_name/lora_path", "type": "BadRequest"},
            )
        app.state.lora_registry[lora_name] = lora_path
        return PlainTextResponse(f"Success: LoRA adapter '{lora_name}' added successfully on server {server_id}.")

    @app.post("/v1/unload_lora_adapter")
    async def unload_lora_adapter(request: Request):
        body = await request.json()
        lora_name = body.get("lora_name")
        if lora_name is None or lora_name not in app.state.lora_registry:
            return JSONResponse(
                status_code=404,
                content={
                    "object": "error",
                    "message": f"adapter '{lora_name}' not found",
                    "type": "NotFoundError",
                },
            )
        del app.state.lora_registry[lora_name]
        return PlainTextResponse(f"Success: LoRA adapter '{lora_name}' removed successfully on server {server_id}.")

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
        data_parallel_size=1,
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
            data_parallel_size=1,
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

            def for_servers(self, world_size_per_server, num_servers, dp_size=1):
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
                "model": client.model_name,
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
                "model": client.model_name,
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
                "model": client.model_name,
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
                "model": client.model_name,
                "prompt": {"chunks": [{"tokens": [10, 20, 30]}]},
                "num_samples": 1,
                "sampling_params": {"temperature": 0.7, "max_tokens": 64},
                "topk_prompt_logprobs": 2,
            }
        }
        result = await client.sample(request_payload)

        assert result["prompt_logprobs"] is None
        assert result["topk_prompt_logprobs"] is None

    @pytest.mark.asyncio
    async def test_sample_with_image(self, client):
        """Sample with [text, image, text] calls render and splices tokens correctly."""
        import base64

        image_bytes = base64.b64encode(b"fake-jpeg-data").decode("ascii")
        request_payload = {
            "json": {
                "model": client.model_name,
                "prompt": {
                    "chunks": [
                        {"type": "encoded_text", "tokens": [100, 101, 102]},
                        {
                            "type": "image",
                            "data": image_bytes,
                            "format": "jpeg",
                        },
                        {"type": "encoded_text", "tokens": [200, 201]},
                    ]
                },
                "num_samples": 1,
                "sampling_params": {"temperature": 0.7, "max_tokens": 64},
            }
        }
        result = await client.sample(request_payload)

        assert result["type"] == "sample"
        assert len(result["sequences"]) == 1

        seq = result["sequences"][0]
        assert "tokens" in seq
        assert "logprobs" in seq
        assert seq["stop_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_sample_with_image_asset_pointer(self, client):
        """Sample with image_asset_pointer sends location URL to render."""
        request_payload = {
            "json": {
                "model": client.model_name,
                "prompt": {
                    "chunks": [
                        {"type": "encoded_text", "tokens": [10, 11]},
                        {
                            "type": "image_asset_pointer",
                            "format": "png",
                            "location": "https://example.com/image.png",
                        },
                        {"type": "encoded_text", "tokens": [20, 21]},
                    ]
                },
                "num_samples": 1,
                "sampling_params": {"temperature": 0.7, "max_tokens": 64},
            }
        }
        result = await client.sample(request_payload)

        assert result["type"] == "sample"
        assert len(result["sequences"]) == 1

    @pytest.mark.asyncio
    async def test_sample_text_only_no_features(self, client):
        """Text-only sample does not include features in the generate payload."""
        request_payload = {
            "json": {
                "model": client.model_name,
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                "num_samples": 1,
                "sampling_params": {"temperature": 0.7, "max_tokens": 64},
            }
        }
        result = await client.sample(request_payload)

        assert result["type"] == "sample"
        assert len(result["sequences"]) == 1


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
            "mm_hashes": {"image": ["hash-0"]},
            "mm_placeholders": {"image": [{"offset": 4, "length": 100}]},
            "kwargs_data": {"image": ["mock-encoded-tensor-0"]},
        }


class TestMultiModalGeneration:
    """Test that mm_features are correctly forwarded through generate()."""

    @pytest.mark.asyncio
    async def test_generate_with_mm_features(self, client, mock_servers):
        """Passing mm_features in InferenceEngineInput sends features in the HTTP payload."""
        mm_features = {
            "mm_hashes": {"image": ["abc123hash"]},
            "mm_placeholders": {"image": [{"offset": 0, "length": 10}]},
        }
        input_batch = {
            "prompt_token_ids": [[1, 2, 3]],
            "sampling_params": {"max_tokens": 50},
            "mm_features": [mm_features],
        }
        result = await client.generate(input_batch)

        assert len(result["responses"]) == 1
        assert len(result["response_ids"]) == 1
        assert result["stop_reasons"][0] == "stop"

        async with httpx.AsyncClient() as http:
            resp = await http.get(f"{mock_servers['proxy_url']}/test/last_generate_features")
            captured = resp.json()
        assert captured["features"] == mm_features


class TestContextManager:
    """Test async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_servers):
        """Test using client as async context manager."""

        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            data_parallel_size=1,
        )

        async with client:
            result = await client.resume()
            assert len(result) == 2

        # Session should be closed after exiting context
        assert client._session is None or client._session.closed


async def _get_lora_registries(server_urls: List[str]) -> List[Dict[str, str]]:
    """Helper: read the per-server LoRA registries from each mock server."""
    registries: List[Dict[str, str]] = []
    async with httpx.AsyncClient() as http:
        for url in server_urls:
            resp = await http.get(f"{url}/test/lora_registry")
            registries.append(resp.json()["registry"])
    return registries


async def _get_last_models(server_urls: List[str]) -> List[Dict[str, Optional[str]]]:
    """Helper: read the last per-method ``model`` field captured by each mock."""
    last: List[Dict[str, Optional[str]]] = []
    async with httpx.AsyncClient() as http:
        for url in server_urls:
            resp = await http.get(f"{url}/test/last_models")
            last.append(resp.json())
    return last


class TestLoRAControlPlane:
    """Test load_lora_adapter / unload_lora_adapter fan-out and bookkeeping."""

    @pytest.mark.asyncio
    async def test_load_lora_adapter_fans_out(self, client, mock_servers):
        result = await client.load_lora_adapter("lora-A", "/tmp/path/lora-A")
        assert len(result) == 2
        for url, response in result.items():
            assert response["status"] == 200
            assert "Success" in response["body"]
            assert "lora-A" in response["body"]

        registries = await _get_lora_registries(mock_servers["server_urls"])
        for reg in registries:
            assert reg.get("lora-A") == "/tmp/path/lora-A"

        await client.unload_lora_adapter("lora-A")

    @pytest.mark.asyncio
    async def test_load_lora_adapter_inplace_reload(self, client, mock_servers):
        await client.load_lora_adapter("lora-X", "/tmp/path/v1")
        await client.load_lora_adapter("lora-X", "/tmp/path/v2")

        registries = await _get_lora_registries(mock_servers["server_urls"])
        for reg in registries:
            assert reg.get("lora-X") == "/tmp/path/v2"

        await client.unload_lora_adapter("lora-X")

    @pytest.mark.asyncio
    async def test_unload_lora_adapter_fans_out(self, client, mock_servers):
        await client.load_lora_adapter("lora-B", "/tmp/path/lora-B")

        result = await client.unload_lora_adapter("lora-B")
        assert len(result) == 2
        for url, response in result.items():
            assert response["status"] == 200
            assert "Success" in response["body"]

        registries = await _get_lora_registries(mock_servers["server_urls"])
        for reg in registries:
            assert "lora-B" not in reg

    @pytest.mark.asyncio
    async def test_unload_unknown_lora_raises(self, client, mock_servers):
        # Server returns 404, surfaced as ClientResponseError via raise_for_status.
        with pytest.raises(aiohttp.ClientResponseError):
            await client.unload_lora_adapter("nonexistent-lora")
        registries = await _get_lora_registries(mock_servers["server_urls"])
        for reg in registries:
            assert "nonexistent-lora" not in reg

    @pytest.mark.asyncio
    async def test_default_lora_adapter_constant(self):
        # Sanity check that the public constant has the documented value used
        # across the SkyRL training paths.
        assert SKYRL_LORA_ADAPTER_NAME == "skyrl-lora"


class TestExplicitModelRequired:
    """Data-plane ``model`` resolution rules.

    Every data-plane method (``generate``, ``sample``, ``chat_completion``,
    ``completion``, ``render_chat_completion``) follows the same rule:

    - If a non-empty ``model`` is supplied (kwarg for ``generate``, body field
      for the others) it is threaded through to the server as-is.
    - If no model is supplied and LoRA is **not** in use
      (``uses_lora_weight_sync=False``), the call falls back to
      ``client.model_name``.
    - If no model is supplied and LoRA **is** in use, the call raises
      ``ValueError`` so requests don't silently target the base model.
    """

    @pytest.mark.asyncio
    async def test_generate_threads_model_into_payload(self, mock_servers):
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            input_batch = {
                "prompt_token_ids": [[1, 2, 3]],
                "sampling_params": {"max_tokens": 50},
            }
            await client.generate(input_batch, model="lora-explicit")
            captured = await _get_last_models(mock_servers["server_urls"])
            # generate routes through proxy_url == first server.
            assert captured[0]["generate"] == "lora-explicit"
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_generate_defaults_to_base_when_no_lora(self, mock_servers):
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            input_batch = {
                "prompt_token_ids": [[1, 2, 3]],
                "sampling_params": {"max_tokens": 50},
            }
            await client.generate(input_batch)
            captured = await _get_last_models(mock_servers["server_urls"])
            assert captured[0]["generate"] == "base-model"
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_generate_raises_when_lora_and_no_model(self, mock_servers):
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            uses_lora_weight_sync=True,
            data_parallel_size=1,
        )
        try:
            input_batch = {
                "prompt_token_ids": [[1, 2, 3]],
                "sampling_params": {"max_tokens": 50},
            }
            with pytest.raises(ValueError, match="LoRA is enabled"):
                await client.generate(input_batch)
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_chat_completion_uses_body_model(self, mock_servers):
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            request_payload = {
                "json": {
                    "model": "lora-chat",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                "headers": {},
            }
            await client.chat_completion(request_payload)
            captured = await _get_last_models(mock_servers["server_urls"])
            assert captured[0]["chat"] == "lora-chat"
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_chat_completion_defaults_to_base_when_no_lora(self, mock_servers):
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            request_payload = {
                "json": {"messages": [{"role": "user", "content": "hi"}]},
                "headers": {},
            }
            await client.chat_completion(request_payload)
            captured = await _get_last_models(mock_servers["server_urls"])
            assert captured[0]["chat"] == "base-model"
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_chat_completion_raises_when_lora_and_no_model(self, mock_servers):
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            uses_lora_weight_sync=True,
            data_parallel_size=1,
        )
        try:
            request_payload = {
                "json": {"messages": [{"role": "user", "content": "hi"}]},
                "headers": {},
            }
            with pytest.raises(ValueError, match="LoRA is enabled"):
                await client.chat_completion(request_payload)
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_completion_uses_body_model(self, mock_servers):
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            request_payload = {
                "json": {"model": "lora-completion", "prompt": "hello"},
                "headers": {},
            }
            await client.completion(request_payload)
            captured = await _get_last_models(mock_servers["server_urls"])
            assert captured[0]["completion"] == "lora-completion"
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_completion_defaults_to_base_when_no_lora(self, mock_servers):
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            request_payload = {"json": {"prompt": "hello"}, "headers": {}}
            await client.completion(request_payload)
            captured = await _get_last_models(mock_servers["server_urls"])
            assert captured[0]["completion"] == "base-model"
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_completion_raises_when_lora_and_no_model(self, mock_servers):
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            uses_lora_weight_sync=True,
            data_parallel_size=1,
        )
        try:
            request_payload = {"json": {"prompt": "hello"}, "headers": {}}
            with pytest.raises(ValueError, match="LoRA is enabled"):
                await client.completion(request_payload)
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_render_chat_completion_uses_body_model(self, mock_servers):
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            request_payload = {
                "json": {
                    "model": "lora-render",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            }
            result = await client.render_chat_completion(request_payload)
            assert result["model"] == "lora-render"
            captured = await _get_last_models(mock_servers["server_urls"])
            assert captured[0]["render"] == "lora-render"
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_render_chat_completion_defaults_to_base_when_no_lora(self, mock_servers):
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            request_payload = {"json": {"messages": [{"role": "user", "content": "hi"}]}}
            result = await client.render_chat_completion(request_payload)
            assert result["model"] == "base-model"
            captured = await _get_last_models(mock_servers["server_urls"])
            assert captured[0]["render"] == "base-model"
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_render_chat_completion_raises_when_lora_and_no_model(self, mock_servers):
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            uses_lora_weight_sync=True,
            data_parallel_size=1,
        )
        try:
            request_payload = {"json": {"messages": [{"role": "user", "content": "hi"}]}}
            with pytest.raises(ValueError, match="LoRA is enabled"):
                await client.render_chat_completion(request_payload)
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_sample_uses_body_model(self, mock_servers):
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            request_payload = {
                "json": {
                    "model": "lora-sample",
                    "prompt": {"chunks": [{"tokens": [1, 2, 3]}]},
                    "num_samples": 1,
                    "sampling_params": {"temperature": 0.7, "max_tokens": 16},
                }
            }
            await client.sample(request_payload)
            captured = await _get_last_models(mock_servers["server_urls"])
            assert captured[0]["generate"] == "lora-sample"
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_sample_defaults_to_base_when_no_lora(self, mock_servers):
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            request_payload = {
                "json": {
                    "prompt": {"chunks": [{"tokens": [1, 2, 3]}]},
                    "num_samples": 1,
                    "sampling_params": {"temperature": 0.7, "max_tokens": 16},
                }
            }
            await client.sample(request_payload)
            captured = await _get_last_models(mock_servers["server_urls"])
            assert captured[0]["generate"] == "base-model"
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_sample_raises_when_lora_and_no_model(self, mock_servers):
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            uses_lora_weight_sync=True,
            data_parallel_size=1,
        )
        try:
            request_payload = {
                "json": {
                    "prompt": {"chunks": [{"tokens": [1, 2, 3]}]},
                    "num_samples": 1,
                    "sampling_params": {"temperature": 0.7, "max_tokens": 16},
                }
            }
            with pytest.raises(ValueError, match="LoRA is enabled"):
                await client.sample(request_payload)
        finally:
            await client.teardown()


# ---------------------------------------------------------------------------
# Targeted (per-LoRA) pause + sample_with_retry tests.
# TRANSIENT: delete this section when vLLM ships native per-LoRA pause and
# the sample_with_retry / per-LoRA gate are removed.
# ---------------------------------------------------------------------------


async def _script_generate(server_url: str, script: List[Dict]) -> None:
    """Helper: pre-load scripted /inference/v1/generate responses on one server."""
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{server_url}/test/script_generate", json={"script": script}) as resp:
            assert resp.status == 200


async def _get_generate_payloads(server_url: str) -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{server_url}/test/generate_payloads") as resp:
            data = await resp.json()
            return data["payloads"]


async def _get_abort_lora_state(server_url: str) -> Dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{server_url}/test/abort_lora_state") as resp:
            return await resp.json()


class TestTargetedLoraPause:
    """Tests for pause_generation(lora_name=...) and sample_with_retry."""

    @pytest.mark.asyncio
    async def test_pause_generation_with_lora_name_fans_out_abort(self, mock_servers, monkeypatch):
        """pause_generation(lora_name=X) clears the event and fans out abort to all servers."""
        # Shorten the grace period so this test runs quickly.
        monkeypatch.setattr(
            "skyrl.backends.skyrl_train.inference_servers.remote_inference_client.ABORT_GENERATION_GRACE_PERIOD_SECONDS",
            0,
        )
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            result = await client.pause_generation(lora_name="lora-A")
            assert len(result) == 2  # fanned out to both servers
            for url, resp in result.items():
                assert resp["status"] == 200
                assert resp["body"]["status"] == "ok"
                assert resp["body"]["server_id"] in (0, 1)
            # Both servers should have observed the abort call.
            for url in mock_servers["server_urls"]:
                state = await _get_abort_lora_state(url)
                assert state["last_abort_lora_name"] == "lora-A"
                assert state["abort_lora_call_count"] >= 1
            # Event is now CLEAR (paused). resume_generation should re-open it.
            ev = client._lora_pause_events["lora-A"]
            assert not ev.is_set()
            await client.resume_generation(lora_name="lora-A")
            assert ev.is_set()
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_pause_generation_without_lora_uses_keep_mode(self, mock_servers):
        """pause_generation() (no lora_name) preserves the global keep-mode path."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            result = await client.pause_generation()
            assert len(result) == 2
            for url, resp in result.items():
                assert resp["status"] == 200
                # The mock /pause echoes back the mode query param.
                assert resp["body"]["mode"] == "keep"
            # No per-LoRA event should have been created.
            assert client._lora_pause_events == {}
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_sample_with_retry_accumulates_on_abort(self, mock_servers):
        """abort-then-stop: tokens are accumulated, max_tokens decremented, final stop_reason='stop'."""
        # Script the first response as abort with 5 tokens, second as stop with 4 tokens.
        # The mock has only one /inference/v1/generate endpoint shared by both servers,
        # so we script both servers identically — the second call may hit either server
        # (random load balancing) but both will produce the same scripted "stop" response.
        for url in mock_servers["server_urls"]:
            await _script_generate(
                url,
                [
                    {
                        "token_ids": [100, 101, 102, 103, 104],
                        "finish_reason": "abort",
                        "logprobs": {"content": [{"logprob": -0.1 * i} for i in range(1, 6)]},
                    },
                    {
                        "token_ids": [200, 201, 202, 203],
                        "finish_reason": "stop",
                        "logprobs": {"content": [{"logprob": -0.2 * i} for i in range(1, 5)]},
                    },
                ],
            )

        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            request_payload = {
                "json": {
                    "model": "lora-A",
                    "prompt": {"chunks": [{"tokens": [10, 20, 30]}]},
                    "num_samples": 1,
                    "sampling_params": {"max_tokens": 16, "temperature": 0.7},
                }
            }
            result = await client.sample_with_retry(request_payload)

            seq = result["sequences"][0]
            assert seq["tokens"] == [100, 101, 102, 103, 104, 200, 201, 202, 203]
            assert seq["stop_reason"] == "stop"
            assert seq["logprobs"] is not None
            assert len(seq["logprobs"]) == 9

            # Verify retry math: second call's token_ids = original + accumulated,
            # max_tokens reduced by len(accumulated). Inspect both servers since
            # routing is random.
            all_payloads = []
            for url in mock_servers["server_urls"]:
                all_payloads.extend(await _get_generate_payloads(url))
            assert len(all_payloads) == 2
            assert all_payloads[0]["token_ids"] == [10, 20, 30]
            assert all_payloads[0]["sampling_params"]["max_tokens"] == 16
            assert all_payloads[1]["token_ids"] == [10, 20, 30, 100, 101, 102, 103, 104]
            assert all_payloads[1]["sampling_params"]["max_tokens"] == 16 - 5
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_sample_with_retry_no_abort_single_shot(self, mock_servers):
        """When no abort happens, sample_with_retry behaves like a single sample()."""
        # Clear any scripts left from prior tests.
        for url in mock_servers["server_urls"]:
            await _script_generate(url, [])
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            request_payload = {
                "json": {
                    "model": "lora-A",
                    "prompt": {"chunks": [{"tokens": [10, 20, 30]}]},
                    "num_samples": 1,
                    "sampling_params": {"max_tokens": 16},
                }
            }
            result = await client.sample_with_retry(request_payload)
            seq = result["sequences"][0]
            # Default mock returns tokens [0, 1, 2] with stop.
            assert seq["tokens"] == [0, 1, 2]
            assert seq["stop_reason"] == "stop"
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_sample_with_retry_blocks_until_resume(self, mock_servers, monkeypatch):
        """While pause_generation(lora_name=X) is active, sample_with_retry for X blocks."""
        # Shorten grace so the test does not stall on the pause helper.
        monkeypatch.setattr(
            "skyrl.backends.skyrl_train.inference_servers.remote_inference_client.ABORT_GENERATION_GRACE_PERIOD_SECONDS",
            0,
        )
        # Script every call as abort so the retry loop is bound entirely by the gate.
        # We only need the gate test, so any abort/stop sequence works once unpaused.
        for url in mock_servers["server_urls"]:
            await _script_generate(
                url,
                [
                    {"token_ids": [1, 2, 3], "finish_reason": "abort"},
                    {"token_ids": [4, 5], "finish_reason": "stop"},
                ],
            )

        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            # Pause first so the event is created and CLEAR before the sample starts.
            await client.pause_generation(lora_name="lora-A")

            request_payload = {
                "json": {
                    "model": "lora-A",
                    "prompt": {"chunks": [{"tokens": [10, 20, 30]}]},
                    "num_samples": 1,
                    "sampling_params": {"max_tokens": 16},
                }
            }
            sample_task = asyncio.create_task(client.sample_with_retry(request_payload))

            # Give the sample loop time to reach the .wait() on the event.
            await asyncio.sleep(0.1)
            assert not sample_task.done(), "sample_with_retry should block while paused"

            # Resume → sample_with_retry continues, completes via scripted stop.
            await client.resume_generation(lora_name="lora-A")
            result = await asyncio.wait_for(sample_task, timeout=5.0)
            seq = result["sequences"][0]
            # First scripted response is abort with 3 tokens, second is stop with 2.
            assert seq["tokens"] == [1, 2, 3, 4, 5]
            assert seq["stop_reason"] == "stop"
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_sample_with_retry_rejects_n_gt_one(self, mock_servers):
        """sample_with_retry only supports num_samples=1."""
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            request_payload = {
                "json": {
                    "model": "lora-A",
                    "prompt": {"chunks": [{"tokens": [1]}]},
                    "num_samples": 2,
                    "sampling_params": {"max_tokens": 8},
                }
            }
            with pytest.raises(ValueError, match="num_samples=1"):
                await client.sample_with_retry(request_payload)
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_sample_with_retry_truncates_prompt_logprobs(self, mock_servers):
        """prompt_logprobs from the final retry must be truncated to original prompt length.

        When a retry fires, the resubmitted request has
        ``token_ids = original_prompt + accumulated_tokens``. The server
        computes prompt_logprobs over that extended prompt and the final
        response carries entries for both the original prompt AND the
        accumulated tokens. The caller asked for prompt_logprobs of their
        original prompt only — the extra entries must be stripped before
        return, otherwise the response shape differs between the
        no-abort and abort-then-retry paths for the same logical request.
        """
        # Script one abort with 5 partial tokens. The retry then falls
        # through to the default mock path which returns prompt_logprobs
        # whenever sampling_params['prompt_logprobs'] is set.
        for url in mock_servers["server_urls"]:
            await _script_generate(
                url,
                [
                    {"token_ids": [100, 101, 102, 103, 104], "finish_reason": "abort"},
                ],
            )
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            request_payload = {
                "json": {
                    "model": "base-model",
                    "prompt": {"chunks": [{"tokens": [10, 20, 30]}]},  # length 3
                    "num_samples": 1,
                    "sampling_params": {"max_tokens": 64},
                    "include_prompt_logprobs": True,
                }
            }
            result = await client.sample_with_retry(request_payload)

            # Final length must be the ORIGINAL prompt length (3), not the
            # extended prompt length (3 + 5 = 8) that the retry sent.
            pl = result["prompt_logprobs"]
            assert pl is not None
            assert len(pl) == 3, f"expected 3 prompt_logprobs entries, got {len(pl)} (likely missing truncation)"
            # Values should match a single-shot sample for the same original
            # prompt (position 0 = no prior context, positions 1-2 follow
            # the mock's autoregressive logprob formula).
            assert pl[0] is None
            assert pl[1] == pytest.approx(-0.5)
            assert pl[2] == pytest.approx(-1.0)

            # Verify the server actually saw an extended-prompt request on
            # the retry (otherwise the test isn't proving truncation).
            all_payloads = []
            for url in mock_servers["server_urls"]:
                all_payloads.extend(await _get_generate_payloads(url))
            assert len(all_payloads) == 2
            assert len(all_payloads[1]["token_ids"]) == 3 + 5
        finally:
            await client.teardown()

    @pytest.mark.asyncio
    async def test_sample_with_retry_no_lora_event_no_blocking(self, mock_servers):
        """When the LoRA has never been paused, sample_with_retry must not block."""
        for url in mock_servers["server_urls"]:
            await _script_generate(url, [])  # default mock returns "stop"
        client = RemoteInferenceClient(
            proxy_url=mock_servers["proxy_url"],
            server_urls=mock_servers["server_urls"],
            model_name="base-model",
            data_parallel_size=1,
        )
        try:
            request_payload = {
                "json": {
                    "model": "unpaused-lora",
                    "prompt": {"chunks": [{"tokens": [1]}]},
                    "num_samples": 1,
                    "sampling_params": {"max_tokens": 8},
                }
            }
            # Should complete promptly with no event ever created for "unpaused-lora".
            result = await asyncio.wait_for(client.sample_with_retry(request_payload), timeout=2.0)
            assert result["sequences"][0]["stop_reason"] == "stop"
            assert "unpaused-lora" not in client._lora_pause_events
        finally:
            await client.teardown()
