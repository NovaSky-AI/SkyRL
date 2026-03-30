"""
Tests for R3 (Rollout Router Replay) data serialization through routers.

Verifies that routed_experts data survives the HTTP round-trip through:
1. Direct connection (no router) - baseline
2. InferenceRouter (Python)
3. VLLMRouter (Rust binary, if installed)

Run with:
    uv run --extra dev --extra fsdp pytest -xvs tests/backends/skyrl_train/inference_servers/test_r3_serialization.py
"""

import asyncio
import shutil
import threading
import time
from typing import List, Optional

import httpx
import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI, Request

from skyrl.backends.skyrl_train.inference_servers.common import get_open_port
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.router import InferenceRouter

# Moonlight-16B-like dimensions
NUM_MOE_LAYERS = 27
TOPK = 6
NUM_EXPERTS = 64
PROMPT_LEN = 32
RESPONSE_LEN = 64
TOTAL_SEQ_LEN = PROMPT_LEN + RESPONSE_LEN

# Pre-generate deterministic routed_experts as np.int32 ndarray, matching
# vLLM's RoutedExpertsReader which stores data as np.ndarray(dtype=np.int32).
# Shape: [total_seq_len, num_moe_layers, topk]
rng = np.random.RandomState(42)
EXPECTED_ROUTED_EXPERTS_NP: np.ndarray = rng.randint(
    0, NUM_EXPERTS, size=(TOTAL_SEQ_LEN, NUM_MOE_LAYERS, TOPK), dtype=np.int32
)
# Python-native list form for assertion comparisons (what .tolist() produces)
EXPECTED_ROUTED_EXPERTS: List[List[List[int]]] = EXPECTED_ROUTED_EXPERTS_NP.tolist()

EXPECTED_TOKEN_IDS = list(range(100, 100 + RESPONSE_LEN))
EXPECTED_LOGPROBS = [{"logprob": -float(i) * 0.1} for i in range(RESPONSE_LEN)]


def create_r3_mock_server() -> FastAPI:
    """Mock vLLM server that returns realistic routed_experts data."""
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/skyrl/v1/generate")
    async def skyrl_generate(request: Request):
        """Replicate the real _skyrl_generate endpoint serialization path.

        vLLM's CompletionOutput.routed_experts is an np.ndarray(dtype=np.int32).
        The real endpoint calls .tolist() before returning JSON — we do the same
        so this test catches any numpy-specific serialization issues (e.g. numpy
        scalars surviving .tolist() that then fail JSON encoding or silently
        change dtype).
        """
        body = await request.json()
        routed_experts = EXPECTED_ROUTED_EXPERTS_NP
        if hasattr(routed_experts, "tolist"):
            routed_experts = routed_experts.tolist()

        return {
            "choices": [
                {
                    "token_ids": EXPECTED_TOKEN_IDS,
                    "finish_reason": "stop",
                    "logprobs": {"content": EXPECTED_LOGPROBS},
                    "routed_experts": routed_experts,
                }
            ]
        }

    @app.post("/tokenize")
    async def tokenize(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "")
        return {"tokens": list(range(len(prompt)))}

    @app.post("/detokenize")
    async def detokenize(request: Request):
        return {"prompt": "mock detokenized text"}

    @app.get("/get_world_size")
    async def get_world_size():
        return {"world_size": 1}

    return app


def start_server(port: int) -> uvicorn.Server:
    app = create_r3_mock_server()
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True).start()
    return server


def wait_ready(url: str, timeout: float = 10.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
                return True
        except httpx.RequestError:
            time.sleep(0.1)
    return False


@pytest.fixture(scope="module")
def backend_server():
    """Start mock vLLM backend server."""
    port = get_open_port()
    url = f"http://127.0.0.1:{port}"
    server = start_server(port)
    assert wait_ready(url), f"Backend server failed to start on {url}"
    yield url
    server.should_exit = True
    time.sleep(0.3)


def _verify_routed_experts(actual: Optional[List], expected: List[List[List[int]]]):
    """Assert routed_experts match exactly, with diagnostic messages."""
    assert actual is not None, "routed_experts is None — data was dropped in transit"
    assert isinstance(actual, list), f"routed_experts should be list, got {type(actual)}"
    assert len(actual) == len(expected), f"seq_len mismatch: got {len(actual)}, expected {len(expected)}"

    for tok_idx in range(len(expected)):
        actual_tok = actual[tok_idx]
        expected_tok = expected[tok_idx]
        assert len(actual_tok) == len(expected_tok), (
            f"num_layers mismatch at token {tok_idx}: " f"got {len(actual_tok)}, expected {len(expected_tok)}"
        )
        for layer_idx in range(len(expected_tok)):
            actual_layer = actual_tok[layer_idx]
            expected_layer = expected_tok[layer_idx]
            assert len(actual_layer) == len(expected_layer), (
                f"topk mismatch at token {tok_idx} layer {layer_idx}: "
                f"got {len(actual_layer)}, expected {len(expected_layer)}"
            )
            for k_idx in range(len(expected_layer)):
                a_val = actual_layer[k_idx]
                e_val = expected_layer[k_idx]
                assert a_val == e_val and type(a_val) is type(e_val), (
                    f"value mismatch at [{tok_idx}][{layer_idx}][{k_idx}]: "
                    f"got {a_val!r} (type {type(a_val).__name__}), "
                    f"expected {e_val!r} (type {type(e_val).__name__})"
                )


class TestDirectConnection:
    """Baseline: verify mock server returns correct data without any router."""

    @pytest.mark.asyncio
    async def test_routed_experts_direct(self, backend_server):
        """routed_experts data is correct when hitting the backend directly."""
        client = RemoteInferenceClient(
            proxy_url=backend_server,
            server_urls=[backend_server],
        )
        try:
            result = await client.generate(
                {"prompt_token_ids": [list(range(PROMPT_LEN))], "sampling_params": {"max_tokens": RESPONSE_LEN}}
            )
            assert result["rollout_expert_indices"] is not None, (
                "rollout_expert_indices is None — RemoteInferenceClient.generate() "
                "did not propagate routed_experts from the server response"
            )
            _verify_routed_experts(result["rollout_expert_indices"][0], EXPECTED_ROUTED_EXPERTS)
        finally:
            await client.teardown()


class TestInferenceRouter:
    """Test routed_experts through the Python InferenceRouter."""

    @pytest.mark.asyncio
    async def test_routed_experts_through_python_router(self, backend_server):
        """routed_experts survives round-trip through InferenceRouter."""
        router_port = get_open_port()
        router = InferenceRouter(server_urls=[backend_server], port=router_port)
        router_url = router.start()
        try:
            client = RemoteInferenceClient(
                proxy_url=router_url,
                server_urls=[backend_server],
            )
            try:
                result = await client.generate(
                    {"prompt_token_ids": [list(range(PROMPT_LEN))], "sampling_params": {"max_tokens": RESPONSE_LEN}}
                )
                assert (
                    result["rollout_expert_indices"] is not None
                ), "rollout_expert_indices is None after InferenceRouter proxy"
                _verify_routed_experts(result["rollout_expert_indices"][0], EXPECTED_ROUTED_EXPERTS)
            finally:
                await client.teardown()
        finally:
            router.shutdown()


@pytest.mark.skipif(
    shutil.which("vllm-router") is None,
    reason="vllm-router binary not installed",
)
class TestVLLMRouter:
    """Test routed_experts through the Rust vllm-router binary."""

    def _make_router(self, backend_server: str):
        from skyrl.backends.skyrl_train.inference_servers.vllm_router import VLLMRouter

        return VLLMRouter(
            server_urls=[backend_server],
            port=get_open_port(),
            prometheus_port=get_open_port(),
        )

    @pytest.mark.asyncio
    async def test_routed_experts_through_vllm_router(self, backend_server):
        """routed_experts survives round-trip through vllm-router (Rust)."""
        router = self._make_router(backend_server)
        router_url = router.start()
        try:
            client = RemoteInferenceClient(
                proxy_url=router_url,
                server_urls=[backend_server],
            )
            try:
                result = await client.generate(
                    {"prompt_token_ids": [list(range(PROMPT_LEN))], "sampling_params": {"max_tokens": RESPONSE_LEN}}
                )
                assert result["rollout_expert_indices"] is not None, (
                    "rollout_expert_indices is None after VLLMRouter (Rust) proxy — "
                    "the vllm-router binary may not be forwarding /skyrl/v1/generate"
                )
                _verify_routed_experts(result["rollout_expert_indices"][0], EXPECTED_ROUTED_EXPERTS)
            finally:
                await client.teardown()
        finally:
            router.shutdown()

    @pytest.mark.asyncio
    async def test_routed_experts_batch(self, backend_server):
        """Multiple prompts in a batch all get correct routed_experts."""
        router = self._make_router(backend_server)
        router_url = router.start()
        batch_size = 4
        try:
            client = RemoteInferenceClient(
                proxy_url=router_url,
                server_urls=[backend_server],
            )
            try:
                result = await client.generate(
                    {
                        "prompt_token_ids": [list(range(PROMPT_LEN)) for _ in range(batch_size)],
                        "sampling_params": {"max_tokens": RESPONSE_LEN},
                    }
                )
                indices = result["rollout_expert_indices"]
                assert indices is not None, "rollout_expert_indices is None for batch"
                assert len(indices) == batch_size, f"batch size mismatch: got {len(indices)}, expected {batch_size}"
                for i in range(batch_size):
                    _verify_routed_experts(indices[i], EXPECTED_ROUTED_EXPERTS)
            finally:
                await client.teardown()
        finally:
            router.shutdown()

    @pytest.mark.asyncio
    async def test_token_ids_and_logprobs_preserved(self, backend_server):
        """token_ids and logprobs are also preserved alongside routed_experts."""
        router = self._make_router(backend_server)
        router_url = router.start()
        try:
            client = RemoteInferenceClient(
                proxy_url=router_url,
                server_urls=[backend_server],
            )
            try:
                result = await client.generate(
                    {
                        "prompt_token_ids": [list(range(PROMPT_LEN))],
                        "sampling_params": {"max_tokens": RESPONSE_LEN, "logprobs": 1},
                    }
                )
                assert result["response_ids"][0] == EXPECTED_TOKEN_IDS, "token_ids mismatch"
                expected_lp = [lp["logprob"] for lp in EXPECTED_LOGPROBS]
                assert result["response_logprobs"][0] == expected_lp, "logprobs mismatch"
            finally:
                await client.teardown()
        finally:
            router.shutdown()
