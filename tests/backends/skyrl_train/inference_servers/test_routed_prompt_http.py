"""The public routed endpoint keeps adapter identity, routes, and prompt scores together."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("vllm")

from vllm.lora.request import LoRARequest

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (
    VLLMServerActor,
)

pytestmark = pytest.mark.vllm


def make_app(score=-0.25):
    app = FastAPI()
    adapter = LoRARequest("adapter", 7, "/adapter")
    app.state.openai_serving_models = SimpleNamespace(
        lora_requests={"adapter": adapter}, is_base_model=lambda model: model == "base"
    )
    calls = []

    async def generate(prompt, sampling_params, request_id, lora_request):
        calls.append((prompt, sampling_params, lora_request))
        for value in [-9.0, score]:
            yield SimpleNamespace(
                prompt_logprobs=[None, {2: SimpleNamespace(logprob=value)}],
                outputs=[
                    SimpleNamespace(
                        token_ids=[3],
                        finish_reason="length",
                        logprobs=None,
                        routed_experts=np.full((2, 2, 1), 4 if value == score else 8, dtype=np.int32),
                    )
                ],
            )

    VLLMServerActor._add_custom_endpoints(app, SimpleNamespace(generate=generate), SimpleNamespace(enable_lora=True))
    return app, adapter, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["base", "adapter"])
async def test_routed_client_returns_same_request_scores_and_loaded_adapter(monkeypatch, model):
    app, adapter, calls = make_app()
    client = RemoteInferenceClient("http://testserver", ["http://testserver"], 1, enable_return_routed_experts=True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as http:

        async def post(url, **kwargs):
            response = await http.post(url, **kwargs)
            response.raise_for_status()
            return response.json()

        monkeypatch.setattr(client, "_post", post)
        monkeypatch.setattr(client, "detokenize", AsyncMock(return_value=["decoded"]))
        output = await client.generate(
            {
                "prompt_token_ids": [[1, 2]],
                "sampling_params": {"max_tokens": 1, "prompt_logprobs": 0, "routed_experts_prompt_start": 0},
            },
            model=model,
        )
    assert len(calls) == 1
    assert calls[0][2] is (adapter if model == "adapter" else None)
    assert calls[0][1].prompt_logprobs == 0
    assert output["prompt_logprobs"] == [[None, -0.25]]
    assert output["rollout_expert_indices"][0].tolist() == [[[4], [4]], [[4], [4]]]


def test_routed_endpoint_rejects_unknown_adapter_without_generation():
    app, _, calls = make_app()
    response = TestClient(app).post("/skyrl/v1/generate", json={"model": "unknown", "token_ids": [1, 2]})
    assert response.status_code == 404
    assert calls == []


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_prompt_diagnostic_never_clamps_nonfinite_scores(score):
    app, _, _ = make_app(score)
    response = TestClient(app).post(
        "/skyrl/v1/generate",
        json={
            "model": "adapter",
            "token_ids": [1, 2],
            "sampling_params": {"max_tokens": 1, "prompt_logprobs": 0},
        },
    )
    assert response.status_code == 500
    assert "nonfinite" in response.json()["detail"]
