from argparse import Namespace
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (
    VLLMServerActor,
)


class _FakeEngine:
    def __init__(self) -> None:
        self.lora_request = None

    async def generate(self, prompt, sampling_params, request_id, lora_request=None):
        self.lora_request = lora_request
        yield SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[17],
                    finish_reason="stop",
                    logprobs=None,
                    routed_experts=np.array([[[1, 2]]], dtype=np.uint8),
                )
            ]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "uses_lora"), [("adapter_test", True), ("base_test", False)])
async def test_route_endpoint_resolves_lora_by_model(model, uses_lora):
    app = FastAPI()
    lora_request = object()
    app.state.openai_serving_models = SimpleNamespace(lora_requests={"adapter_test": lora_request})
    engine = _FakeEngine()
    VLLMServerActor._add_custom_endpoints(app, engine, Namespace())

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/skyrl/v1/generate",
            json={
                "model": model,
                "token_ids": [1, 2],
                "sampling_params": {"max_tokens": 1, "temperature": 0.0},
            },
        )

    assert response.status_code == 200
    assert engine.lora_request is (lora_request if uses_lora else None)
    assert response.json()["choices"][0]["routed_experts"] is not None
