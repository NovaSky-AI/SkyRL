import asyncio
from argparse import Namespace
from collections import defaultdict
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

pytest.importorskip("vllm", reason="LoRA NCCL control plane extends the vLLM server")
pytestmark = pytest.mark.vllm

from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (  # noqa: E402
    VLLMServerActor,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport import (  # noqa: E402
    LoRAUpdateRequest,
)


def _add_custom_endpoints(app, engine):
    gate = VLLMServerActor._prepare_lora_transport_admission(Namespace(middleware=[]))
    VLLMServerActor._bind_lora_transport_admission(app, gate)
    VLLMServerActor._add_custom_endpoints(
        app,
        engine,
        Namespace(),
        lora_transport_admission_gate=gate,
    )


def _request(generation):
    return LoRAUpdateRequest(
        adapter_name="adapter",
        generation=generation,
        layout_digest="a" * 64,
        source_dtype="float32",
    )


def _models():
    adapter_ids = iter(range(10, 20))
    return SimpleNamespace(
        lora_requests={},
        lora_resolver_lock=defaultdict(asyncio.Lock),
        lora_id_counter=SimpleNamespace(inc=lambda amount: next(adapter_ids)),
    )


@pytest.mark.asyncio
async def test_nccl_stage_reuses_atomic_activation_and_commit():
    app = FastAPI()
    calls = []

    async def collective_rpc(method, kwargs):
        calls.append((method, kwargs))

    _add_custom_endpoints(app, SimpleNamespace(collective_rpc=collective_rpc))
    models = _models()
    app.state.openai_serving_models = models
    body = {
        "lora_name": "adapter",
        "request": _request(3).to_json_dict(),
        "transport": "nccl",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app),
        base_url="http://example.com",
    ) as client:
        response = await client.post(
            "/skyrl/v1/stage_lora_nccl_adapter",
            json=body,
        )
        assert response.status_code == 200
        adapter_id = response.json()["lora_int_id"]
        transaction = {**body, "adapter_id": adapter_id}
        assert (
            await client.post(
                "/skyrl/v1/activate_lora_transport_adapter",
                json=transaction,
            )
        ).status_code == 200
        assert (
            await client.post(
                "/skyrl/v1/commit_lora_transport_adapter",
                json=transaction,
            )
        ).status_code == 200

    assert models.lora_requests["adapter"].lora_int_id == adapter_id
    assert models.lora_requests["adapter"].lora_path == "lora_nccl://adapter"
    assert [method for method, _ in calls] == [
        "stage_lora_nccl_adapter",
        "activate_lora_transport_adapter",
    ]


@pytest.mark.asyncio
async def test_failed_nccl_stage_discards_partial_generation_without_a_route():
    app = FastAPI()
    calls = []

    async def collective_rpc(method, kwargs):
        calls.append((method, kwargs))
        if method == "stage_lora_nccl_adapter":
            raise RuntimeError("injected receive failure")

    _add_custom_endpoints(app, SimpleNamespace(collective_rpc=collective_rpc))
    models = _models()
    app.state.openai_serving_models = models
    body = {
        "lora_name": "adapter",
        "request": _request(3).to_json_dict(),
        "transport": "nccl",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app),
        base_url="http://example.com",
    ) as client:
        response = await client.post(
            "/skyrl/v1/stage_lora_nccl_adapter",
            json=body,
        )

    assert response.status_code == 500
    assert "adapter" not in models.lora_requests
    assert [method for method, _ in calls] == [
        "stage_lora_nccl_adapter",
        "discard_lora_transport_adapter",
    ]
    assert models._skyrl_lora_transport_lifecycle._staged == {}
