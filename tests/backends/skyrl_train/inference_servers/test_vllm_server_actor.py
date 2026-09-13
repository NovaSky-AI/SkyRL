import asyncio
import hashlib
import tempfile
import uuid
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

pytest.importorskip("vllm", reason="vLLM server actor tests require the vLLM extra")

from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (
    _LORA_UPLOAD_MAX_BYTES,
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
@pytest.mark.parametrize(
    ("model", "status", "uses_lora"),
    [("adapter_test", 200, True), ("base_test", 200, False), ("missing_test", 404, False)],
)
async def test_route_endpoint_resolves_lora_by_model(model, status, uses_lora):
    app = FastAPI()
    lora_request = object()
    app.state.openai_serving_models = SimpleNamespace(lora_requests={"adapter_test": lora_request})
    engine = _FakeEngine()
    VLLMServerActor._add_custom_endpoints(app, engine, Namespace(model="base_test"))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/skyrl/v1/generate",
            json={
                "model": model,
                "token_ids": [1, 2],
                "sampling_params": {"max_tokens": 1, "temperature": 0.0},
            },
        )

    assert response.status_code == status
    assert engine.lora_request is (lora_request if uses_lora else None)
    if status == 200:
        assert response.json()["choices"][0]["routed_experts"] is not None


@pytest.mark.asyncio
async def test_lora_upload_rejects_bad_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    app = FastAPI()
    VLLMServerActor._add_custom_endpoints(app, _FakeEngine(), Namespace())
    upload_id = str(uuid.uuid4())
    content = b"adapter-bytes"

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.put(
            f"/skyrl/v1/lora-adapters/{upload_id}/adapter_model.safetensors",
            content=content,
            headers={"X-SkyRL-SHA256": hashlib.sha256(content).hexdigest(), "X-SkyRL-File-Size": str(len(content))},
        )
        bad_content = b"config"
        bad = await client.put(
            f"/skyrl/v1/lora-adapters/{upload_id}/adapter_config.json",
            content=bad_content,
            headers={"X-SkyRL-SHA256": "not-the-file-sha", "X-SkyRL-File-Size": str(len(bad_content))},
        )

    assert response.status_code == 200
    assert bad.status_code == 400
    assert not (tmp_path / "skyrl_lora_uploads" / upload_id).exists()


@pytest.mark.asyncio
async def test_lora_upload_rejects_oversized_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    app = FastAPI()
    VLLMServerActor._add_custom_endpoints(app, _FakeEngine(), Namespace())
    upload_id = str(uuid.uuid4())

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.put(
            f"/skyrl/v1/lora-adapters/{upload_id}/adapter_model.safetensors",
            content=b"",
            headers={
                "X-SkyRL-SHA256": hashlib.sha256(b"").hexdigest(),
                "X-SkyRL-File-Size": str(_LORA_UPLOAD_MAX_BYTES + 1),
            },
        )

    assert response.status_code == 413
    assert not (tmp_path / "skyrl_lora_uploads" / upload_id).exists()


class _FakeLoraEngineClient:
    def __init__(self) -> None:
        self.adapter_bytes = None

    async def add_lora(self, request) -> None:
        self.adapter_bytes = Path(request.lora_path, "adapter_model.safetensors").read_bytes()


@pytest.mark.asyncio
async def test_lora_upload_loads_verified_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    engine_client = _FakeLoraEngineClient()
    app = FastAPI()
    app.state.openai_serving_models = SimpleNamespace(
        lora_requests={},
        lora_resolver_lock=defaultdict(asyncio.Lock),
        lora_id_counter=SimpleNamespace(inc=lambda _: 1),
        engine_client=engine_client,
    )
    VLLMServerActor._add_custom_endpoints(app, _FakeEngine(), Namespace())
    upload_id = str(uuid.uuid4())
    files = {
        "adapter_model.safetensors": b"adapter-bytes",
        "adapter_config.json": b'{"r": 32}',
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for filename, content in files.items():
            response = await client.put(
                f"/skyrl/v1/lora-adapters/{upload_id}/{filename}",
                content=content,
                headers={"X-SkyRL-SHA256": hashlib.sha256(content).hexdigest(), "X-SkyRL-File-Size": str(len(content))},
            )
            assert response.status_code == 200
        response = await client.post(
            "/skyrl/v1/load_lora_adapter",
            json={"lora_name": "adapter_test", "upload_id": upload_id},
        )

    assert response.status_code == 200
    assert engine_client.adapter_bytes == files["adapter_model.safetensors"]
    assert app.state.openai_serving_models.lora_requests["adapter_test"].load_inplace is False
    assert not (tmp_path / "skyrl_lora_uploads" / upload_id).exists()
