"""Worker metadata for external servers using the native vLLM HTTP app."""

import httpx
import pytest

pytest.importorskip("vllm")

from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (
    VLLMServerActor,
    _build_standalone_cli_args,
)

pytestmark = pytest.mark.vllm


@pytest.mark.asyncio
async def test_external_worker_metadata_is_not_captured_by_prometheus_mount():
    from vllm.entrypoints.openai.api_server import build_app

    args = _build_standalone_cli_args(["--model", "unused"])
    app = build_app(args, supported_tasks=("generate",))
    info = {"worker_id": "frontend", "role": "prefill"}
    VLLMServerActor._add_custom_endpoints(app, None, args, metrics_info=info)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/get_metrics_worker_info")
    assert response.status_code == 200
    assert response.json() == info
