import asyncio
import json
from argparse import Namespace

import pytest
from fastapi import FastAPI, HTTPException, Request

pytest.importorskip("vllm", reason="vllm_server_actor imports vllm at module scope")

from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (
    VLLMServerActor,
)

pytestmark = pytest.mark.vllm


@pytest.mark.asyncio
async def test_skyrl_generate_disconnect_cancels_engine_generation():
    class _Engine:
        def __init__(self):
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def generate(self, *args, **kwargs):
            self.started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            yield

    engine = _Engine()
    app = FastAPI()
    VLLMServerActor._add_custom_endpoints(app, engine, Namespace(enable_lora=False))
    endpoint = next(route.endpoint for route in app.routes if route.path == "/skyrl/v1/generate")

    messages = asyncio.Queue()
    messages.put_nowait(
        {
            "type": "http.request",
            "body": json.dumps({"token_ids": [1], "sampling_params": {}}).encode(),
            "more_body": False,
        }
    )
    request = Request(
        {"type": "http", "method": "POST", "path": "/skyrl/v1/generate", "headers": []},
        messages.get,
    )
    call = asyncio.create_task(endpoint(request))
    await asyncio.wait_for(engine.started.wait(), timeout=1)
    messages.put_nowait({"type": "http.disconnect"})

    with pytest.raises(HTTPException, match="Client disconnected") as exc_info:
        await asyncio.wait_for(call, timeout=1)
    assert exc_info.value.status_code == 499
    assert engine.cancelled.is_set()
