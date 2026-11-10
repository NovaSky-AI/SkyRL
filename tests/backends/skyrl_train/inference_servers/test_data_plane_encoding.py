"""Data-plane request bodies are encoded with orjson: same JSON on the wire as stdlib json for the payloads
SkyRL sends, plus numpy arrays and non-string keys (e.g. a logit_bias keyed by token id)."""

import json

import numpy as np
import pytest
from aiohttp import web

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)


@pytest.mark.asyncio
async def test_post_body_and_headers():
    seen = {}

    async def handle(request: web.Request) -> web.Response:
        seen["content_type"] = request.headers.get("Content-Type")
        seen["custom"] = request.headers.get("X-Session-ID")
        seen["body"] = json.loads(await request.read())
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/inference/v1/generate", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    client = RemoteInferenceClient(proxy_url=url, server_urls=[url], model_name="m", data_parallel_size=1)
    payload = {
        "token_ids": list(range(1000)),
        "sampling_params": {"max_tokens": 8, "temperature": 0.7, "logit_bias": {5: -1.5}},
        "embeds": np.arange(4, dtype=np.int64),
        "model": "m",
    }
    try:
        out = await client._post(f"{url}/inference/v1/generate", json=payload, headers={"X-Session-ID": "s1"})
    finally:
        await client.aclose()
        await runner.cleanup()
    assert out == {"ok": True}
    assert seen["content_type"] == "application/json"
    assert seen["custom"] == "s1"
    assert seen["body"]["token_ids"] == list(range(1000))
    assert seen["body"]["sampling_params"] == {"max_tokens": 8, "temperature": 0.7, "logit_bias": {"5": -1.5}}
    assert seen["body"]["embeds"] == [0, 1, 2, 3]
