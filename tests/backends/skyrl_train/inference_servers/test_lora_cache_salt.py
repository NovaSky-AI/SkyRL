import asyncio

import orjson

from skyrl.backends.skyrl_train.inference_servers.lora_cache_salt import (
    LoraCacheSaltMiddleware,
)


def _post(loads, path, payload):
    seen = {}

    async def app(scope, receive, send):
        message = await receive()
        seen["body"] = orjson.loads(message["body"])
        seen["length"] = dict(scope["headers"])[b"content-length"]

    async def receive():
        body = orjson.dumps(payload)
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [(b"content-length", b"1")],
    }
    asyncio.run(LoraCacheSaltMiddleware(app, loads)(scope, receive, None))
    return seen


def test_each_adapter_reload_gets_a_new_cache_salt():
    loads = {"model_a": 1}
    first = _post(loads, "/v1/completions", {"model": "model_a", "prompt": [1, 2]})
    loads["model_a"] = 2
    second = _post(loads, "/v1/completions", {"model": "model_a", "prompt": [1, 2]})

    assert first["body"]["cache_salt"] == "model_a:1"
    assert second["body"]["cache_salt"] == "model_a:2"
    assert second["length"] == str(len(orjson.dumps(second["body"]))).encode()


def test_base_model_and_caller_salts_keep_their_keys():
    loads = {"model_a": 3}
    base = _post(loads, "/v1/completions", {"model": "base", "prompt": [1]})
    salted = _post(
        loads,
        "/skyrl/v1/generate",
        {"model": "model_a", "token_ids": [1], "cache_salt": "x"},
    )

    assert "cache_salt" not in base["body"]
    assert salted["body"]["cache_salt"] == "model_a:3:x"
