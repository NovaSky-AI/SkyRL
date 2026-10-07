"""Salt each LoRA adapter's prefix cache with its load count.

vLLM keys LoRA prefix-cache blocks by adapter name, and SkyRL reloads an adapter in place under the
same name. Salting requests with the adapter's load count keeps a reloaded adapter from reusing KV
computed with its previous weights, without clearing the cache of every other adapter.
"""

from typing import Any, Awaitable, Callable

import orjson

SALTED_PATHS = ("/v1/completions", "/skyrl/v1/generate")

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class LoraCacheSaltMiddleware:
    def __init__(
        self,
        app: Callable[[Scope, Receive, Send], Awaitable[None]],
        loads: dict[str, int],
    ) -> None:
        self.app = app
        self.loads = loads

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in SALTED_PATHS or not self.loads:
            await self.app(scope, receive, send)
            return
        chunks = []
        while True:
            message = await receive()
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = salt_request_body(b"".join(chunks), self.loads)
        headers = [(name, value) for name, value in scope["headers"] if name != b"content-length"]
        headers.append((b"content-length", str(len(body)).encode()))
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app({**scope, "headers": headers}, replay, send)


def salt_request_body(body: bytes, loads: dict[str, int]) -> bytes:
    try:
        request = orjson.loads(body)
    except orjson.JSONDecodeError:
        return body
    if not isinstance(request, dict) or request.get("model") not in loads:
        return body
    salt = f"{request['model']}:{loads[request['model']]}"
    if request.get("cache_salt"):
        salt = f"{salt}:{request['cache_salt']}"
    return orjson.dumps({**request, "cache_salt": salt})
