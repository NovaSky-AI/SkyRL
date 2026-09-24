"""A deterministic OpenAI-compatible upstream for tests.

The reply is ``"re: <last message content>"`` unless the request asks for
something else through a ``mock`` field in the body:

    {"reply": "text"}        the assistant content
    {"tool_call": "name"}    reply with one tool call instead
    {"status": 500}          fail with that status
    {"delay": 0.5}           sleep before answering
"""

from __future__ import annotations

import asyncio
import itertools
from typing import Any

import orjson
from aiohttp import web

API_KEY = "upstream-secret"


class MockOpenAI:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self._ids = itertools.count()

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_get("/v1/models", self.models)
        return app

    async def models(self, request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": "policy", "object": "model"}]})

    async def chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(body)
        self.headers.append(dict(request.headers))
        mock = body.get("mock") or {}
        if mock.get("delay"):
            await asyncio.sleep(mock["delay"])
        if request.headers.get("Authorization") != f"Bearer {API_KEY}":
            return web.json_response({"error": {"message": "bad key"}}, status=401)
        if mock.get("status"):
            return web.json_response({"error": {"message": "boom", "type": "api_error"}}, status=mock["status"])

        last = body["messages"][-1].get("content") if body["messages"] else ""
        message: dict[str, Any] = {"role": "assistant", "content": mock.get("reply", f"re: {last}")}
        finish = "stop"
        if mock.get("tool_call"):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": mock["tool_call"], "arguments": '{"q": "x"}'},
                    }
                ],
            }
            finish = "tool_calls"
        response_id = f"chatcmpl-{next(self._ids)}"
        usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        if not body.get("stream"):
            return web.json_response(
                {
                    "id": response_id,
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model"),
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": usage,
                }
            )
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for delta in _deltas(message):
            await response.write(_frame(response_id, body.get("model"), delta, None))
        await response.write(_frame(response_id, body.get("model"), {}, finish))
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response


def _deltas(message: dict[str, Any]) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = [{"role": "assistant"}]
    content = message.get("content")
    if content:
        middle = len(content) // 2
        deltas += [{"content": content[:middle]}, {"content": content[middle:]}]
    for index, call in enumerate(message.get("tool_calls") or ()):
        arguments = call["function"]["arguments"]
        deltas.append(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["function"]["name"], "arguments": arguments[:3]},
                    }
                ]
            }
        )
        deltas.append({"tool_calls": [{"index": index, "function": {"arguments": arguments[3:]}}]})
    return deltas


def _frame(response_id: str, model: str | None, delta: dict[str, Any], finish: str | None) -> bytes:
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return b"data: " + orjson.dumps(chunk) + b"\n\n"
