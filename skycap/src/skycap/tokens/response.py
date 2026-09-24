"""The OpenAI chat.completion a token-mode turn answers with, whole or as SSE.

The engine returns a whole completion, so a streamed answer is synthesized
from it after generation. The content is identical; only the timing differs.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Iterator
from typing import Any

import orjson


def finish_reason(engine_reason: str, message: dict[str, Any]) -> str:
    """The engine never says ``tool_calls``; a parsed tool call makes a stop one."""
    if engine_reason == "stop" and message.get("tool_calls"):
        return "tool_calls"
    return engine_reason


def completion(
    message: dict[str, Any], *, model: str | None, reason: str, prompt_tokens: int, completion_tokens: int
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{secrets.token_hex(12)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": reason, "logprobs": None}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def stream_frames(body: dict[str, Any]) -> Iterator[bytes]:
    choice = body["choices"][0]
    message = choice["message"]

    def frame(delta: dict[str, Any], reason: str | None = None, usage: dict[str, Any] | None = None) -> bytes:
        chunk: dict[str, Any] = {
            "id": body["id"],
            "object": "chat.completion.chunk",
            "created": body["created"],
            "model": body["model"],
            "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
        }
        if usage is not None:
            chunk["usage"] = usage
        return b"data: " + orjson.dumps(chunk) + b"\n\n"

    yield frame({"role": "assistant"})
    if message.get("reasoning_content"):
        yield frame({"reasoning_content": message["reasoning_content"]})
    if message.get("content"):
        yield frame({"content": message["content"]})
    for index, call in enumerate(message.get("tool_calls") or ()):
        yield frame({"tool_calls": [{"index": index, **call}]})
    yield frame({}, choice["finish_reason"], body["usage"])
    yield b"data: [DONE]\n\n"
