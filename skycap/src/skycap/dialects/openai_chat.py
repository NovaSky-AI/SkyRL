"""OpenAI Chat Completions: what capture reads out of a request and an upstream reply, and the
chat.completion a token-mode turn answers with.

A chat request already carries chat-template turns and a reply is one assistant
message, so nothing is converted either way. Parsing is shallow and never
reaches the forward path: a body that can't be read here is still forwarded,
and the call is recorded as a failure instead of a graph change.

In token mode the engine returns a whole completion, so a streamed answer is
synthesized from it after generation. The content is identical; only the timing
differs.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import orjson

from skycap.dialects.base import (
    SAMPLING_KEYS,
    Assembler,
    CallRequest,
    Dialect,
    DialectType,
    Reply,
    RequestError,
    Turn,
    sse_data,
)


class ChatDialect(Dialect):
    type = DialectType.OPENAI_CHAT
    path = "/chat/completions"
    raw_content = True

    def parse_request(self, body: Any) -> CallRequest:
        return parse_request(body)

    def reply(self, call: CallRequest, turn: Turn) -> dict[str, Any]:
        # vLLM with no reasoning parser still sends `reasoning_content: null`, and clients rely on
        # it: LiteLLM splits `<think>` out of `content` only when the field is absent.
        message = {**turn.message, "reasoning_content": None} if turn.raw_content else turn.message
        return completion(
            message,
            model=turn.model,
            reason=turn.finish_reason,
            prompt_tokens=turn.prompt_tokens,
            completion_tokens=turn.completion_tokens,
        )

    def stream(self, reply: dict[str, Any]) -> Iterator[bytes]:
        return stream_frames(reply)

    def parse_reply(self, body: Any) -> Reply:
        return parse_response(body)

    def assembler(self) -> Assembler:
        return StreamAssembler()


# -- the request and an upstream's reply (text mode) -------------------------------------------
def parse_request(body: Any) -> CallRequest:
    """The call: its messages are already chat-template turns."""
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object")
    messages = body.get("messages")
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise RequestError("`messages` must be a list of objects")
    if body.get("n", 1) != 1:
        raise RequestError("`n` > 1 is not supported: one call produces one sample")
    tools = body.get("tools")
    return CallRequest(
        messages=messages,
        tools=tools if isinstance(tools, list) and tools else None,
        model=body.get("model"),
        stream=bool(body.get("stream")),
        sampling={key: body[key] for key in SAMPLING_KEYS if body.get(key) is not None},
        body=body,
    )


def parse_response(body: Any) -> Reply:
    """The first choice's message from a non-streamed chat.completion."""
    if not isinstance(body, dict):
        raise ValueError("response is not a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("response choice has no message")
    return Reply(message=message, finish_reason=choices[0].get("finish_reason"), usage=body.get("usage"))


@dataclass(slots=True)
class StreamAssembler(Assembler):
    """Rebuilds the assistant message a chat.completion.chunk stream carried."""

    role: str = "assistant"
    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    saw_done: bool = False
    _buffer: bytes = b""

    @property
    def ending(self) -> bool:
        return self.finish_reason is not None

    @property
    def done(self) -> bool:
        return self.saw_done

    def event(self, event: bytes) -> None:
        data = sse_data(event)
        if data is None:
            return
        if data.strip() == b"[DONE]":
            self.saw_done = True
            return
        try:
            payload = orjson.loads(data)
        except orjson.JSONDecodeError:
            return
        if isinstance(payload, dict):
            self._payload(payload)

    def _payload(self, payload: dict[str, Any]) -> None:
        if isinstance(payload.get("usage"), dict):
            self.usage = payload["usage"]
        for choice in payload.get("choices") or ():
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            if delta.get("role"):
                self.role = delta["role"]
            if isinstance(delta.get("content"), str):
                self.content.append(delta["content"])
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(reasoning, str):
                self.reasoning.append(reasoning)
            for call in delta.get("tool_calls") or ():
                if isinstance(call, dict):
                    self._tool_call(call)

    def _tool_call(self, call: dict[str, Any]) -> None:
        slot = self.tool_calls.setdefault(
            int(call.get("index") or 0),
            {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if call.get("id"):
            slot["id"] = call["id"]
        function = call.get("function")
        if isinstance(function, dict):
            if function.get("name"):
                slot["function"]["name"] += function["name"]
            if function.get("arguments"):
                slot["function"]["arguments"] += function["arguments"]

    def reply(self) -> Reply:
        if self.finish_reason is None:
            raise ValueError("stream ended without a finish_reason")
        message: dict[str, Any] = {
            "role": self.role,
            "content": "".join(self.content) if self.content else None,
        }
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        if self.tool_calls:
            message["tool_calls"] = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        return Reply(message=message, finish_reason=self.finish_reason, usage=self.usage)


# -- the reply to a sampled turn (token mode) ---------------------------------------------
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
