"""The OpenAI Chat Completions wire: what capture reads out of a request and a reply.

Only v1's dialect. Parsing is shallow and never reaches the forward path: a
body that can't be read here is still forwarded, and the call is recorded as
a failure instead of a graph change.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import orjson

#: Request fields recorded on a call as its sampling settings.
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "max_completion_tokens",
    "seed",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
)


@dataclass(slots=True)
class ChatRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    model: str | None
    stream: bool
    sampling: dict[str, Any]
    body: dict[str, Any]


class RequestError(ValueError):
    """A request capture refuses before forwarding it."""


def parse_request(body: Any) -> ChatRequest:
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object")
    messages = body.get("messages")
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise RequestError("`messages` must be a list of objects")
    if body.get("n", 1) != 1:
        raise RequestError("`n` > 1 is not supported: one call produces one sample")
    tools = body.get("tools")
    return ChatRequest(
        messages=messages,
        tools=tools if isinstance(tools, list) and tools else None,
        model=body.get("model"),
        stream=bool(body.get("stream")),
        sampling={key: body[key] for key in SAMPLING_KEYS if body.get(key) is not None},
        body=body,
    )


@dataclass(slots=True)
class ChatReply:
    message: dict[str, Any]
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


def parse_response(body: Any) -> ChatReply:
    """The first choice's message from a non-streamed chat.completion."""
    if not isinstance(body, dict):
        raise ValueError("response is not a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("response choice has no message")
    return ChatReply(message=message, finish_reason=choices[0].get("finish_reason"), usage=body.get("usage"))


@dataclass(slots=True)
class StreamAssembler:
    """Rebuilds the assistant message a chat.completion.chunk stream carried."""

    role: str = "assistant"
    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    saw_done: bool = False
    _buffer: bytes = b""

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        while True:
            end = _event_end(self._buffer)
            if end is None:
                return
            event, self._buffer = self._buffer[: end[0]], self._buffer[end[1] :]
            self._event(event)

    def _event(self, event: bytes) -> None:
        for line in event.splitlines():
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                self.saw_done = True
                continue
            try:
                payload = orjson.loads(data)
            except orjson.JSONDecodeError:
                continue
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

    def reply(self) -> ChatReply:
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
        return ChatReply(message=message, finish_reason=self.finish_reason, usage=self.usage)


def assemble(chunks: Iterable[bytes]) -> ChatReply:
    assembler = StreamAssembler()
    for chunk in chunks:
        assembler.feed(chunk)
    assembler.feed(b"\n\n")
    return assembler.reply()


def _event_end(buffer: bytes) -> tuple[int, int] | None:
    """Where the first complete SSE event ends: (end of event, start of next)."""
    best: tuple[int, int] | None = None
    for separator in (b"\n\n", b"\r\n\r\n"):
        index = buffer.find(separator)
        if index != -1 and (best is None or index < best[0]):
            best = (index, index + len(separator))
    return best


def error_body(message: str, *, kind: str = "invalid_request_error", code: str | None = None) -> bytes:
    return orjson.dumps({"error": {"message": message, "type": kind, "param": None, "code": code}})
