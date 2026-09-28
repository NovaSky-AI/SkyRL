"""OpenAI Chat Completions protocol for the tokens route.

The external client keeps speaking text-space OpenAI. Only the proxy knows the
request became token IDs, so the request must be validated and the response
rebuilt faithfully enough that an unchanged SDK cannot tell the difference.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import orjson

from skyrl_capture.tito.types import Message, ToolSpec


class ProtocolError(Exception):
    """A client-visible protocol error, carrying its own HTTP status and body."""

    def __init__(self, status: int, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = {
            "error": {
                "message": message,
                "type": "invalid_request_error" if status < 500 else "api_error",
                "param": None,
                "code": code,
            }
        }


# Sampling fields forwarded to the token-in/token-out endpoint.
_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "seed",
    "stop",
    "logit_bias",
)

#: Everything capture reads off a chat request itself. A field outside both
#: this set and `_SAMPLING_FIELDS` is one capture has no opinion about, and is
#: offered to the upstream protocol rather than dropped -- see `extras`.
_CONSUMED_FIELDS = frozenset(
    {
        "messages",
        "model",
        "tools",
        "tool_choice",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "logprobs",
        "top_logprobs",
        "n",
    }
)


@dataclass
class ChatRequest:
    model: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] | None
    stream: bool
    max_tokens: int | None
    logprobs: bool
    top_logprobs: int | None
    sampling: dict[str, Any] = field(default_factory=dict)
    #: Request fields capture neither consumes nor recognises as sampling.
    #:
    #: An engine behind capture may accept parameters capture has never heard
    #: of, and the caller is the only one who knows it wants them. Naming them
    #: here would make capture the place every upstream's vocabulary
    #: accumulates; dropping them would make those upstreams unreachable
    #: through capture at all. So they are carried, unread, and the protocol
    #: for that upstream decides what any of them mean.
    #:
    #: A protocol that does not recognise one drops it: `VLLMTitoProtocol`
    #: filters against what `SamplingParams` accepts, which is what keeps an
    #: unknown key from reaching an engine that would reject the request.
    extras: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def parse_chat_request(body: bytes, *, registered_model: str | None) -> ChatRequest:
    try:
        payload = orjson.loads(body)
    except orjson.JSONDecodeError as error:
        raise ProtocolError(400, f"request body must be valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ProtocolError(400, "request body must be a JSON object")

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ProtocolError(400, "messages must be a non-empty array")
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ProtocolError(400, "each message must be an object with a string role")

    if payload.get("n") not in (None, 1):
        # One turn commits exactly one sampled assistant node; n>1 would need
        # n sibling nodes and a defined ordering between them.
        raise ProtocolError(400, "n>1 is not supported on the tokens route")

    tools = payload.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise ProtocolError(400, "tools must be an array")

    model = payload.get("model") or registered_model
    if not isinstance(model, str) or not model:
        raise ProtocolError(400, "model is required")

    sampling = {key: payload[key] for key in _SAMPLING_FIELDS if key in payload}
    extras = {
        key: value
        for key, value in payload.items()
        if key not in _CONSUMED_FIELDS and key not in sampling
    }
    return ChatRequest(
        model=model,
        messages=tuple(messages),
        tools=tuple(tools) if tools else None,
        stream=bool(payload.get("stream")),
        max_tokens=payload.get("max_completion_tokens") or payload.get("max_tokens"),
        logprobs=bool(payload.get("logprobs")),
        top_logprobs=payload.get("top_logprobs"),
        sampling=sampling,
        extras=extras,
        raw=payload,
    )


_FINISH_REASONS = {
    "stop": "stop",
    "eos": "stop",
    "length": "length",
    "context_length": "length",
    "max_tokens": "length",
    "tool_call": "tool_calls",
    "abort": "stop",
    "error": "stop",
}


def finish_reason_for(stop_reason: str, *, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    return _FINISH_REASONS.get(stop_reason, "stop")


def build_chat_response(
    *,
    request: ChatRequest,
    assistant_message: Message,
    prompt_token_ids: Sequence[int],
    completion_ids: Sequence[int],
    completion_logprobs: Sequence[float],
    stop_reason: str,
    decode_token: Any,
    response_id: str,
) -> dict[str, Any]:
    """Assemble an OpenAI-shaped response from exact token output."""
    has_tool_calls = bool(assistant_message.get("tool_calls"))
    choice: dict[str, Any] = {
        "index": 0,
        "message": assistant_message,
        "finish_reason": finish_reason_for(stop_reason, has_tool_calls=has_tool_calls),
    }
    if request.logprobs:
        choice["logprobs"] = {
            "content": [
                {
                    "token": decode_token(token_id),
                    "logprob": float(logprob),
                    "bytes": list(decode_token(token_id).encode()),
                    "top_logprobs": [],
                }
                for token_id, logprob in zip(completion_ids, completion_logprobs, strict=True)
            ]
        }
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [choice],
        "usage": {
            "prompt_tokens": len(prompt_token_ids),
            "completion_tokens": len(completion_ids),
            "total_tokens": len(prompt_token_ids) + len(completion_ids),
        },
        # Surfaced so a tokens client can see the token identity it was served
        # without needing the capture API.
        "capture": {"tokens": True, "stop_reason": stop_reason},
    }


def build_stream_frames(response: dict[str, Any], *, decode_token: Any, completion_ids: Sequence[int]) -> list[bytes]:
    """Synthesize an SSE stream from a completed response.

    The token-in/token-out endpoint returns a whole completion, so a client that
    asked for ``stream: true`` is served chunks reconstructed from the exact
    sampled tokens after generation finished. The token content is identical to
    the non-streaming path; only the arrival timing differs, and the exchange
    records that the stream was synthesized.
    """
    identifier = response["id"]
    model = response["model"]
    created = response["created"]
    message = response["choices"][0]["message"]
    frames: list[bytes] = []

    def frame(delta: dict[str, Any], finish: str | None = None) -> bytes:
        payload = {
            "id": identifier,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return b"data: " + orjson.dumps(payload) + b"\n\n"

    frames.append(frame({"role": "assistant"}))
    if message.get("content"):
        for token_id in completion_ids:
            piece = decode_token(token_id)
            if piece:
                frames.append(frame({"content": piece}))
    if message.get("tool_calls"):
        frames.append(frame({"tool_calls": message["tool_calls"]}))
    final = {
        "id": identifier,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": response["choices"][0]["finish_reason"]}],
        "usage": response["usage"],
    }
    frames.append(b"data: " + orjson.dumps(final) + b"\n\n")
    frames.append(b"data: [DONE]\n\n")
    return frames
