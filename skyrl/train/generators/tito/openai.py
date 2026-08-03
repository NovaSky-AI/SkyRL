"""OpenAI Chat Completions translation for the TITO proxy."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .types import Message, ToolSpec

_TRANSPORT_FIELDS = {"cache_salt", "session_id", "request_id"}
_SUPPORTED_FIELDS = {
    "cache_salt",
    "frequency_penalty",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "messages",
    "min_p",
    "min_tokens",
    "model",
    "n",
    "presence_penalty",
    "repetition_penalty",
    "request_id",
    "reasoning_effort",
    "response_format",
    "return_token_ids",
    "seed",
    "session_id",
    "stop",
    "stream",
    "temperature",
    "tool_choice",
    "tools",
    "top_k",
    "top_logprobs",
    "top_p",
}
_SAMPLING_FIELD_MAP = {
    "frequency_penalty": "frequency_penalty",
    "min_p": "min_p",
    "min_tokens": "min_tokens",
    "presence_penalty": "presence_penalty",
    "repetition_penalty": "repetition_penalty",
    "seed": "seed",
    "temperature": "temperature",
    "top_k": "top_k",
    "top_p": "top_p",
}


@dataclass(frozen=True)
class ParsedChatRequest:
    model: str
    messages: Tuple[Message, ...]
    tools: Optional[Tuple[ToolSpec, ...]]
    sampling_params: Dict[str, Any]
    return_logprobs: bool
    return_token_ids: bool
    request_key: str


def _canonical_request_key(body: Mapping[str, Any]) -> str:
    canonical = {key: value for key, value in body.items() if key not in _TRANSPORT_FIELDS}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_chat_request(
    body: Mapping[str, Any],
    *,
    registered_model: str,
    stop_token_ids: Sequence[int],
) -> ParsedChatRequest:
    """Validate the supported request subset and build engine sampling params."""
    unsupported = sorted(key for key in set(body) - _SUPPORTED_FIELDS if body[key] is not None)
    if unsupported:
        raise ValueError(f"Unsupported Chat Completions fields: {', '.join(unsupported)}")
    if body.get("stream", False):
        raise ValueError("Streaming Chat Completions are not supported")
    if body.get("n", 1) != 1:
        raise ValueError("Only n=1 is supported")
    if body.get("tool_choice") not in (None, "auto"):
        raise ValueError("Only tool_choice='auto' is supported")
    response_format = body.get("response_format")
    if response_format not in (None, {"type": "text"}):
        raise ValueError("Only response_format={'type': 'text'} is supported")
    if body.get("reasoning_effort") is not None:
        raise ValueError("reasoning_effort is not supported")
    if body.get("top_logprobs") not in (None, 0):
        raise ValueError("top_logprobs is not supported")

    model = body.get("model") or registered_model
    if model != registered_model:
        raise ValueError(f"Request model {model!r} does not match registered model {registered_model!r}")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("Every message must be an object")

    tools = body.get("tools")
    if tools is not None and (not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools)):
        raise ValueError("tools must be a list of objects")

    max_tokens = body.get("max_tokens")
    max_completion_tokens = body.get("max_completion_tokens")
    if max_tokens is not None and max_completion_tokens is not None and max_tokens != max_completion_tokens:
        raise ValueError("max_tokens and max_completion_tokens must match when both are provided")

    sampling_params: Dict[str, Any] = {}
    for request_field, sampling_field in _SAMPLING_FIELD_MAP.items():
        value = body.get(request_field)
        if value is not None:
            sampling_params[sampling_field] = value
    resolved_max_tokens = max_completion_tokens if max_completion_tokens is not None else max_tokens
    if resolved_max_tokens is not None:
        sampling_params["max_tokens"] = resolved_max_tokens
    if body.get("stop") is not None:
        sampling_params["stop"] = body["stop"]
    if stop_token_ids:
        sampling_params["stop_token_ids"] = list(dict.fromkeys(int(token_id) for token_id in stop_token_ids))

    # SkyRL always needs selected-token rollout logprobs for training.
    sampling_params["logprobs"] = 1
    return ParsedChatRequest(
        model=model,
        messages=tuple(messages),
        tools=tuple(tools) if tools is not None else None,
        sampling_params=sampling_params,
        return_logprobs=bool(body.get("logprobs", False)),
        return_token_ids=bool(body.get("return_token_ids", False)),
        request_key=_canonical_request_key(body),
    )


def build_chat_response(
    *,
    request_id: str,
    model: str,
    assistant_message: Message,
    prompt_token_ids: Sequence[int],
    completion_ids: Sequence[int],
    completion_logprobs: Sequence[float],
    finish_reason: str,
    return_logprobs: bool,
    return_token_ids: bool,
    decode_token,
    provider_extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct the supported OpenAI response plus optional token-ID extensions."""
    if finish_reason in ("length", "max_tokens"):
        openai_finish_reason = "length"
    elif assistant_message.get("tool_calls"):
        openai_finish_reason = "tool_calls"
    else:
        openai_finish_reason = "stop"

    choice: Dict[str, Any] = {
        "index": 0,
        "message": assistant_message,
        "finish_reason": openai_finish_reason,
        "logprobs": None,
    }
    if return_logprobs:
        content: List[Dict[str, Any]] = []
        for token_id, logprob in zip(completion_ids, completion_logprobs):
            token = decode_token(token_id)
            content.append(
                {
                    "token": token,
                    "logprob": float(logprob),
                    "bytes": list(token.encode("utf-8")),
                    "top_logprobs": [],
                    "token_id": int(token_id),
                }
            )
        choice["logprobs"] = {"content": content}
    if return_token_ids:
        choice["token_ids"] = list(completion_ids)
    if provider_extra:
        choice.update(provider_extra)

    response: Dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
        "usage": {
            "prompt_tokens": len(prompt_token_ids),
            "completion_tokens": len(completion_ids),
            "total_tokens": len(prompt_token_ids) + len(completion_ids),
        },
    }
    if return_token_ids:
        response["prompt_token_ids"] = list(prompt_token_ids)
    return response
