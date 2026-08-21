"""vLLM-owned OpenAI protocol handling for the TITO proxy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, cast

import msgspec
from pydantic import ValidationError
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens

from .types import Message, ToolSpec

_TRANSPORT_FIELDS = {"cache_salt", "request_id", "session_id"}
_SUPPORTED_EXTRA_FIELDS = {"session_id"}


class OpenAIProtocolError(Exception):
    """OpenAI-compatible request error with a vLLM error body."""

    def __init__(self, message: str, status_code: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status_code = status_code
        self.body = ErrorResponse(
            error=ErrorInfo(
                message=message,
                type=HTTPStatus(status_code).phrase,
                code=status_code,
            )
        ).model_dump(mode="json")


@dataclass(frozen=True)
class ParsedChatRequest:
    request: ChatCompletionRequest
    messages: Tuple[Message, ...]
    tools: Optional[Tuple[ToolSpec, ...]]
    return_logprobs: bool
    return_token_ids: bool
    request_key: str

    @property
    def model(self) -> str:
        return cast(str, self.request.model)


def _canonical_request_key(request: ChatCompletionRequest) -> str:
    canonical = request.model_dump(mode="json", exclude_none=True)
    for field in _TRANSPORT_FIELDS:
        canonical.pop(field, None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _message_dict(message: Any) -> Dict[str, Any]:
    if isinstance(message, Mapping):
        return dict(message)
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json", exclude_none=True)
    raise OpenAIProtocolError(f"Unsupported chat message type: {type(message).__name__}")


def _renderer_message(message: Any) -> Message:
    normalized = _message_dict(message)
    reasoning = normalized.pop("reasoning", None)
    if reasoning is not None:
        normalized["reasoning_content"] = reasoning
    return normalized


def _validate_tito_request(request: ChatCompletionRequest) -> None:
    if request.stream:
        raise OpenAIProtocolError("Streaming Chat Completions are not supported")
    if request.n != 1:
        raise OpenAIProtocolError("Only n=1 is supported")
    if request.use_beam_search:
        raise OpenAIProtocolError("Beam search is not supported")
    if request.tools and request.tool_choice != "auto":
        raise OpenAIProtocolError("Only tool_choice='auto' is supported")
    if not request.tools and request.tool_choice not in (None, "none", "auto"):
        raise OpenAIProtocolError("Only automatic tool choice is supported")
    if request.reasoning_effort is not None:
        raise OpenAIProtocolError("reasoning_effort is not supported")
    if request.top_logprobs not in (None, 0):
        raise OpenAIProtocolError("top_logprobs is not supported")
    if request.structured_outputs is not None:
        raise OpenAIProtocolError("structured_outputs is not supported")
    if request.chat_template is not None or request.chat_template_kwargs is not None:
        raise OpenAIProtocolError("Per-request chat templates are not supported")
    if request.documents is not None:
        raise OpenAIProtocolError("documents are not supported")
    if request.truncate_prompt_tokens is not None:
        raise OpenAIProtocolError("truncate_prompt_tokens is not supported")
    if not request.add_generation_prompt or request.continue_final_message or request.add_special_tokens:
        raise OpenAIProtocolError("Custom chat-template control fields are not supported")
    if request.echo or request.prompt_logprobs is not None:
        raise OpenAIProtocolError("Prompt echo and prompt logprobs are not supported")

    response_format = request.response_format
    if response_format is not None and getattr(response_format, "type", None) != "text":
        raise OpenAIProtocolError("Only response_format={'type': 'text'} is supported")

    unsupported_extra = set(request.model_extra or {}) - _SUPPORTED_EXTRA_FIELDS
    if unsupported_extra:
        raise OpenAIProtocolError(f"Unsupported Chat Completions fields: {', '.join(sorted(unsupported_extra))}")

    for message in request.messages:
        content = _message_dict(message).get("content")
        if isinstance(content, list):
            part_types = {part.get("type") for part in content if isinstance(part, Mapping)}
            if not part_types.issubset({"text"}):
                raise OpenAIProtocolError("Multimodal message content is not supported")


def parse_chat_request(body: Mapping[str, Any], *, registered_model: str) -> ParsedChatRequest:
    """Validate through vLLM, then enforce only TITO-specific restrictions."""
    try:
        request = ChatCompletionRequest.model_validate(body)
    except ValidationError as exc:
        raise OpenAIProtocolError(str(exc)) from exc

    if request.model is None:
        raise OpenAIProtocolError("model is required")
    if request.model != registered_model:
        raise OpenAIProtocolError(
            f"Request model {request.model!r} does not match registered model {registered_model!r}"
        )
    _validate_tito_request(request)

    messages = tuple(_renderer_message(message) for message in request.messages)
    tools = tuple(tool.model_dump(mode="json", exclude_none=True) for tool in request.tools) if request.tools else None
    return ParsedChatRequest(
        request=request,
        messages=messages,
        tools=tools,
        return_logprobs=bool(request.logprobs),
        return_token_ids=bool(request.return_token_ids),
        request_key=_canonical_request_key(request),
    )


def build_sampling_params(
    parsed: ParsedChatRequest,
    *,
    prompt_token_count: int,
    max_model_len: int,
    renderer_stop_token_ids: Sequence[int],
    default_max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Use vLLM's request conversion and apply TITO bookkeeping overrides."""
    request = parsed.request
    requested_max_tokens = request.max_completion_tokens
    if requested_max_tokens is None:
        requested_max_tokens = request.__dict__.get("max_tokens")
    default_sampling_params = {"max_tokens": default_max_tokens} if default_max_tokens is not None else {}
    try:
        max_tokens = get_max_tokens(
            max_model_len,
            requested_max_tokens,
            prompt_token_count,
            default_sampling_params=default_sampling_params,
        )
        sampling = request.to_sampling_params(
            max_tokens,
            default_sampling_params=default_sampling_params,
        )
    except ValueError as exc:
        raise OpenAIProtocolError(str(exc)) from exc

    sampling_params = msgspec.to_builtins(sampling)
    sampling_params["logprobs"] = 1
    sampling_params["stop_token_ids"] = list(
        dict.fromkeys([*(sampling_params.get("stop_token_ids") or []), *map(int, renderer_stop_token_ids)])
    )
    return sampling_params


def _chat_message(assistant_message: Message, *, include_reasoning: bool) -> ChatMessage:
    tool_calls = []
    for tool_call in assistant_message.get("tool_calls") or []:
        function = tool_call["function"]
        tool_calls.append(
            ToolCall(
                id=tool_call["id"],
                function=FunctionCall(
                    name=function["name"],
                    arguments=function["arguments"],
                ),
            )
        )

    reasoning = assistant_message.get("reasoning_content") if include_reasoning else None
    return ChatMessage(
        role=assistant_message.get("role", "assistant"),
        content=assistant_message.get("content"),
        reasoning=reasoning,
        tool_calls=tool_calls,
    )


def build_chat_response(
    *,
    parsed: ParsedChatRequest,
    assistant_message: Message,
    prompt_token_ids: Sequence[int],
    completion_ids: Sequence[int],
    completion_logprobs: Sequence[float],
    finish_reason: str,
    decode_token,
) -> Dict[str, Any]:
    """Construct a typed vLLM response."""
    if finish_reason in ("length", "max_tokens"):
        openai_finish_reason = "length"
    elif assistant_message.get("tool_calls"):
        openai_finish_reason = "tool_calls"
    else:
        openai_finish_reason = "stop"

    logprobs = None
    if parsed.return_logprobs:
        logprobs = ChatCompletionLogProbs(
            content=[
                ChatCompletionLogProbsContent.model_validate(
                    {
                        "token": token,
                        "logprob": float(logprob),
                        "bytes": list(token.encode("utf-8")),
                        "top_logprobs": [],
                    }
                )
                for token_id, logprob in zip(completion_ids, completion_logprobs)
                for token in [decode_token(token_id)]
            ]
        )

    choice = ChatCompletionResponseChoice(
        index=0,
        message=_chat_message(
            assistant_message,
            include_reasoning=parsed.request.include_reasoning,
        ),
        finish_reason=openai_finish_reason,
        logprobs=logprobs,
        token_ids=list(completion_ids) if parsed.return_token_ids else None,
    )
    response = ChatCompletionResponse(
        id=parsed.request.request_id,
        model=parsed.model,
        choices=[choice],
        usage=UsageInfo(
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=len(completion_ids),
            total_tokens=len(prompt_token_ids) + len(completion_ids),
        ),
        prompt_token_ids=list(prompt_token_ids) if parsed.return_token_ids else None,
    ).model_dump(mode="json", exclude_none=True)
    reasoning = assistant_message.get("reasoning_content")
    if parsed.request.include_reasoning and reasoning is not None:
        response["choices"][0]["message"]["reasoning_content"] = reasoning
    return response
