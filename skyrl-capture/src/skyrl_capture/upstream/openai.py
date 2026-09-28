"""OpenAI: Chat Completions and Responses.

One adapter, two endpoint shapes, because they are one provider's protocol and
a client picks between them per request rather than per process. Everything
here is shallow reading of already-forwarded bytes: nothing in this file can
fail a request, and an unrecognised shape becomes a recorded derivation error.
"""

from __future__ import annotations

from typing import Any

import orjson

from skyrl_capture.domain.extraction import ExtractedExchange, ExtractedMessage
from skyrl_capture.upstream.text import BaseTextProtocol, decode_frames

ERROR_TYPES = {
    401: "invalid_request_error",
    403: "invalid_request_error",
    404: "invalid_request_error",
    410: "invalid_request_error",
    413: "invalid_request_error",
    502: "api_error",
    503: "api_error",
}

# Request fields that are context or transport rather than "what was asked
# for". Everything else is kept in `parameters` for replay and export.
_CHAT_NON_PARAM = frozenset(
    {"messages", "model", "stream", "tools", "max_tokens", "max_completion_tokens"}
)
_RESPONSES_NON_PARAM = frozenset(
    {"input", "model", "stream", "tools", "max_output_tokens", "instructions"}
)


class OpenAIProtocol(BaseTextProtocol):
    name = "openai"

    def error_body(self, status: int, message: str, *, code: str | None = None) -> bytes:
        return orjson.dumps(
            {
                "error": {
                    "message": message,
                    "type": ERROR_TYPES.get(status, "api_error"),
                    "param": None,
                    "code": code,
                }
            }
        )

    # -- extraction --------------------------------------------------------
    def extract(
        self,
        *,
        endpoint_kind: str,
        request_body: bytes,
        response_body: bytes,
        streaming: bool,
        status: int | None,
    ) -> ExtractedExchange:
        out = ExtractedExchange()
        request: Any = None
        if request_body:
            try:
                request = orjson.loads(request_body)
            except orjson.JSONDecodeError as error:
                out.errors.append(f"request: {error}")

        if isinstance(request, dict):
            out.model = request.get("model")
            tools = request.get("tools")
            if isinstance(tools, list):
                out.tools = tools
            if endpoint_kind == "responses":
                self._responses_request(out, request)
            else:
                self._chat_request(out, request)

        if not response_body:
            return out
        if streaming:
            self._stream(out, response_body)
            return out

        try:
            response = orjson.loads(response_body)
        except orjson.JSONDecodeError as error:
            out.errors.append(f"response: {error}")
            return out
        if not isinstance(response, dict):
            return out
        if status is not None and status >= 400:
            error_payload = response.get("error")
            out.provider_error = error_payload if isinstance(error_payload, dict) else {"raw": response}
            return out
        if isinstance(response.get("error"), dict):
            out.provider_error = response["error"]

        if endpoint_kind == "responses":
            self._responses_response(out, response)
        else:
            self._chat_response(out, response)
        return out

    # -- chat completions --------------------------------------------------
    def _chat_request(self, out: ExtractedExchange, request: dict[str, Any]) -> None:
        messages = request.get("messages")
        if isinstance(messages, list):
            out.input_messages = [
                ExtractedMessage(message) for message in messages if isinstance(message, dict)
            ]
        out.max_output_tokens = request.get("max_completion_tokens") or request.get("max_tokens")
        out.parameters = {key: value for key, value in request.items() if key not in _CHAT_NON_PARAM}

    def _chat_response(self, out: ExtractedExchange, response: dict[str, Any]) -> None:
        out.provider_response_id = response.get("id")
        out.usage = _usage(response.get("usage"))
        out.model = out.model or response.get("model")
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            out.finish_reason = choice.get("finish_reason")
            message = choice.get("message")
            if isinstance(message, dict):
                out.output_message = ExtractedMessage(message)

    # -- responses ---------------------------------------------------------
    def _responses_request(self, out: ExtractedExchange, request: dict[str, Any]) -> None:
        """Responses accepts ``input`` as a string or a structured item list."""
        out.previous_response_id = request.get("previous_response_id")
        out.max_output_tokens = request.get("max_output_tokens")
        messages: list[ExtractedMessage] = []
        instructions = request.get("instructions")
        if isinstance(instructions, str) and instructions:
            messages.append(
                ExtractedMessage(
                    {"role": "system", "content": instructions},
                    derivation={"materialized_from": "instructions"},
                )
            )
        value = request.get("input")
        if isinstance(value, str):
            messages.append(ExtractedMessage({"role": "user", "content": value}))
        elif isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                if item.get("role"):
                    messages.append(ExtractedMessage(item))
                else:
                    # Non-message items (tool outputs, reasoning) still occupy a
                    # position in the context and so still get a node.
                    messages.append(
                        ExtractedMessage(
                            {"role": item.get("type", "item"), "content": item},
                            derivation={"materialized_from": "input_item"},
                        )
                    )
        out.input_messages = messages
        out.parameters = {
            key: value for key, value in request.items() if key not in _RESPONSES_NON_PARAM
        }

    def _responses_response(self, out: ExtractedExchange, response: dict[str, Any]) -> None:
        out.provider_response_id = response.get("id")
        out.previous_response_id = response.get("previous_response_id") or out.previous_response_id
        out.finish_reason = response.get("status")
        usage = response.get("usage")
        if isinstance(usage, dict):
            input_details = usage.get("input_tokens_details")
            output_details = usage.get("output_tokens_details")
            out.usage = {
                "prompt_tokens": usage.get("input_tokens"),
                "completion_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "cached_tokens": (
                    input_details.get("cached_tokens") if isinstance(input_details, dict) else None
                ),
                "reasoning_tokens": (
                    output_details.get("reasoning_tokens")
                    if isinstance(output_details, dict)
                    else None
                ),
                "raw": usage,
            }
        output = response.get("output")
        if not isinstance(output, list):
            return
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                out.output_message = ExtractedMessage(
                    {"role": item.get("role", "assistant"), "content": item.get("content")}
                )
                return
        if output:
            out.output_message = ExtractedMessage(
                {"role": "assistant", "content": output},
                derivation={"materialized_from": "response_output"},
            )

    # -- streaming ---------------------------------------------------------
    def _stream(self, out: ExtractedExchange, body: bytes) -> None:
        """Reconstruct the assistant message this stream would have returned."""
        frames = decode_frames(body)
        out.stream_frames = frames
        out.stream_summary = {"event_count": len(frames), "frame_count": len(frames)}

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        role = "assistant"
        for frame in frames:
            payload = frame.get("data")
            if not isinstance(payload, dict):
                continue
            if payload.get("id") and not out.provider_response_id:
                out.provider_response_id = payload["id"]
            if payload.get("model") and not out.model:
                out.model = payload["model"]
            if isinstance(payload.get("usage"), dict):
                out.usage = _usage(payload["usage"])
            choices = payload.get("choices")
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                if choice.get("finish_reason"):
                    out.finish_reason = choice["finish_reason"]
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                if delta.get("role"):
                    role = delta["role"]
                piece = delta.get("content")
                if isinstance(piece, str):
                    content_parts.append(piece)
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(reasoning, str):
                    reasoning_parts.append(reasoning)
                for call in delta.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    index = int(call.get("index", 0))
                    slot = tool_calls.setdefault(
                        index,
                        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if call.get("id"):
                        slot["id"] = call["id"]
                    function = call.get("function")
                    if isinstance(function, dict):
                        if function.get("name"):
                            slot["function"]["name"] = function["name"]
                        if function.get("arguments"):
                            slot["function"]["arguments"] += function["arguments"]

        if not content_parts and not tool_calls and not reasoning_parts:
            return
        message: dict[str, Any] = {
            "role": role,
            "content": "".join(content_parts) if content_parts else None,
        }
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        out.output_message = ExtractedMessage(
            message, derivation={"assembled_from": "openai_stream", "frames": len(frames)}
        )
        out.stream_summary["assembled_chars"] = sum(len(part) for part in content_parts)


def _usage(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": details.get("cached_tokens") if isinstance(details, dict) else None,
        "reasoning_tokens": (
            completion_details.get("reasoning_tokens")
            if isinstance(completion_details, dict)
            else None
        ),
        "raw": usage,
    }
