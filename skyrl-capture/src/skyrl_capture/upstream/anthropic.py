"""Anthropic Messages.

Differs from OpenAI in five places, and all five are here rather than in a
central parser: the route the client is pointed at, the environment variables
it reads, the header that carries a credential, the error envelope, and the
body -- a top-level ``system`` field, block-shaped content, and a streaming
protocol that builds content blocks by index rather than concatenating a
string.
"""

from __future__ import annotations

from typing import Any

import orjson

from skyrl_capture.domain.extraction import ExtractedExchange, ExtractedMessage
from skyrl_capture.upstream.text import BaseTextProtocol, decode_frames

ERROR_TYPES = {
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    410: "invalid_request_error",
    413: "request_too_large",
    502: "api_error",
    503: "overloaded_error",
}

_NON_PARAM = frozenset({"messages", "model", "stream", "tools", "max_tokens", "system"})


class AnthropicProtocol(BaseTextProtocol):
    name = "anthropic"
    # The Anthropic SDK appends /v1 itself, so the client-facing route must not.
    client_suffix = ""
    environment_map = (
        ("ANTHROPIC_BASE_URL", "base_url"),
        ("ANTHROPIC_API_KEY", "api_key"),
        ("ANTHROPIC_AUTH_TOKEN", "api_key"),
    )

    def auth_headers(self, api_key: str) -> list[tuple[bytes, bytes]]:
        return [(b"x-api-key", api_key.encode())]

    def error_body(self, status: int, message: str, *, code: str | None = None) -> bytes:
        return orjson.dumps(
            {
                "type": "error",
                "error": {"type": ERROR_TYPES.get(status, "api_error"), "message": message},
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
            self._request(out, request)

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
        self._response(out, response)
        return out

    def _request(self, out: ExtractedExchange, request: dict[str, Any]) -> None:
        messages: list[ExtractedMessage] = []
        system = request.get("system")
        if system:
            # The system prompt is a top-level field here rather than a
            # message, so it is materialized as a leading one: without it, two
            # requests differing only in system prompt would share a prefix
            # they do not actually share. The transformation travels onto the
            # node as evidence rather than being hidden.
            messages.append(
                ExtractedMessage(
                    {"role": "system", "content": system},
                    derivation={"materialized_from": "system_field"},
                )
            )
        raw_messages = request.get("messages")
        if isinstance(raw_messages, list):
            messages.extend(ExtractedMessage(item) for item in raw_messages if isinstance(item, dict))
        out.input_messages = messages
        out.max_output_tokens = request.get("max_tokens")
        out.parameters = {key: value for key, value in request.items() if key not in _NON_PARAM}

    def _response(self, out: ExtractedExchange, response: dict[str, Any]) -> None:
        out.provider_response_id = response.get("id")
        out.finish_reason = response.get("stop_reason")
        out.model = out.model or response.get("model")
        out.usage = _usage(response.get("usage"))
        content = response.get("content")
        if content is not None:
            out.output_message = ExtractedMessage(
                {"role": response.get("role", "assistant"), "content": content}
            )

    # -- streaming ---------------------------------------------------------
    def _stream(self, out: ExtractedExchange, body: bytes) -> None:
        """Reconstruct a message from its block-delta stream."""
        frames = decode_frames(body)
        out.stream_frames = frames
        out.stream_summary = {"event_count": len(frames), "frame_count": len(frames)}

        blocks: dict[int, dict[str, Any]] = {}
        role = "assistant"
        usage: dict[str, Any] = {}
        for frame in frames:
            payload = frame.get("data")
            if not isinstance(payload, dict):
                continue
            kind = payload.get("type")
            if kind == "message_start":
                message = payload.get("message") or {}
                out.provider_response_id = message.get("id") or out.provider_response_id
                out.model = out.model or message.get("model")
                role = message.get("role", role)
                if isinstance(message.get("usage"), dict):
                    usage.update(message["usage"])
            elif kind == "content_block_start":
                blocks[int(payload.get("index", 0))] = dict(payload.get("content_block") or {})
            elif kind == "content_block_delta":
                index = int(payload.get("index", 0))
                delta = payload.get("delta") or {}
                block = blocks.setdefault(index, {"type": delta.get("type", "text")})
                delta_type = delta.get("type")
                if delta_type == "text_delta":
                    block["text"] = (block.get("text") or "") + (delta.get("text") or "")
                elif delta_type == "thinking_delta":
                    block["thinking"] = (block.get("thinking") or "") + (delta.get("thinking") or "")
                elif delta_type == "input_json_delta":
                    block["_partial_json"] = (block.get("_partial_json") or "") + (
                        delta.get("partial_json") or ""
                    )
                elif delta_type == "signature_delta":
                    block["signature"] = (block.get("signature") or "") + (
                        delta.get("signature") or ""
                    )
            elif kind == "message_delta":
                delta = payload.get("delta") or {}
                if delta.get("stop_reason"):
                    out.finish_reason = delta["stop_reason"]
                if isinstance(payload.get("usage"), dict):
                    usage.update(payload["usage"])

        for block in blocks.values():
            partial = block.pop("_partial_json", None)
            if partial is not None:
                try:
                    block["input"] = orjson.loads(partial)
                except orjson.JSONDecodeError:
                    block["input"] = {"_unparsed_partial_json": partial}

        if usage:
            out.usage = _usage(usage)
        if blocks:
            out.output_message = ExtractedMessage(
                {"role": role, "content": [blocks[index] for index in sorted(blocks)]},
                derivation={"assembled_from": "anthropic_stream", "frames": len(frames)},
            )


def _usage(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    return {
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
        "cached_tokens": usage.get("cache_read_input_tokens"),
        "cache_creation_tokens": usage.get("cache_creation_input_tokens"),
        "raw": usage,
    }
