"""Renderer abstraction used by the TITO proxy."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import threading
import types
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

from .types import Message, RoutedExperts, ToolSpec


@dataclass(frozen=True)
class RenderedPrompt:
    """Prompt token IDs and per-token message attribution."""

    token_ids: Tuple[int, ...]
    message_indices: Tuple[int, ...]
    reused_prefix_length: int = 0


class TITORenderer(Protocol):
    """Model-aware message/token conversion required by the proxy."""

    def render(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
    ) -> RenderedPrompt:
        """Fully render a chat request through the next generation prompt."""
        ...

    def bridge(
        self,
        previous_prompt_ids: Sequence[int],
        previous_completion_ids: Sequence[int],
        new_messages: Sequence[Message],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
    ) -> Optional[RenderedPrompt]:
        """Safely extend an exact previous prompt and completion."""
        ...

    def parse_response(
        self,
        token_ids: Sequence[int],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
    ) -> Message:
        """Parse exact sampled IDs into an assistant message."""
        ...

    def get_stop_token_ids(self) -> Sequence[int]:
        """Return generation stop token IDs for the model format."""
        ...

    def decode_token(self, token_id: int) -> str:
        """Decode one token for an OpenAI-compatible logprob response."""
        ...


def convert_routed_experts(value: Any) -> Optional[RoutedExperts]:
    if value is None:
        return None
    return tuple(tuple(tuple(int(expert) for expert in layer) for layer in token) for token in value)


def _ensure_transformers_tokenization_utils_compat() -> None:
    """Provide the Transformers 4.x import path expected by pinned renderers."""
    try:
        module = importlib.import_module("transformers.tokenization_utils")
        if hasattr(module, "PreTrainedTokenizer"):
            return
    except (ImportError, ModuleNotFoundError):
        module = types.ModuleType("transformers.tokenization_utils")
    from transformers import PreTrainedTokenizer

    setattr(module, "PreTrainedTokenizer", PreTrainedTokenizer)
    sys.modules["transformers.tokenization_utils"] = module


def _normalize_tools(tools: Optional[Sequence[ToolSpec]]) -> Optional[list[ToolSpec]]:
    if tools is None:
        return None
    normalized = []
    for tool in tools:
        function = tool.get("function") if tool.get("type") == "function" else None
        normalized.append(dict(function) if isinstance(function, dict) else dict(tool))
    return normalized


class PrimeRendererAdapter:
    """Thread-safe adapter around an auto-resolved Prime renderer."""

    def __init__(
        self,
        tokenizer,
        *,
        config: Any = None,
        chat_template_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        _ensure_transformers_tokenization_utils_compat()
        from renderers import AutoRendererConfig, create_renderer

        self._tokenizer = tokenizer
        renderer_config = config or AutoRendererConfig(thinking_retention="all")
        self._renderer: Any = create_renderer(
            tokenizer,
            renderer_config,
            chat_template_kwargs=chat_template_kwargs,
        )
        self._lock = threading.Lock()

    @property
    def renderer_name(self) -> str:
        return str(self._renderer.config.name)

    def render(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
    ) -> RenderedPrompt:
        with self._lock:
            rendered = self._renderer.render(
                list(messages),
                tools=_normalize_tools(tools),
                add_generation_prompt=True,
            )
        return RenderedPrompt(
            token_ids=tuple(rendered.token_ids),
            message_indices=tuple(rendered.message_indices),
        )

    def bridge(
        self,
        previous_prompt_ids: Sequence[int],
        previous_completion_ids: Sequence[int],
        new_messages: Sequence[Message],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
    ) -> Optional[RenderedPrompt]:
        with self._lock:
            rendered = self._renderer.bridge_to_next_turn(
                list(previous_prompt_ids),
                list(previous_completion_ids),
                list(new_messages),
                tools=_normalize_tools(tools),
            )
        if rendered is None:
            return None
        reused_prefix_length = len(previous_prompt_ids) + len(previous_completion_ids)
        expected_prefix = tuple(previous_prompt_ids) + tuple(previous_completion_ids)
        if tuple(rendered.token_ids[:reused_prefix_length]) != expected_prefix:
            raise ValueError("Renderer bridge did not preserve exact previous token IDs")
        return RenderedPrompt(
            token_ids=tuple(rendered.token_ids),
            message_indices=tuple(rendered.message_indices),
            reused_prefix_length=reused_prefix_length,
        )

    def parse_response(
        self,
        token_ids: Sequence[int],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
    ) -> Message:
        from renderers.base import ToolCallParseStatus

        with self._lock:
            parsed = self._renderer.parse_response(
                list(token_ids),
                tools=_normalize_tools(tools),
            )

        message: Message = {"role": "assistant", "content": parsed.content}
        if parsed.reasoning_content is not None:
            message["reasoning_content"] = parsed.reasoning_content

        tool_calls = []
        token_digest = hashlib.sha256(json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")).hexdigest()
        for index, tool_call in enumerate(parsed.tool_calls):
            if tool_call.status != ToolCallParseStatus.OK or not tool_call.name:
                continue
            arguments = tool_call.arguments
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, separators=(",", ":"), ensure_ascii=False)
            tool_calls.append(
                {
                    "id": tool_call.id or f"call_{token_digest[:20]}_{index}",
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": arguments,
                    },
                }
            )
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def get_stop_token_ids(self) -> Sequence[int]:
        with self._lock:
            return tuple(self._renderer.get_stop_token_ids())

    def decode_token(self, token_id: int) -> str:
        return self._tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
