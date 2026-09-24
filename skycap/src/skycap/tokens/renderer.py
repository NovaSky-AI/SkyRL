"""Messages to tokens and back, through the ``renderers`` library.

Model quirks (where reasoning goes, how tool calls are framed, which tokens are
scaffold) live in per-model ``renderers`` code, not in patched chat templates.
The library also gives what one-node-per-message needs: per-token message
attribution, and ``bridge_to_next_turn``, which extends the previous turn's
exact tokens instead of re-rendering what the model sampled.
"""

from __future__ import annotations

import hashlib
import json
import queue
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Rendered:
    """A prompt and who owns its tokens.

    ``tail_indices[i]`` is the message index (relative to the messages rendered,
    or to ``new_messages`` for a bridge) that ``token_ids[reused + i]`` belongs
    to, or ``-1`` for template scaffold. The first ``reused`` tokens are the
    previous turn's prompt and completion, unchanged.
    """

    token_ids: list[int]
    tail_indices: list[int]
    reused: int = 0


class TokenRenderer(Protocol):
    name: str

    def render(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None) -> Rendered: ...

    def bridge(
        self,
        previous_prompt: Sequence[int],
        previous_completion: Sequence[int],
        new_messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
    ) -> Rendered | None: ...

    def parse(self, completion_ids: Sequence[int], tools: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]: ...

    def stop_token_ids(self) -> list[int]: ...

    def decode_spans(self, token_ids: Sequence[int], context: Sequence[int] = ()) -> tuple[str, list[int]]: ...


def normalize_tools(tools: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]] | None:
    """OpenAI's ``{"type": "function", "function": {...}}`` wrapper, flattened."""
    if not tools:
        return None
    flat = []
    for tool in tools:
        function = tool.get("function") if tool.get("type") == "function" else None
        flat.append(dict(function) if isinstance(function, Mapping) else dict(tool))
    return flat


def join_pieces(pieces: Sequence[str]) -> tuple[str, list[int]]:
    """Per-token decoded pieces as one text and each piece's UTF-8 byte offset in it."""
    offsets, cursor = [], 0
    for piece in pieces:
        offsets.append(cursor)
        cursor += len(piece.encode("utf-8"))
    return "".join(pieces), offsets


def tool_call_id(completion_ids: Sequence[int], index: int) -> str:
    """Derived from the sampled tokens, so re-parsing a completion gives the same id."""
    digest = hashlib.sha256(json.dumps(list(completion_ids)).encode()).hexdigest()
    return f"call_{digest[:20]}_{index}"


class RenderersRenderer:
    """A pool of ``renderers`` renderers, one tokenizer each, used from threads.

    ``thinking_retention="all"`` keeps a reasoning model's earlier thinking in
    the history. Dropping it would re-render the previous turn differently from
    what was sampled, and every turn would fork instead of extending.
    """

    def __init__(
        self,
        tokenizer: str,
        *,
        size: int = 8,
        thinking_retention: str = "all",
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        from renderers import AutoRendererConfig, create_renderer
        from renderers.base import load_tokenizer

        self.name = tokenizer

        def build() -> tuple[Any, Any]:
            loaded = load_tokenizer(tokenizer)
            renderer = create_renderer(
                loaded,
                AutoRendererConfig(thinking_retention=thinking_retention),
                chat_template_kwargs=chat_template_kwargs,
            )
            return renderer, loaded

        #: (renderer, its tokenizer) pairs, each used by one thread at a time.
        self._slots: queue.Queue[tuple[Any, Any]] = queue.Queue()
        with ThreadPoolExecutor(max_workers=min(size, 8)) as pool:
            for slot in pool.map(lambda _: build(), range(size)):
                self._slots.put(slot)
        with self._checkout() as (renderer, _):
            self._stop_ids = [int(t) for t in renderer.get_stop_token_ids()]

    @contextmanager
    def _checkout(self) -> Iterator[tuple[Any, Any]]:
        slot = self._slots.get()
        try:
            yield slot
        finally:
            self._slots.put(slot)

    def render(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None) -> Rendered:
        with self._checkout() as (renderer, _):
            out = renderer.render(list(messages), tools=normalize_tools(tools), add_generation_prompt=True)
        return Rendered(token_ids=list(out.token_ids), tail_indices=list(out.message_indices))

    def bridge(
        self,
        previous_prompt: Sequence[int],
        previous_completion: Sequence[int],
        new_messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
    ) -> Rendered | None:
        with self._checkout() as (renderer, _):
            out = renderer.bridge_to_next_turn(
                list(previous_prompt),
                list(previous_completion),
                list(new_messages),
                tools=normalize_tools(tools),
            )
        if out is None:
            return None
        reused = len(previous_prompt) + len(previous_completion)
        token_ids = list(out.token_ids)
        # The library may trim the previous turn to its last turn-close token,
        # which moves the boundary; a full render is always correct, so decline.
        if token_ids[:reused] != [*previous_prompt, *previous_completion]:
            return None
        return Rendered(token_ids=token_ids, tail_indices=list(out.message_indices[reused:]), reused=reused)

    def parse(self, completion_ids: Sequence[int], tools: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
        """Only cleanly parsed tool calls become ``tool_calls``; a malformed one stays in the text."""
        from renderers import ToolCallParseStatus

        with self._checkout() as (renderer, _):
            parsed = renderer.parse_response(list(completion_ids), tools=normalize_tools(tools))
        message: dict[str, Any] = {"role": "assistant", "content": parsed.content}
        if getattr(parsed, "reasoning_content", None) is not None:
            message["reasoning_content"] = parsed.reasoning_content
        calls = []
        for index, call in enumerate(getattr(parsed, "tool_calls", None) or ()):
            if call.status != ToolCallParseStatus.OK or not call.name:
                continue
            arguments = call.arguments
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, separators=(",", ":"), ensure_ascii=False)
            calls.append(
                {
                    "id": call.id or tool_call_id(completion_ids, index),
                    "type": "function",
                    "function": {"name": call.name, "arguments": arguments},
                }
            )
        if calls:
            message["tool_calls"] = calls
        return message

    def stop_token_ids(self) -> list[int]:
        return list(self._stop_ids)

    def decode_spans(self, token_ids: Sequence[int], context: Sequence[int] = ()) -> tuple[str, list[int]]:
        """The text ``token_ids`` decode to, and each token's UTF-8 byte offset in it.

        Decodes as a stream: a token that holds only part of a character adds
        nothing, and the token that completes it adds the whole character. The
        stream is primed with ``context`` (the tokens just before), whose text
        is not returned, so a decoder that treats the start of a sequence
        specially decodes this node as it appears mid-sequence.
        """
        from tokenizers.decoders import DecodeStream

        with self._checkout() as (_, tokenizer):
            backend = getattr(tokenizer, "backend_tokenizer", None) or tokenizer._tokenizer
            stream = DecodeStream(skip_special_tokens=False)
            for token in context:
                stream.step(backend, int(token))
            pieces = [stream.step(backend, int(token)) or "" for token in token_ids]
        return join_pieces(pieces)
