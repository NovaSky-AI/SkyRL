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
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from renderers.base import MultiModalData


@dataclass(frozen=True, slots=True)
class Media:
    """One multimodal item, such as an image, and where its placeholder tokens sit.

    ``offset`` is relative to the tokens the item is listed with: a ``Rendered``'s
    ``token_ids``, a node's tokens, or a sample's ``input_ids``. ``hash`` identifies
    the item's content. ``data`` is the processor's output for the item (for
    Qwen-VL, ``pixel_values`` and ``image_grid_thw``), or None for an item read
    back from a record, which keeps placeholders only. Two items are equal, and
    hash alike, by everything but ``data``, which ``hash`` already identifies.
    """

    modality: str
    offset: int
    length: int
    hash: str
    data: Mapping[str, np.ndarray] | None = field(default=None, compare=False)

    def shifted(self, delta: int) -> Media:
        return replace(self, offset=self.offset + delta)


@dataclass(frozen=True, slots=True)
class Rendered:
    """A prompt and who owns its tokens.

    ``tail_indices[i]`` is the message index (relative to the messages rendered,
    or to ``new_messages`` for a bridge) that ``token_ids[reused + i]`` belongs
    to, or ``-1`` for template scaffold. The first ``reused`` tokens are the
    previous turn's prompt and completion, unchanged.

    ``media`` are the multimodal items whose placeholders the render added, in
    order, with offsets into ``token_ids``. A bridge lists only the new
    messages' items; the previous turn's are the caller's.
    """

    token_ids: list[int]
    tail_indices: list[int]
    reused: int = 0
    media: tuple[Media, ...] = ()


class TokenRenderer(Protocol):
    name: str

    def render(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None) -> Rendered: ...

    def bridge(
        self,
        previous_prompt: Sequence[int],
        previous_completion: Sequence[int],
        new_messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        previous_media: Sequence[Media] = (),
    ) -> Rendered | None: ...

    def features(self, media: Sequence[Media]) -> dict[str, Any]: ...

    def parse(
        self,
        completion_ids: Sequence[int],
        tools: Sequence[Mapping[str, Any]] | None,
        prompt_ids: Sequence[int] = (),
    ) -> dict[str, Any]: ...

    def stop_token_ids(self) -> list[int]: ...

    def decode_spans(self, token_ids: Sequence[int], context: Sequence[int] = ()) -> tuple[str, list[int]]: ...


def normalize_tools(tools: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]] | None:
    """The request's tools as sent, OpenAI's ``{"type": "function", "function": {...}}`` wrapper kept.

    Chat templates that print the tool list (Qwen's ``tool | tojson``) print what the server passes
    them, and vLLM passes the request's tools as they came. The renderers that read a tool's
    fields unwrap the envelope themselves.
    """
    if not tools:
        return None
    return [dict(tool) for tool in tools]


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


def _media(data: MultiModalData | None, start: int = 0) -> tuple[Media, ...]:
    """A ``renderers`` ``MultiModalData``'s items from position ``start`` on, in stream order."""
    if data is None:
        return ()
    found = []
    for modality, ranges in data.mm_placeholders.items():
        hashes, items = data.mm_hashes.get(modality) or [], data.mm_items.get(modality) or []
        for placeholder, digest, item in zip(ranges, hashes, items, strict=True):
            if placeholder.offset >= start:
                found.append(Media(modality, placeholder.offset, placeholder.length, digest, item))
    return tuple(sorted(found, key=lambda media: media.offset))


def _multi_modal_data(media: Sequence[Media]) -> MultiModalData | None:
    """``media`` as a ``renderers`` ``MultiModalData``, or None without any."""
    if not media:
        return None
    from renderers.base import MultiModalData, PlaceholderRange

    data = MultiModalData()
    for item in media:
        data.mm_hashes.setdefault(item.modality, []).append(item.hash)
        data.mm_placeholders.setdefault(item.modality, []).append(PlaceholderRange(item.offset, item.length))
        data.mm_items.setdefault(item.modality, []).append(dict(item.data))
    return data


class RenderersRenderer:
    """A pool of ``renderers`` renderers, one tokenizer each, used from threads.

    ``thinking_retention="all"`` keeps a reasoning model's earlier thinking in
    the history. Dropping it would re-render the previous turn differently from
    what was sampled, and every turn would fork instead of extending.

    A multimodal renderer (Qwen-VL, Qwen3.5, Gemma 4, ...) processes images
    itself, with the model's Hugging Face processor. ``processor_kwargs`` are
    passed when loading it, and must be the engine's (vLLM's
    ``mm_processor_kwargs``, e.g. ``max_pixels``): an image processed
    differently has a different number of placeholder tokens than the engine
    expects.
    """

    #: How many encoded items ``features`` keeps.
    ENCODED_CACHE = 256

    def __init__(
        self,
        tokenizer: str,
        *,
        size: int = 8,
        renderer: str | None = None,
        thinking_retention: str = "all",
        chat_template_kwargs: Mapping[str, Any] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """``renderer`` names the ``renderers`` renderer (``qwen3-vl``, ``qwen3.5``, ...). By default it is
        looked up from ``tokenizer``, which finds only the exact Hugging Face names the library lists, so a
        local checkpoint or a fine-tune names it."""
        from pydantic import TypeAdapter
        from renderers import AutoRendererConfig, RendererConfig, create_renderer
        from renderers.base import is_multimodal, load_tokenizer

        self.name = tokenizer
        if renderer is None:
            config = AutoRendererConfig(thinking_retention=thinking_retention)
        else:
            config = TypeAdapter(RendererConfig).validate_python(
                {"name": renderer, "thinking_retention": thinking_retention}
            )

        def build() -> tuple[Any, Any]:
            loaded = load_tokenizer(tokenizer)
            renderer = create_renderer(loaded, config, chat_template_kwargs=chat_template_kwargs)
            if processor_kwargs:
                if not is_multimodal(renderer):
                    raise ValueError(f"processor_kwargs are for a multimodal model; {tokenizer} renders text only")
                from transformers import AutoProcessor

                # A multimodal renderer loads its processor lazily, without kwargs; one given here wins.
                renderer._processor = AutoProcessor.from_pretrained(tokenizer, **processor_kwargs)
            return renderer, loaded

        #: (renderer, its tokenizer) pairs, each used by one thread at a time.
        self._slots: queue.Queue[tuple[Any, Any]] = queue.Queue()
        # The first slot loads on this thread, so a tokenizer that isn't cached yet is
        # downloaded once; the rest load from the cache in parallel.
        self._slots.put(build())
        with ThreadPoolExecutor(max_workers=min(size, 8)) as pool:
            for slot in pool.map(lambda _: build(), range(size - 1)):
                self._slots.put(slot)
        with self._checkout() as (renderer, _):
            self._stop_ids = [int(t) for t in renderer.get_stop_token_ids()]
            self.multimodal = is_multimodal(renderer)
        # An item's encoded vLLM features (its processed arrays) don't depend on where its placeholders
        # sit; the offset is sent separately, in ``mm_placeholders``. So ``features`` keys this cache by the
        # item moved to offset 0, which is its content, and an image is encoded once, not every turn.
        self._encode = lru_cache(maxsize=self.ENCODED_CACHE)(self._encode_item)

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
        return Rendered(
            token_ids=list(out.token_ids),
            tail_indices=list(out.message_indices),
            media=_media(getattr(out, "multi_modal_data", None)),
        )

    def bridge(
        self,
        previous_prompt: Sequence[int],
        previous_completion: Sequence[int],
        new_messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        previous_media: Sequence[Media] = (),
    ) -> Rendered | None:
        """``previous_media`` are the previous prompt's items, which a template that numbers images counts on."""
        extra: dict[str, Any] = {}
        if self.multimodal:
            extra["previous_multi_modal_data"] = _multi_modal_data(previous_media)
        with self._checkout() as (renderer, _):
            out = renderer.bridge_to_next_turn(
                list(previous_prompt),
                list(previous_completion),
                list(new_messages),
                tools=normalize_tools(tools),
                **extra,
            )
        if out is None:
            return None
        reused = len(previous_prompt) + len(previous_completion)
        token_ids = list(out.token_ids)
        # The library may trim the previous turn to its last turn-close token,
        # which moves the boundary; a full render is always correct, so decline.
        if token_ids[:reused] != [*previous_prompt, *previous_completion]:
            return None
        return Rendered(
            token_ids=token_ids,
            tail_indices=list(out.message_indices[reused:]),
            reused=reused,
            media=_media(getattr(out, "multi_modal_data", None), start=reused),
        )

    def features(self, media: Sequence[Media]) -> dict[str, Any]:
        """vLLM's ``features`` for a prompt with these items: hashes, placeholders and encoded items.

        Encoding is the ``renderers`` library's, per model family, and needs ``vllm`` installed.
        """
        out: dict[str, Any] = {"mm_hashes": {}, "mm_placeholders": {}, "kwargs_data": {}}
        for item in media:
            encoded = self._encode(item.shifted(-item.offset))
            out["mm_hashes"].setdefault(item.modality, []).append(item.hash)
            out["mm_placeholders"].setdefault(item.modality, []).append({"offset": item.offset, "length": item.length})
            out["kwargs_data"].setdefault(item.modality, []).append(encoded)
        return out

    def _encode_item(self, item: Media) -> Any:
        """One item's vLLM ``kwargs_data`` entry."""
        from renderers.client import _build_mm_features

        with self._checkout() as (renderer, _):
            single = _build_mm_features(renderer, _multi_modal_data([item]))
        return single["kwargs_data"][item.modality][0]

    def parse(
        self,
        completion_ids: Sequence[int],
        tools: Sequence[Mapping[str, Any]] | None,
        prompt_ids: Sequence[int] = (),
    ) -> dict[str, Any]:
        """Only cleanly parsed tool calls become ``tool_calls``; a malformed one stays in the text.

        ``prompt_ids`` is the prompt the completion was sampled after. A template that opens the
        thinking block in the generation prompt (Qwen3.5's ``<think>``) leaves the completion
        starting inside it, and the parser only splits ``reasoning_content`` off when it can see that.
        """
        from renderers import ToolCallParseStatus

        with self._checkout() as (renderer, _):
            parsed = renderer.parse_response(
                list(completion_ids), tools=normalize_tools(tools), prompt_ids=list(prompt_ids)
            )
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
            backend = getattr(tokenizer, "backend_tokenizer", None) or getattr(tokenizer, "_tokenizer", None)
            if backend is None:
                raise TypeError(f"decoding token spans needs a fast tokenizer; {self.name} loaded a slow one")
            stream = DecodeStream(skip_special_tokens=False)
            for token in context:
                stream.step(backend, int(token))
            pieces = [stream.step(backend, int(token)) or "" for token in token_ids]
        return join_pieces(pieces)
