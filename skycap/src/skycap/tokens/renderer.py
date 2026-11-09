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
    ``token_ids``, a node's tokens, or a sample's ``input_ids``. ``length`` is the
    number of placeholder tokens, which is not always the number of embeddings the
    model puts there: Kimi K2.5 has one placeholder per image and expands it inside
    the model. ``hash`` identifies the item's content (its decoded pixels and size).
    ``data`` is the model's processor output for the item, whose keys vary by family
    (Qwen-VL and Qwen3.5: ``pixel_values``, ``image_grid_thw``; Gemma 4:
    ``pixel_values``, ``image_position_ids``; Inkling: ``pixel_values``), or None
    for an item read back from a record, which keeps placeholders only. Two items
    are equal, and hash alike, by everything but ``data``, which ``hash`` already
    identifies.
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
    """How token mode renders and parses. A renderer may also expose ``chat_template``, its chat
    template's text, which decides whether ``developer`` messages need folding (``tokens.developer``);
    one without it gets them as sent."""

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
        prompt_ids: Sequence[int] | None = None,
    ) -> dict[str, Any]: ...

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


@lru_cache(maxsize=16)
def _png_url(width: int, height: int) -> str:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def png_part(width: int, height: int) -> dict[str, Any]:
    """A plain white PNG of this size as an OpenAI ``image_url`` content part (a data URI). Needs Pillow."""
    return {"type": "image_url", "image_url": {"url": _png_url(width, height)}}


class RenderersRenderer:
    """A pool of ``renderers`` renderers, one tokenizer each, used from threads.

    ``thinking_retention="all"`` keeps a reasoning model's earlier thinking in
    the history. Dropping it would re-render the previous turn differently from
    what was sampled, and every turn would fork instead of extending.

    A multimodal renderer (Qwen-VL, Qwen3.5, Gemma 4, ...) processes images
    itself, with the model's Hugging Face processor. ``processor_kwargs`` are
    passed when loading it and must be the engine's (vLLM's
    ``mm_processor_kwargs``, e.g. ``max_pixels``): an image processed
    differently has a different number of placeholder tokens than the engine
    expects. ``processor_probes`` gives the requests that check this against
    the engine, before the first call with images
    (``TokensBackend._check_media``).
    """

    #: How many encoded items ``features`` keeps.
    ENCODED_CACHE = 256

    def __init__(
        self,
        tokenizer: str,
        *,
        size: int = 8,
        thinking_retention: str = "all",
        chat_template_kwargs: Mapping[str, Any] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        from renderers import AutoRendererConfig, create_renderer
        from renderers.base import is_multimodal, load_tokenizer

        self.name = tokenizer

        def build() -> tuple[Any, Any]:
            loaded = load_tokenizer(tokenizer)
            renderer = create_renderer(
                loaded,
                AutoRendererConfig(thinking_retention=thinking_retention),
                chat_template_kwargs=chat_template_kwargs,
            )
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
        with self._checkout() as (renderer, loaded):
            self._stop_ids = [int(t) for t in renderer.get_stop_token_ids()]
            self.multimodal = is_multimodal(renderer)
            template = getattr(loaded, "chat_template", None)
        if isinstance(template, dict):
            template = template.get("default")
        #: The tokenizer's chat template text, which ``tokens.developer`` tests for the ``developer`` role.
        self.chat_template: str | None = template if isinstance(template, str) else None
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

    def validate_media_encoding(self) -> str | None:
        """Encode a probe image for vLLM: None if it works (or the model takes no images), else why not.

        It runs what an image request runs: the chat template and the model's processor (``render``), then
        the ``renderers`` encoding for vLLM (``features``). So it catches a family ``renderers`` renders but
        can't encode for vLLM (Kimi K2.5, Inkling), ``vllm`` missing beside skycap, and an encoder that fails.
        """
        if not self.multimodal:
            return None
        try:
            probe = [{"role": "user", "content": [png_part(64, 64), {"type": "text", "text": "?"}]}]
            rendered = self.render(probe, None)
            if not rendered.media:
                return "the probe image rendered no multimodal item"
            self.features(rendered.media)
        except Exception as error:  # noqa: BLE001 - whatever the processor or the encoder raises
            return f"{type(error).__name__}: {error}"
        return None

    def processor_probes(self) -> list[tuple[str, list[dict[str, Any]]]]:
        """Requests whose prompt token counts show whether images are processed here as in the engine.

        ``(label, messages)`` pairs, text first: a text-only difference is the chat template's, not the
        processor's. The images are sized to reach the processor's limits (min and max pixels, aspect ratio)
        and an ordinary screenshot. Empty for a model that takes no images.
        """
        if not self.multimodal:
            return []
        ask = {"type": "text", "text": "Describe the image."}
        probes = [("text", [{"role": "user", "content": "Describe the image."}])]
        for width, height in ((8, 8), (6000, 1000), (1024, 768)):
            probes.append((f"{width}x{height} image", [{"role": "user", "content": [png_part(width, height), ask]}]))
        return probes

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
        prompt_ids: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Return extracted calls without schema validation; keep failed parses as text.

        ``prompt_ids`` is the prompt the completion was sampled from. The parser reads its tail to
        tell whether the template already opened the reasoning block: Qwen3.5 with thinking on ends
        every generation prompt in ``<think>\\n``, so the completion starts mid-reasoning and has no
        opening tag of its own.
        """
        from renderers import ToolCallParseStatus

        with self._checkout() as (renderer, _):
            parsed = renderer.parse_response(
                list(completion_ids),
                tools=normalize_tools(tools),
                prompt_ids=list(prompt_ids) if prompt_ids is not None else None,
            )
        message: dict[str, Any] = {"role": "assistant", "content": parsed.content}
        if getattr(parsed, "reasoning_content", None) is not None:
            message["reasoning_content"] = parsed.reasoning_content
        calls = []
        for index, call in enumerate(getattr(parsed, "tool_calls", None) or ()):
            # XML parsers also use INVALID_JSON when an extracted parameter cannot be
            # coerced to its declared type. Its argument object is still complete;
            # vLLM's XML parser returns the original string for the caller to validate.
            extracted_arguments = call.status == ToolCallParseStatus.INVALID_JSON and isinstance(call.arguments, dict)
            if not call.name or (call.status != ToolCallParseStatus.OK and not extracted_arguments):
                # Parsers extract even failed attempts from content. Restore their raw text;
                # UNKNOWN_TOOL and structural failures must not become callable actions.
                message["content"] = "\n".join(part for part in (message["content"], call.raw) if part)
                continue
            arguments = call.arguments
            if not isinstance(arguments, str):
                arguments = json.dumps(
                    {} if arguments is None else arguments, separators=(",", ":"), ensure_ascii=False
                )
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
