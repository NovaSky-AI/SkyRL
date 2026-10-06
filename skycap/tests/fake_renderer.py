"""A deterministic stand-in for a ``renderers`` renderer, for tests only.

Characters are tokens (``ord(c) + 100``) and three specials frame a message:

    START role NL content END NL

The generation prompt is ``START "assistant" NL``. A completion is the
content followed by ``END``. Reasoning renders as ``THINK:<text>|`` before the
content, and a completion beginning ``CALL:<name>:<args>`` parses as a tool
call. The bridge keeps the contract the real library does: the previous
prompt and completion come back unchanged, it declines a completion that
doesn't end in ``END``, and it attributes only the new messages.

Content may be a list of parts. An ``image_url`` part whose URL is
``fake://<name>/<n>`` (``image(name, n)``) renders as ``IMG_START``, ``n``
``IMG`` placeholders and ``IMG_END``, and is an item hashed ``name``, whose
``pixel_values`` are ``n`` rows filled with ``len(name)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from skycap.tokens.renderer import Media, Rendered, join_pieces

START, END, NL = 1, 2, 3
IMG_START, IMG, IMG_END = 4, 5, 6
SPECIAL_TEXT = {START: "<s>", END: "</s>", NL: "\n", IMG_START: "<img>", IMG: "<pad>", IMG_END: "</img>"}


def image(name: str, placeholders: int) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"fake://{name}/{placeholders}"}}


def image_data(name: str, placeholders: int) -> dict[str, np.ndarray]:
    return {
        "pixel_values": np.full((placeholders, 2), float(len(name)), dtype=np.float32),
        "image_grid_thw": np.array([[1, 1, placeholders]], dtype=np.int64),
    }


def encode(text: str) -> list[int]:
    return [ord(c) + 100 for c in text]


def decode(tokens: Sequence[int]) -> str:
    return "".join(chr(t - 100) for t in tokens if t >= 100)


def _body(message: Mapping[str, Any], empty_content: str = "") -> str:
    content = message.get("content")
    text = empty_content if content == "" else content if isinstance(content, str) else ""
    if message.get("reasoning_content"):
        text = f"THINK:{message['reasoning_content']}|{text}"
    for call in message.get("tool_calls") or ():
        text += f"CALL:{call['function']['name']}:{call['function']['arguments']}"
    return text


class FakeRenderer:
    name = "fake"

    def __init__(self) -> None:
        #: Set to make the next render attribute a token out of range.
        self.corrupt = False
        #: What ``content: ""`` renders as. Empty by default, so it renders like no content.
        self.empty_content = ""
        #: Full renders requested (``render``); bridges don't count.
        self.renders = 0
        #: Set to decline every bridge, so each call renders in full.
        self.no_bridge = False
        #: The ``previous_media`` of each bridge that received some.
        self.bridged_media: list[list[Media]] = []

    def _message(self, message: Mapping[str, Any], start: int) -> tuple[list[int], list[Media]]:
        tokens = [START, *encode(str(message.get("role"))), NL]
        media: list[Media] = []
        content = message.get("content")
        for part in content if isinstance(content, list) else ():
            if part.get("type") == "text":
                tokens += encode(part["text"])
            elif part.get("type") == "image_url":
                name, _, count = part["image_url"]["url"].removeprefix("fake://").partition("/")
                tokens.append(IMG_START)
                media.append(Media("image", start + len(tokens), int(count), name, image_data(name, int(count))))
                tokens += [IMG] * int(count) + [IMG_END]
        tokens += [*encode(_body(message, self.empty_content)), END, NL]
        return tokens, media

    def render(self, messages: Sequence[Mapping[str, Any]], tools: Any) -> Rendered:
        self.renders += 1
        return self._render(messages, tools)

    def _render(self, messages: Sequence[Mapping[str, Any]], tools: Any, start: int = 0) -> Rendered:
        tokens: list[int] = []
        indices: list[int] = []
        media: list[Media] = []
        for index, message in enumerate(messages):
            chunk, items = self._message(message, start + len(tokens))
            tokens += chunk
            indices += [index] * len(chunk)
            media += items
        prompt = [START, *encode("assistant"), NL]
        tokens += prompt
        indices += [-1] * len(prompt)
        if self.corrupt:
            indices[0] = len(messages) + 5
        return Rendered(token_ids=tokens, tail_indices=indices, media=tuple(media))

    def bridge(
        self,
        previous_prompt: Sequence[int],
        previous_completion: Sequence[int],
        new_messages: Sequence[Mapping[str, Any]],
        tools: Any,
        previous_media: Sequence[Media] = (),
    ) -> Rendered | None:
        if not previous_completion or previous_completion[-1] != END or self.corrupt or self.no_bridge:
            return None
        if any(m.get("role") == "assistant" for m in new_messages):
            return None
        if previous_media:
            self.bridged_media.append(list(previous_media))
        reused = len(previous_prompt) + len(previous_completion)
        tail = self._render(new_messages, tools, start=reused + 1)
        tokens = [*previous_prompt, *previous_completion, NL, *tail.token_ids]
        indices = [-1, *tail.tail_indices]
        return Rendered(token_ids=tokens, tail_indices=indices, reused=reused, media=tail.media)

    def features(self, media: Sequence[Media]) -> dict[str, Any]:
        return {
            "mm_hashes": {"image": [item.hash for item in media]},
            "mm_placeholders": {"image": [{"offset": item.offset, "length": item.length} for item in media]},
            "kwargs_data": {"image": [f"encoded:{item.hash}" for item in media]},
        }

    def parse(self, completion_ids: Sequence[int], tools: Any) -> dict[str, Any]:
        text = decode([t for t in completion_ids if t != END])
        message: dict[str, Any] = {"role": "assistant", "content": text}
        reasoning_content = None
        if text.startswith("THINK:") and "|" in text:
            reasoning_content, _, text = text[6:].partition("|")
            message = {"role": "assistant", "content": text, "reasoning_content": reasoning_content}
        if text.startswith("CALL:"):
            name, _, arguments = text[5:].partition(":")
            message = {
                "role": "assistant",
                "content": "" if reasoning_content is not None else None,
                "tool_calls": [
                    {"id": "call_0", "type": "function", "function": {"name": name, "arguments": arguments}}
                ],
            }
            if reasoning_content is not None:
                message["reasoning_content"] = reasoning_content
        return message

    def stop_token_ids(self) -> list[int]:
        return [END]

    def decode_spans(self, token_ids: Sequence[int], context: Sequence[int] = ()) -> tuple[str, list[int]]:
        return join_pieces([SPECIAL_TEXT[t] if t in SPECIAL_TEXT else chr(t - 100) for t in token_ids])
