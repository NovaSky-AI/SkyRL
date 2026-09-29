"""A deterministic stand-in for a ``renderers`` renderer, for tests only.

Characters are tokens (``ord(c) + 100``) and three specials frame a message:

    START role NL content END NL

The generation prompt is ``START "assistant" NL``. A completion is the
content followed by ``END``. Reasoning renders as ``THINK:<text>|`` before the
content, and a completion beginning ``CALL:<name>:<args>`` parses as a tool
call. The bridge keeps the contract the real library does: the previous
prompt and completion come back unchanged, it declines a completion that
doesn't end in ``END``, and it attributes only the new messages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from skycap.tokens.renderer import Rendered, join_pieces

START, END, NL = 1, 2, 3
SPECIAL_TEXT = {START: "<s>", END: "</s>", NL: "\n"}


def encode(text: str) -> list[int]:
    return [ord(c) + 100 for c in text]


def decode(tokens: Sequence[int]) -> str:
    return "".join(chr(t - 100) for t in tokens if t >= 100)


def _body(message: Mapping[str, Any]) -> str:
    text = message.get("content") or ""
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

    def _message(self, message: Mapping[str, Any]) -> list[int]:
        return [START, *encode(str(message.get("role"))), NL, *encode(_body(message)), END, NL]

    def render(self, messages: Sequence[Mapping[str, Any]], tools: Any) -> Rendered:
        tokens: list[int] = []
        indices: list[int] = []
        for index, message in enumerate(messages):
            chunk = self._message(message)
            tokens += chunk
            indices += [index] * len(chunk)
        prompt = [START, *encode("assistant"), NL]
        tokens += prompt
        indices += [-1] * len(prompt)
        if self.corrupt:
            indices[0] = len(messages) + 5
        return Rendered(token_ids=tokens, tail_indices=indices)

    def bridge(
        self,
        previous_prompt: Sequence[int],
        previous_completion: Sequence[int],
        new_messages: Sequence[Mapping[str, Any]],
        tools: Any,
    ) -> Rendered | None:
        if not previous_completion or previous_completion[-1] != END or self.corrupt:
            return None
        if any(m.get("role") == "assistant" for m in new_messages):
            return None
        tail = self.render(new_messages, tools)
        tokens = [*previous_prompt, *previous_completion, NL, *tail.token_ids]
        indices = [-1, *tail.tail_indices]
        return Rendered(token_ids=tokens, tail_indices=indices, reused=len(previous_prompt) + len(previous_completion))

    def parse(self, completion_ids: Sequence[int], tools: Any) -> dict[str, Any]:
        text = decode([t for t in completion_ids if t != END])
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if text.startswith("THINK:") and "|" in text:
            reasoning, _, text = text[6:].partition("|")
            message = {"role": "assistant", "content": text, "reasoning_content": reasoning}
        if text.startswith("CALL:"):
            name, _, arguments = text[5:].partition(":")
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_0", "type": "function", "function": {"name": name, "arguments": arguments}}
                ],
            }
        return message

    def stop_token_ids(self) -> list[int]:
        return [END]

    def decode_spans(self, token_ids: Sequence[int], context: Sequence[int] = ()) -> tuple[str, list[int]]:
        return join_pieces([SPECIAL_TEXT[t] if t in SPECIAL_TEXT else chr(t - 100) for t in token_ids])
