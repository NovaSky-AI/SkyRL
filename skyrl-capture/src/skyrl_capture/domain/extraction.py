"""What a text protocol hands back, and what the domain does with it.

This is the provider-independent shape. A protocol adapter reads one raw
request/response pair and returns *its own provider's* messages, unchanged:
OpenAI messages stay OpenAI messages and Anthropic messages stay Anthropic
messages. What it must not do is decide identity -- normalizing, hashing and
counting are domain rules, applied here the same way for every provider, so a
contributed adapter cannot get message identity subtly wrong and a graph built
from one provider means the same thing as a graph built from another.

The division, stated once:

* **the adapter** knows that Anthropic carries the system prompt in a
  top-level field, that Responses calls the input ``input``, and how a
  streaming delta accumulates;
* **the domain** knows what makes two messages the same message.

An adapter that materializes a message out of something that was not one --
Anthropic's ``system``, Responses' ``instructions`` -- says so in
``derivation``, which travels onto the node as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skyrl_capture.domain.hashing import normalize_with_hash


@dataclass(slots=True)
class ExtractedMessage:
    """One message, as its own provider spells it."""

    value: dict[str, Any]
    #: How this message came to exist, when it was not literally in the body.
    derivation: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedExchange:
    """Everything one adapter derives from one request/response pair.

    Every field is optional: parsing is best-effort by design. A body that
    cannot be read, or a provider that returns a shape the adapter does not
    recognise, records the reason in ``errors`` and leaves the rest empty --
    the raw bytes are in the record either way, and a derivation can be
    improved later against them.
    """

    input_messages: list[ExtractedMessage] = field(default_factory=list)
    output_message: ExtractedMessage | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    tools: list[dict[str, Any]] | None = None
    max_output_tokens: int | None = None
    provider_response_id: str | None = None
    previous_response_id: str | None = None
    #: Request fields that are neither context nor transport: what was asked
    #: for, kept whole so a training row can be reproduced.
    parameters: dict[str, Any] = field(default_factory=dict)
    #: The provider's own error envelope, when it returned one.
    provider_error: dict[str, Any] | None = None
    #: Decoded streaming frames, in arrival order, and whatever the adapter
    #: can say about the stream as a whole.
    stream_frames: list[dict[str, Any]] = field(default_factory=list)
    stream_summary: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class NormalizedMessage:
    """An extracted message with the domain's identity applied."""

    message: dict[str, Any]
    role: str
    message_hash: str
    content_block_count: int
    char_count: int
    derivation: dict[str, Any] = field(default_factory=dict)


def normalize(extracted: ExtractedMessage) -> NormalizedMessage:
    """Canonical form, hash and shape counts for one extracted message.

    The hash is over the canonical JSON of the message exactly as its provider
    wrote it. There is deliberately **no cross-provider canonicalization**:
    prefix matching only ever compares messages within one trajectory, and one
    trajectory is one provider, so normalizing across providers would add
    ambiguity for no benefit.
    """
    value, message_hash = normalize_with_hash(extracted.value)
    blocks, characters = count_content(extracted.value.get("content"))
    return NormalizedMessage(
        message=value,
        role=str(extracted.value.get("role") or "unknown"),
        message_hash=message_hash,
        content_block_count=blocks,
        char_count=characters,
        derivation=dict(extracted.derivation),
    )


def count_content(content: Any) -> tuple[int, int]:
    """``(block_count, char_count)`` for a message's content.

    Provider-shaped but not provider-specific: a string is one block, a list is
    its blocks, and the characters are whatever text those blocks carry under
    the keys every provider uses for it.
    """
    if content is None:
        return 0, 0
    if isinstance(content, str):
        return 1, len(content)
    if isinstance(content, list):
        characters = 0
        for block in content:
            if isinstance(block, dict):
                for key in ("text", "thinking", "content"):
                    value = block.get(key)
                    if isinstance(value, str):
                        characters += len(value)
            elif isinstance(block, str):
                characters += len(block)
        return len(content), characters
    return 1, len(str(content))
