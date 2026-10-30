"""Identity hashes for the context graph.

A trajectory's graph dedupes by content, so what counts as "the same message"
is decided here, once. Hashes are taken over canonical JSON: sorted keys, and
top-level keys whose value is ``None``, ``[]`` or ``{}`` dropped, because
clients disagree about spelling "absent". An SDK replaying an assistant
message may drop ``"refusal": null`` or ``"annotations": []`` that the server
sent; those are the same message, and hashing them apart would fork the graph
on every turn. Token mode goes further and hashes only the fields a renderer
reads (``MatchKey.tokens``).

Two hashes are taken per node: a match hash, by a ``MatchKey``, and a delta
hash (``*_delta_hash``), which builds on it.

Nothing is canonicalized across providers: one trajectory speaks one dialect.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import orjson

#: Request fields that shape the distribution a model node was sampled from.
#: Two identical outputs under different values are different samples.
SAMPLING_IDENTITY_KEYS = ("temperature", "top_p", "top_k", "min_p")


def canonical_bytes(value: Any) -> bytes:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


def digest(value: Any) -> str:
    return hashlib.blake2b(canonical_bytes(value), digest_size=16).hexdigest()


def canonical_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """The message with empty top-level values dropped."""
    return {key: value for key, value in message.items() if value not in (None, [], {})}


def tools_hash(tools: Sequence[Mapping[str, Any]] | None) -> str:
    """Order matters: a chat template renders tools in the order given."""
    return digest(list(tools)) if tools else ""


@dataclass(frozen=True, slots=True)
class MatchKey:
    """What one call's messages are matched by: the fields that count, the call's tool set and its model.

    Calling it on a message gives the message's match hash (see ``MessageGraph``).
    Tools count because a chat template renders their schemas into the prompt;
    the model, because two models answering alike are still two samples. Which
    fields count depends on the mode. Text mode sends the whole message
    upstream, so every field counts (``text``). Token mode renders the message
    itself, so only the fields a renderer reads count (``tokens``). Two
    spellings that differ in a rendered field, such as ``content: ""`` against
    no ``content``, hash apart in both; whether a chat template renders them
    alike is decided when a token-mode request is planned (``tokens.turn``).
    """

    #: The call's ``tools_hash``.
    tools: str
    model: str | None
    rendered_only: bool

    @classmethod
    def text(cls, tools: str, model: str | None) -> MatchKey:
        return cls(tools, model, rendered_only=False)

    @classmethod
    def tokens(cls, tools: str, model: str | None) -> MatchKey:
        return cls(tools, model, rendered_only=True)

    def fields(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """The fields of ``message`` that count, with empty values dropped."""
        return canonical_message(rendered_fields(message) if self.rendered_only else message)

    def __call__(self, message: Mapping[str, Any]) -> str:
        return digest([digest(self.fields(message)), self.tools, self.model or ""])


#: The message fields a ``renderers`` renderer reads: at the top level, in a tool call, and in a
#: tool call's ``function``. Token mode matches a message on these alone. Anything else is client
#: metadata (LiteLLM's ``provider_specific_fields``, the SDK's ``refusal`` and ``annotations``)
#: and can't change the tokens. Text mode matches on every field, because its upstream sees them all.
RENDERED_FIELDS = frozenset(
    {
        "role",
        "content",
        "reasoning_content",
        # Reasoning under its other name (DeepSeek V4, Gemma 4, Hunyuan 3, Laguna).
        "reasoning",
        "name",
        "tool_calls",
        # On a tool result: the call it answers. DeepSeek V4, GLM 5 and Gemma 4 pair results with calls by it.
        "tool_call_id",
        # Gemma 4: tool results carried on the assistant message.
        "tool_responses",
        # DeepSeek V4: ``task`` emits a task token after the message, ``wo_eos`` omits the
        # end-of-turn token, and ``response_format`` renders a JSON schema into the message.
        "task",
        "wo_eos",
        "response_format",
    }
)
#: ``id`` is rendered by Kimi K2 and K2.5, and pairs calls with results in DeepSeek V4, GLM 5 and Gemma 4.
#: The flat ``name`` / ``arguments`` / ``tool_call_id`` are the spellings renderers accept besides ``function``.
RENDERED_TOOL_CALL_FIELDS = frozenset({"id", "function", "name", "arguments", "tool_call_id"})
RENDERED_FUNCTION_FIELDS = frozenset({"name", "arguments"})


def rendered_fields(message: Mapping[str, Any]) -> dict[str, Any]:
    """The part of ``message`` a renderer reads."""
    kept = _pick(message, RENDERED_FIELDS)
    calls = kept.get("tool_calls")
    if isinstance(calls, list):
        kept["tool_calls"] = [_rendered_call(call) for call in calls]
    return kept


def _rendered_call(call: Any) -> Any:
    if not isinstance(call, Mapping):
        return call
    kept = _pick(call, RENDERED_TOOL_CALL_FIELDS)
    function = kept.get("function")
    if isinstance(function, Mapping):
        kept["function"] = _pick(function, RENDERED_FUNCTION_FIELDS)
    return kept


def _pick(mapping: Mapping[str, Any], fields: frozenset[str]) -> dict[str, Any]:
    """The entries of ``mapping`` whose keys are in ``fields``."""
    picked = {}
    for key, value in mapping.items():
        if key in fields:
            picked[key] = value
    return picked


def sampling_key(sampling: Mapping[str, Any] | None) -> dict[str, Any]:
    if not sampling:
        return {}
    return {key: sampling[key] for key in SAMPLING_IDENTITY_KEYS if sampling.get(key) is not None}


def model_delta_hash(match: str, sampling: Mapping[str, Any] | None) -> str:
    """A model node's identity: its match hash plus the sampling support params.

    Always distinct from a client node's delta (which is the bare match hash),
    so a harness-written message and an identical sample stay two nodes: only
    the sample is trainable.
    """
    return digest(["model", match, sampling_key(sampling)])


def _token_digest(token_ids: Sequence[int]) -> str:
    return hashlib.blake2b(np.asarray(token_ids, dtype=np.int64).tobytes(), digest_size=16).hexdigest()


def client_token_delta_hash(match: str, token_ids: Sequence[int], media: Sequence[Any] = ()) -> str:
    """A client node's identity in token mode: its match hash, its exact tokens and its images.

    Two identical messages that tokenized differently are two nodes, so a path's
    tokens are always the tokens its nodes were committed with. Images count too
    (``media``, the node's ``Media`` items): two images of one size have the same
    placeholder tokens, and a node must keep the images its tokens were sent with.
    A node without images hashes as it did before images were captured.
    """
    if not media:
        return digest(["client", match, _token_digest(token_ids)])
    images = [[item.modality, item.offset, item.length, item.hash] for item in media]
    return digest(["client", match, _token_digest(token_ids), images])


def model_token_delta_hash(
    match: str, sampling: Mapping[str, Any] | None, token_ids: Sequence[int], sampled_start: int
) -> str:
    return digest(["model", match, sampling_key(sampling), sampled_start, _token_digest(token_ids)])
