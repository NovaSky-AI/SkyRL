"""Identity hashes for the context graph.

A trajectory's graph dedupes by content, so what counts as "the same message"
is decided here, once. Hashes are taken over canonical JSON: sorted keys, and
top-level keys whose value is ``None``, ``[]`` or ``{}`` dropped, because
clients disagree about spelling "absent". An SDK replaying an assistant
message may drop ``"refusal": null`` or ``"annotations": []`` that the server
sent; those are the same message, and hashing them apart would fork the graph
on every turn.

Nothing is canonicalized across providers: one trajectory speaks one dialect.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
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


def message_hash(message: Mapping[str, Any]) -> str:
    return digest(canonical_message(message))


def tools_hash(tools: Sequence[Mapping[str, Any]] | None) -> str:
    """Order matters: a chat template renders tools in the order given."""
    return digest(list(tools)) if tools else ""


def match_hash(message: Mapping[str, Any], *, tools: str, model: str | None) -> str:
    """What a message is matched by: itself, the tool set and the model it was seen with.

    Tools, because a chat template renders their schemas into the prompt. The
    model, because two models answering alike are still two samples.
    """
    return digest([message_hash(message), tools, model or ""])


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


def client_token_delta_hash(match: str, token_ids: Sequence[int]) -> str:
    """A client node's identity in token mode: its match hash and its exact tokens.

    Two identical messages that tokenized differently are two nodes, so a path's
    tokens are always the tokens its nodes were committed with.
    """
    return digest(["client", match, _token_digest(token_ids)])


def model_token_delta_hash(
    match: str, sampling: Mapping[str, Any] | None, token_ids: Sequence[int], sampled_start: int
) -> str:
    return digest(["model", match, sampling_key(sampling), sampled_start, _token_digest(token_ids)])
