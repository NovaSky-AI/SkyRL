"""Every hash in the system, and the canonical JSON they are taken over.

Kept in the domain because node identity is a domain fact: which messages are
the same message, which contexts are the same context. The parser and the
token trace both compute these, and they have to agree byte for byte.
"""

from __future__ import annotations

import hashlib
from typing import Any

import orjson

_CANONICAL = orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS


def canonical_bytes(value: Any) -> bytes:
    """Deterministic JSON encoding used for every hash in the system."""
    return orjson.dumps(value, option=_CANONICAL)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def bytes_hash(data: bytes) -> str:
    """A hash of raw bytes, for a request body that is never re-encoded.

    Separate from `stable_hash` because that one canonicalizes JSON, and the
    thing being fingerprinted here is the exact bytes a client sent.
    """
    return hashlib.sha256(data).hexdigest()


def normalize_json(value: Any) -> Any:
    """Canonical form: sorted keys, tuples as lists, key order irrelevant.

    One orjson round trip rather than a recursive Python walk. orjson sorts in
    C and renders tuples as arrays, so the result is identical and costs about
    a fifth as much -- and every ingested message goes through here.
    """
    return orjson.loads(canonical_bytes(value))


def normalize_with_hash(value: Any) -> tuple[Any, str]:
    """Canonical form and its hash from one pass, rather than serializing twice."""
    raw = canonical_bytes(value)
    return orjson.loads(raw), hashlib.sha256(raw).hexdigest()


def node_identity(message_hash: str, tools_hash: str | None, model: str | None) -> str:
    """A node's delta hash in text mode: the message, its tools, and the model.

    Node identity has to cover everything that makes two generations different,
    or the graph merges them and reports one call where there were two.

    **Tools**, because a chat template renders their schemas into the prompt,
    so the same message under a different tool set is a different context.

    **Model**, because it produced the generation. It does not change the
    prompt the way tools do, but two models answering the same prompt the same
    way are still two samples -- and a dataset that means to distil a large
    model into a small one has to be able to tell them apart.

    Token mode folds the exact token delta and the sampled boundary in instead;
    see `tito/trace.py`. Two identical messages with different tokenizations
    are different nodes there.
    """
    return stable_hash([message_hash, tools_hash or "", model or ""])


def chain_hash(parent_context_hash: str, delta_hash: str) -> str:
    """Fold one node's delta into its parent's context hash.

    Equal context hashes mean equal full conversations, and unlike the
    ``(trajectory, parent, delta)`` key that identifies a node, this holds
    *across* trajectories -- which is what makes it usable for cross-run
    deduplication and prefix-cache lookup.
    """
    return hashlib.sha256(f"{parent_context_hash}:{delta_hash}".encode()).hexdigest()


def canonical_hash(payload: Any) -> str:
    """The hash an idempotency key is checked against: same body, same hash."""
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()
