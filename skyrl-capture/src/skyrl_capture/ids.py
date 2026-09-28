"""Prefixed, sortable, URL-safe identifiers.

Every ID is ``<prefix>_<26 chars>`` where the payload is a base32 encoding of
a 48-bit millisecond timestamp followed by 80 bits of randomness. IDs sort
lexicographically by generation millisecond (order within one millisecond is
arbitrary) without needing a coordination service, which keeps append-mostly
collections such as exchanges and message nodes cheap to scan in order. The
prefix makes IDs self-describing in logs and object keys.
"""

from __future__ import annotations

import os
import time

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32, no I/L/O/U.
_ENCODED_LEN = 26
_RANDOM_BYTES = 10


def _encode(value: int, length: int) -> str:
    chars = ["0"] * length
    for index in range(length - 1, -1, -1):
        chars[index] = _ALPHABET[value & 0x1F]
        value >>= 5
    return "".join(chars)


def new_id(prefix: str) -> str:
    """Return a fresh sortable ID with ``prefix``."""
    timestamp = int(time.time() * 1000) & ((1 << 48) - 1)
    randomness = int.from_bytes(os.urandom(_RANDOM_BYTES), "big")
    return f"{prefix}_{_encode((timestamp << 80) | randomness, _ENCODED_LEN)}"


def trajectory_id() -> str:
    return new_id("tr")


def exchange_id() -> str:
    return new_id("ex")


def node_id() -> str:
    return new_id("nd")


def export_id() -> str:
    return new_id("exp")


