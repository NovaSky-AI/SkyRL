"""API dialects: one module per wire format a harness can speak to skycap (``skycap.dialects.base``)."""

from skycap.dialects.base import (
    SAMPLING_KEYS,
    Assembler,
    CallRequest,
    Dialect,
    DialectType,
    Reply,
    RequestError,
    Turn,
    error_body,
)

__all__ = [
    "SAMPLING_KEYS",
    "Assembler",
    "CallRequest",
    "Dialect",
    "DialectType",
    "Reply",
    "RequestError",
    "Turn",
    "error_body",
]
