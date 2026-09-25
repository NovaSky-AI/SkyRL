"""Upstream forwarding and header policy.

The proxy is transparent by default: provider-specific request fields, unknown
paths, and unusual headers all pass through so the raw exchange stays a faithful
replay source. Only three classes of header are touched.

1. **Hop-by-hop headers** are dropped, as any HTTP intermediary must.
2. **Authentication** is replaced. Whatever the client sent is dropped --
   capture authenticates nothing inbound, so an inbound credential is neither
   forwarded nor recorded -- and the configured upstream credential is applied
   instead. A client never learns the upstream key, and the upstream never sees
   what the client sent.
3. **Accept-encoding** is pinned to ``identity``. Capturing a gzip-compressed
   body would leave the stored payload unparseable without an extra decompress
   step on every read, and would break streaming chunk attribution. The
   client's original value is preserved in the exchange so replay can reproduce
   the original request exactly.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Headers an intermediary must not forward.
HOP_BY_HOP = frozenset(
    {
        b"connection",
        b"keep-alive",
        b"proxy-authenticate",
        b"proxy-authorization",
        b"proxy-connection",
        b"te",
        b"trailer",
        b"transfer-encoding",
        b"upgrade",
        b"host",
        b"content-length",
        b"accept-encoding",
        b"authorization",
        b"x-api-key",
    }
)

# Response headers the proxy generates itself rather than copying.
RESPONSE_STRIP = frozenset(
    {
        b"connection",
        b"keep-alive",
        b"transfer-encoding",
        b"content-encoding",
        b"content-length",
    }
)


def prepare_headers(
    raw_headers: list[tuple[bytes, bytes]],
    *,
    protocol: Any,
    credential: str | None,
) -> list[tuple[bytes, bytes]]:
    """Apply the header policy described in this module's docstring.

    Which header carries the credential is the protocol's business, not this
    module's, so it is asked rather than branched on.
    """
    forwarded: list[tuple[bytes, bytes]] = []
    for name, value in raw_headers:
        if name.lower() in HOP_BY_HOP:
            continue
        forwarded.append((name, value))
    forwarded.append((b"accept-encoding", b"identity"))
    if credential:
        forwarded.extend(protocol.auth_headers(credential))
    return forwarded


def filter_response_headers(raw_headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    """Drop the response headers the proxy generates itself.

    Content length and transfer encoding are re-framed for the proxy's own
    client, and content encoding is never present because ``accept-encoding``
    is pinned to ``identity`` upstream.
    """
    out: list[tuple[bytes, bytes]] = []
    for name, value in raw_headers:
        if name.lower() in RESPONSE_STRIP:
            continue
        out.append((name, value))
    return out
