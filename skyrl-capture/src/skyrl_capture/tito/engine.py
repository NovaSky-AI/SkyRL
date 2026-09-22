"""The token-in/token-out inference client.

The upstream contract is deliberately minimal and matches what a SkyRL-style
router exposes: send exact prompt token IDs, receive exact completion token IDs
with per-token selected-token logprobs.

Selected-token logprobs are **required**, not optional. Without them a captured
token capture exchange cannot be used for the training it exists to serve, so a response
that omits them is an error rather than a partial success.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import orjson

from skyrl_capture.tito.protocol import ChatRequest
from skyrl_capture.tito.types import TokenError, TokenUpstreamError
from skyrl_capture.transport.http import TransportError, UpstreamTransport

__all__ = ["TokenEngine", "TokenError", "TokenUpstreamError", "build_sampling_params",
           "canonical_sampling_params"]


def build_sampling_params(
    request: ChatRequest,
    *,
    prompt_token_count: int,
    max_model_len: int,
    stop_token_ids: Sequence[int],
    default_max_tokens: int | None = None,
) -> dict[str, Any]:
    """Build sampling parameters, clamped to the model's context window.

    ``max_tokens`` is clamped against the space actually left after the prompt,
    which is what stops a long context from producing a request the engine must
    reject. The renderer's stop token IDs are always passed so generation ends
    the way the model's own chat format expects.
    """
    remaining = max(0, max_model_len - prompt_token_count)
    requested = request.max_tokens or default_max_tokens or remaining
    max_tokens = max(1, min(int(requested), remaining)) if remaining else 1
    params: dict[str, Any] = {
        "max_tokens": max_tokens,
        "logprobs": True,
    }
    # What the caller sent and capture has no opinion about, offered to the
    # protocol first so a named sampling field always wins over one of these.
    params.update(request.extras)
    params.update(request.sampling)
    if stop_token_ids:
        params["stop_token_ids"] = list(stop_token_ids)
    if remaining == 0:
        params["_context_exhausted"] = True
    return params


def canonical_sampling_params(params: dict[str, Any]) -> dict[str, Any]:
    """Sort keys so the recorded parameters hash deterministically."""
    return orjson.loads(orjson.dumps(params, option=orjson.OPT_SORT_KEYS))


class TokenEngine:
    def __init__(self, transport: UpstreamTransport) -> None:
        self._transport = transport

    async def generate(
        self,
        *,
        protocol: Any,
        url: str,
        credential: str | None,
        prompt_token_ids: Sequence[int],
        sampling_params: dict[str, Any],
        model: str | None,
        session_id: str,
    ) -> dict[str, Any]:
        # The wire belongs to `protocol`: what a request looks like, where
        # session affinity goes, and how a response is read back. Everything
        # here is the part that does not vary -- transport, credentials, and
        # error handling.
        payload, wire_headers = protocol.request(
            prompt_token_ids=prompt_token_ids,
            # Keys starting with `_` are capture's own markers, never sent.
            sampling_params={
                key: value for key, value in sampling_params.items() if not key.startswith("_")
            },
            model=model,
            session_id=session_id,
        )
        url = protocol.url(url)

        headers = [(b"content-type", b"application/json")]
        if credential:
            headers.append((b"authorization", f"Bearer {credential}".encode()))
        if wire_headers:
            named = {name.lower() for name in wire_headers}
            headers = [(k, v) for k, v in headers if k.decode("latin-1").lower() not in named]
            headers.extend(
                (name.encode("latin-1"), value.encode("latin-1"))
                for name, value in wire_headers.items()
            )
        try:
            response = await self._transport.send(
                "POST", url, headers, orjson.dumps(payload), stream=False
            )
        except TransportError as error:
            raise TokenUpstreamError(f"token-in/token-out request failed: {error}") from error
        try:
            content = await response.read()
        finally:
            await response.aclose()
        if response.status_code >= 400:
            raise TokenUpstreamError(
                f"token-in/token-out endpoint returned {response.status_code}: "
                f"{content[:400].decode('utf-8', 'replace')}",
                status=response.status_code,
            )
        try:
            body = orjson.loads(content)
        except orjson.JSONDecodeError as error:
            raise TokenUpstreamError(f"token-in/token-out response was not JSON: {error}") from error
        return protocol.response(body)
