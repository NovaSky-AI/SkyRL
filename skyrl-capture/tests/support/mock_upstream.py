"""Offline provider stand-ins.

The whole test suite and the benchmark run against these, so nothing requires
an API key, a network, or a downloaded model. They are raw ASGI apps for the
same reason the data plane is: in the benchmark the mock upstream must not be
the bottleneck.

Implemented surfaces:

* OpenAI Chat Completions, streaming and non-streaming
* OpenAI Responses
* Anthropic Messages, streaming and non-streaming
* A token-in/token-out ``/generate`` endpoint for token capture

Behaviour can be steered per request with ``x-mock-*`` headers so tests can
provoke errors, latency, and specific token output without a separate server.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

import orjson

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

JSON = [(b"content-type", b"application/json")]
SSE = [(b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache")]


class MockUpstream:
    """Configurable fake provider."""

    def __init__(
        self,
        *,
        reply: str = "mock reply",
        chunk_delay: float = 0.0,
        first_token_delay: float = 0.0,
        chunks: int = 4,
        require_credential: str | None = None,
        session_lengths: Any = None,
    ) -> None:
        self.reply = reply
        self.chunk_delay = chunk_delay
        self.first_token_delay = first_token_delay
        self.chunks = chunks
        self.require_credential = require_credential
        self.requests: list[dict[str, Any]] = []
        self.record_requests = True
        self.counter = 0
        # Serving-latency model for the tokens route. Zero means answer at
        # once, which is what every test wants.
        self._tpot_ms = float(os.environ.get("MOCK_ENGINE_TPOT_MS", "0") or 0)
        self._ttft_ms_per_100 = float(os.environ.get("MOCK_ENGINE_TTFT_MS_PER_100", "0") or 0)
        # What an engine pays to turn prompt *text* into tokens. A token-in
        # endpoint never pays it, because the proxy already did the work -- so
        # leaving it at zero hides the one cost token-in/token-out removes and
        # makes any text-vs-token comparison a foregone conclusion.
        #
        # Unlike TTFT, a prefix cache does not help here: caching saves
        # attention compute over tokens the engine has already seen, but it must
        # still tokenize the whole prompt to know what it was sent. So this is
        # charged on every prompt token, every turn.
        self._tokenize_ms_per_1k = float(os.environ.get("MOCK_ENGINE_TOKENIZE_MS_PER_1K", "0") or 0)
        # Prefix cache, per session: how many tokens this session's last call
        # left behind. A length rather than the tokens themselves, because a
        # conversation only ever appends -- so the shared prefix is exactly the
        # shorter of the two -- and because an integer can be shared across
        # worker processes where a growing tuple cannot.
        self._session_lengths: Any = {} if session_lengths is None else session_lengths
        self.uncached_tokens = 0
        self.cached_tokens = 0
        # Prompt tokens the engine had to tokenize itself. Zero on the token
        # path by construction: that is the saving being measured.
        self.tokenized_tokens = 0

    # -- ASGI --------------------------------------------------------------
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        path = scope["path"]
        headers = {name.decode(): value.decode() for name, value in scope["headers"]}
        body = await _read_body(receive)
        payload: dict[str, Any] = {}
        if body:
            try:
                payload = orjson.loads(body)
            except orjson.JSONDecodeError:
                payload = {}
        self.counter += 1
        if self.record_requests:
            self.requests.append({"path": path, "headers": headers, "body": payload, "raw": body})

        if self.require_credential is not None:
            presented = headers.get("authorization", "").removeprefix("Bearer ") or headers.get("x-api-key", "")
            if presented != self.require_credential:
                await _json(send, 401, {"error": {"message": "bad upstream credential", "type": "auth"}})
                return

        # Per-request behaviour overrides for tests.
        status = int(headers.get("x-mock-status", "200"))
        delay = float(headers.get("x-mock-delay", "0"))
        if delay:
            await asyncio.sleep(delay)
        if status >= 400:
            await _json(send, status, {"error": {"message": "mock provider error", "type": "server_error"}})
            return

        if path.endswith("/chat/completions"):
            await self._chat_completions(send, payload, headers)
        elif path.endswith("/responses"):
            await self._responses(send, payload)
        elif path.endswith("/messages"):
            await self._messages(send, payload, headers)
        elif path.endswith("/generate"):
            await self._generate(send, payload, headers)
        elif path.endswith("/models"):
            await _json(send, 200, {"object": "list", "data": [{"id": "mock-model", "object": "model"}]})
        else:
            await _json(send, 404, {"error": {"message": f"mock: no route {path}", "type": "invalid_request_error"}})

    # -- OpenAI ------------------------------------------------------------
    def _reply_text(self, headers: dict[str, str], payload: dict[str, Any] | None = None) -> str:
        """The reply, sized to `max_tokens` when the latency model is on.

        Both arms have to generate the same number of output tokens or the
        comparison is between two different workloads. The token path already
        honours `max_tokens` (capped by MOCK_ENGINE_MAX_COMPLETION); without
        this the text path answered every request with the same short fixture
        and looked faster for a reason that had nothing to do with capture.
        """
        override = headers.get("x-mock-reply")
        if override is not None:
            return override
        if payload is None or self._tpot_ms <= 0:
            return self.reply
        ceiling = int(os.environ.get("MOCK_ENGINE_MAX_COMPLETION", "16") or 16)
        wanted = payload.get("max_tokens") or payload.get("max_completion_tokens") or ceiling
        tokens = max(1, min(int(wanted), ceiling))
        # `_estimate_tokens` reads this back at _CHARS_PER_TOKEN, so the text
        # arm's completion counts the same as the token arm's.
        return ("mock " * tokens)[: tokens * _CHARS_PER_TOKEN]

    async def _chat_completions(self, send: Send, payload: dict[str, Any], headers: dict[str, str]) -> None:
        model = payload.get("model", "mock-model")
        text = self._reply_text(headers, payload)
        delay = self._text_delay(payload, text)
        if delay > 0:
            await asyncio.sleep(delay)
        identifier = f"chatcmpl-mock{self.counter:06d}"
        if payload.get("stream"):
            await send({"type": "http.response.start", "status": 200, "headers": SSE})
            if self.first_token_delay:
                await asyncio.sleep(self.first_token_delay)
            pieces = _split(text, self.chunks)
            for index, piece in enumerate(pieces):
                frame = {
                    "id": identifier,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": ({"role": "assistant", "content": piece} if index == 0 else {"content": piece}),
                            "finish_reason": None,
                        }
                    ],
                }
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"data: " + orjson.dumps(frame) + b"\n\n",
                        "more_body": True,
                    }
                )
                if self.chunk_delay:
                    await asyncio.sleep(self.chunk_delay)
            final = {
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": _estimate_prompt_tokens(payload),
                    "completion_tokens": len(text.split()),
                    "total_tokens": _estimate_prompt_tokens(payload) + len(text.split()),
                },
            }
            await send(
                {"type": "http.response.body", "body": b"data: " + orjson.dumps(final) + b"\n\n", "more_body": True}
            )
            await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return

        prompt_tokens = _estimate_prompt_tokens(payload)
        response = {
            "id": identifier,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": _assistant_message(text, headers),
                    "finish_reason": "tool_calls" if "x-mock-tool-arguments" in headers else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(text.split()),
                "total_tokens": prompt_tokens + len(text.split()),
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
        await _json(send, 200, response, extra_headers=[(b"x-request-id", identifier.encode())])

    async def _responses(self, send: Send, payload: dict[str, Any]) -> None:
        identifier = f"resp-mock{self.counter:06d}"
        response = {
            "id": identifier,
            "object": "response",
            "created_at": int(time.time()),
            "model": payload.get("model", "mock-model"),
            "status": "completed",
            "previous_response_id": payload.get("previous_response_id"),
            "output": [
                {
                    "id": f"msg-{identifier}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": self.reply, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
        }
        await _json(send, 200, response, extra_headers=[(b"x-request-id", identifier.encode())])

    # -- Anthropic ---------------------------------------------------------
    async def _messages(self, send: Send, payload: dict[str, Any], headers: dict[str, str]) -> None:
        text = self._reply_text(headers)
        identifier = f"msg_mock{self.counter:06d}"
        model = payload.get("model", "mock-claude")
        if payload.get("stream"):
            await send({"type": "http.response.start", "status": 200, "headers": SSE})
            if self.first_token_delay:
                await asyncio.sleep(self.first_token_delay)

            async def event(name: str, data: dict[str, Any]) -> None:
                frame = b"event: " + name.encode() + b"\ndata: " + orjson.dumps(data) + b"\n\n"
                await send({"type": "http.response.body", "body": frame, "more_body": True})

            await event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": identifier,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": 14, "output_tokens": 0},
                    },
                },
            )
            await event(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            )
            for piece in _split(text, self.chunks):
                await event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}},
                )
                if self.chunk_delay:
                    await asyncio.sleep(self.chunk_delay)
            await event("content_block_stop", {"type": "content_block_stop", "index": 0})
            await event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": len(text.split())},
                },
            )
            await event("message_stop", {"type": "message_stop"})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return

        response = {
            "id": identifier,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": _estimate_prompt_tokens(payload),
                "output_tokens": len(text.split()),
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        }
        await _json(send, 200, response, extra_headers=[(b"request-id", identifier.encode())])

    # -- token capture --------------------------------------------------------------
    def _text_delay(self, payload: dict[str, Any], reply: str) -> float:
        """The same engine, reached through its text API.

        Identical generation cost to `_engine_delay`, plus the tokenization a
        token-in endpoint skips. Prefix caching applies to the attention work,
        as it does on the token path, so both arms get it -- the honest
        difference between them is who tokenized, not who cached.

        A conversation is keyed on its first message, which is stable across
        the turns of one session and differs between sessions. That is what a
        prefix-caching engine keys on in effect, without the mock having to be
        told about trajectories.
        """
        if self._tpot_ms <= 0 and self._ttft_ms_per_100 <= 0 and self._tokenize_ms_per_1k <= 0:
            return 0.0
        messages = payload.get("messages") or []
        prompt_tokens = _estimate_tokens(messages)
        completion_tokens = max(1, len(reply) // _CHARS_PER_TOKEN)
        session = ""
        if messages:
            first = messages[0]
            session = hashlib.blake2b(
                repr(first.get("content", "")).encode(), digest_size=8
            ).hexdigest()
        shared = min(int(self._session_lengths.get(session, 0)), prompt_tokens)
        self._session_lengths[session] = prompt_tokens + completion_tokens
        uncached = prompt_tokens - shared
        self.uncached_tokens += uncached
        self.cached_tokens += shared
        self.tokenized_tokens += prompt_tokens
        return (
            (uncached / 100.0) * (self._ttft_ms_per_100 / 1000.0)
            + completion_tokens * (self._tpot_ms / 1000.0)
            + (prompt_tokens / 1000.0) * (self._tokenize_ms_per_1k / 1000.0)
        )

    def _engine_delay(self, prompt_ids: list[int], completion: list[int], session: str) -> float:
        """A serving engine's latency, to the shape that matters here.

        Time to first token is paid only on the part of the prompt the engine
        has not already got in its prefix cache, which is what makes reusing a
        token prefix worth anything. Everything after that is one tick per
        generated token.

        Off unless MOCK_ENGINE_TPOT_MS is set, so the test suite stays fast.
        """
        if self._tpot_ms <= 0 and self._ttft_ms_per_100 <= 0:
            return 0.0
        shared = min(int(self._session_lengths.get(session, 0)), len(prompt_ids))
        # What this session now holds: the prompt plus what was just generated,
        # which is the prefix the next turn extends.
        self._session_lengths[session] = len(prompt_ids) + len(completion)
        uncached = len(prompt_ids) - shared
        self.uncached_tokens += uncached
        self.cached_tokens += shared
        return (uncached / 100.0) * (self._ttft_ms_per_100 / 1000.0) + len(completion) * (
            self._tpot_ms / 1000.0
        )

    async def _generate(self, send: Send, payload: dict[str, Any], headers: dict[str, str]) -> None:
        """Token-in/token-out inference.

        Accepts ``prompt_token_ids`` and returns exact completion token IDs
        with per-token logprobs, mirroring the contract a SkyRL-style router
        exposes. The reply tokens are derived deterministically from the prompt
        so tests can assert exact token equality.
        """
        prompt_ids = payload.get("prompt_token_ids") or []
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        sampling = payload.get("sampling_params") or {}
        # The tokens route builds its own upstream headers and does not forward
        # the client's, so behaviour here is steered by the payload the proxy
        # actually sends -- the model name.
        model = str(payload.get("model") or "")
        override = headers.get("x-mock-completion-ids")
        if override:
            completion = [int(item) for item in override.split(",") if item.strip()]
        else:
            completion = _mock_completion_ids(prompt_ids, sampling)
        logprobs = [round(-0.05 * ((index % 7) + 1), 6) for index in range(len(completion))]
        delay = self._engine_delay(prompt_ids, completion, str(payload.get("session_id") or ""))
        if delay > 0:
            await asyncio.sleep(delay)
        response: dict[str, Any] = {
            "response_ids": [completion],
            "response_logprobs": [logprobs],
            "stop_reasons": [
                "context_length" if "overlong" in model else headers.get("x-mock-stop-reason", "stop")
            ],
        }
        if "moe" in model or headers.get("x-mock-routed-experts"):
            # Two layers, two experts per token, over prompt + completion.
            total = len(prompt_ids) + len(completion)
            response["rollout_expert_indices"] = [
                [[[index % 8, (index + 1) % 8], [(index + 2) % 8, (index + 3) % 8]] for index in range(total)]
            ]
        await _json(send, 200, response)


def _mock_completion_ids(prompt_ids: list[int], sampling: dict[str, Any]) -> list[int]:
    """Deterministic pseudo-completion derived from the prompt."""
    # Capped so a stray max_tokens cannot make a test allocate forever. The
    # benchmark raises it, because output length is most of a turn's latency.
    ceiling = int(os.environ.get("MOCK_ENGINE_MAX_COMPLETION", "16") or 16)
    limit = int(sampling.get("max_tokens") or 8)
    limit = max(1, min(limit, ceiling))
    seed = sum(prompt_ids) % 997 if prompt_ids else 1
    return [1000 + (seed + index * 7) % 500 for index in range(limit)]


def _split(text: str, parts: int) -> list[str]:
    if parts <= 1 or not text:
        return [text]
    size = max(1, len(text) // parts)
    pieces = [text[index : index + size] for index in range(0, len(text), size)]
    return pieces or [text]


def _estimate_prompt_tokens(payload: dict[str, Any]) -> int:
    messages = payload.get("messages") or []
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += max(1, len(content) // 4)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += max(1, len(str(block.get("text", ""))) // 4)
    return total or 8


# Characters per token, for the text path, which has no tokenizer. The token
# path counts real token IDs, so if this disagrees with the workload's actual
# ratio the two arms are charged for different amounts of input and the
# difference reads as proxy overhead. Set it to match the traffic being
# replayed: ordinary English runs near 5, dense markers nearer 2.
_CHARS_PER_TOKEN = max(1, int(os.environ.get("MOCK_CHARS_PER_TOKEN", "4") or 4))


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Prompt tokens a chat request would render to, near enough to charge for."""
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif content is not None:
            total += len(repr(content))
        for call in message.get("tool_calls") or ():
            total += len(repr(call))
        if reasoning := message.get("reasoning_content"):
            total += len(reasoning)
        total += 8  # role and scaffold
    return max(1, total // _CHARS_PER_TOKEN)


async def _read_body(receive: Receive) -> bytes:
    parts = []
    while True:
        message = await receive()
        if message["type"] == "http.request":
            parts.append(message.get("body") or b"")
            if not message.get("more_body", False):
                break
        elif message["type"] == "http.disconnect":
            break
    return b"".join(parts)


async def _json(
    send: Send,
    status: int,
    payload: dict[str, Any],
    *,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = orjson.dumps(payload)
    headers = list(JSON)
    if extra_headers:
        headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body, "more_body": False})


def _assistant_message(text: str, headers: dict[str, str]) -> dict[str, Any]:
    """A plain reply, or a tool call whose arguments the caller chose.

    ``x-mock-tool-arguments`` exists so a test can sample a *structurally*
    malformed tool call and then repair only ``arguments`` -- the shape real
    tool repair takes, as opposed to editing message text.
    """
    arguments = headers.get("x-mock-tool-arguments")
    if arguments is None:
        return {"role": "assistant", "content": text}
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": headers.get("x-mock-tool-call-id", "call_1"),
                "type": "function",
                "function": {"name": headers.get("x-mock-tool-name", "search"), "arguments": arguments},
            }
        ],
    }
