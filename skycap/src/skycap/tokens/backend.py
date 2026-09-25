"""Token mode: skycap renders the prompt, calls a token-in/token-out engine, and answers in OpenAI shape.

Turns of one trajectory run one at a time, engine call included, because turn
N+1 extends turn N's exact tokens. Different trajectories run in parallel;
rendering runs on threads over a pool of renderers.

A turn is committed before it's answered. If its tokens can't be attributed
exactly, the harness still gets the completion (the engine generated it), but
the trajectory is marked ``failed``: it accepts no further turns and
``finish`` reports the status, so no approximation reaches training.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from typing import Any

import aiohttp
import orjson
from aiohttp import web

from skycap import hashing
from skycap.graph import CallInfo
from skycap.openai_chat import ChatRequest, error_body
from skycap.tokens import response, turn
from skycap.tokens.engine import EngineError, VLLMEngine
from skycap.tokens.renderer import TokenRenderer
from skycap.trajectory import Failure, Trajectory

logger = logging.getLogger(__name__)

STATUS_HEADER = "x-skycap-status"


def _error(message: str, status: int, *, kind: str = "invalid_request_error", code: str | None = None) -> web.Response:
    return web.Response(body=error_body(message, kind=kind, code=code), status=status, content_type="application/json")


class TokensBackend:
    def __init__(
        self,
        engine_url: str,
        renderer: TokenRenderer,
        *,
        engine: VLLMEngine | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_model_len: int | None = None,
        sampling_overrides: Mapping[str, Any] | None = None,
        sampling_mask: bool = False,
        logprobs_mode: str = "processed_logprobs",
    ) -> None:
        self.engine_url = engine_url.rstrip("/")
        self.renderer = renderer
        self.engine = engine or VLLMEngine()
        self.api_key = api_key
        self.model = model
        self.max_model_len = max_model_len
        self.sampling_overrides = dict(sampling_overrides or {})
        self.sampling_mask = sampling_mask
        self.logprobs_mode = logprobs_mode
        self._locks: dict[str, asyncio.Lock] = {}
        self._session: aiohttp.ClientSession | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "tokens",
            "engine": self.engine.name,
            "tokenizer": self.renderer.name,
            "logprobs_mode": self.logprobs_mode,
            "sampling_overrides": self.sampling_overrides,
        }

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30),
            connector=aiohttp.TCPConnector(limit=0),
        )

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        assert self._session is not None, "backend not started"
        return self._session

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def release(self, trajectory: Trajectory) -> None:
        """Tell the engine's router this trajectory's session is over. Best effort."""
        self._locks.pop(trajectory.id, None)
        if self.engine.release_path is None:
            return
        try:
            async with self.session.post(
                f"{self.engine_url}{self.engine.release_path}",
                params={"session_id": trajectory.id},
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as released:
                await released.read()
                if released.status >= 400:
                    logger.warning("releasing session %s: HTTP %d", trajectory.id, released.status)
        except (aiohttp.ClientError, TimeoutError) as error:
            logger.warning("releasing session %s failed: %s", trajectory.id, error)

    async def finalize(self, trajectory: Trajectory) -> None:
        """Record each token node's text and per-token byte offsets, so the record reads without a tokenizer."""
        await asyncio.to_thread(self._decode_nodes, trajectory)

    #: Tokens from the parent node that prime a node's decode.
    DECODE_CONTEXT = 8

    def _decode_nodes(self, trajectory: Trajectory) -> None:
        graph = trajectory.graph
        for node in list(graph):
            tokens = node.tokens
            if tokens is None or tokens.text is not None:
                continue
            parent = graph.nodes[node.parent].tokens if node.parent is not None else None
            context = parent.token_ids[-self.DECODE_CONTEXT :] if parent is not None else []
            tokens.text, tokens.text_offsets = self.renderer.decode_spans(tokens.token_ids, context)

    async def models(self, request: web.Request) -> web.Response:
        """The configured ``--model``, or else the engine's own model list."""
        if self.model:
            return web.json_response(
                {"object": "list", "data": [{"id": self.model, "object": "model", "owned_by": "skycap"}]}
            )
        try:
            async with self.session.get(f"{self.engine_url}/v1/models", headers=self._headers()) as up:
                return web.Response(body=await up.read(), status=up.status, content_type=up.content_type)
        except aiohttp.ClientError as error:
            return _error(f"engine: {error}", 502, kind="api_error")

    async def chat(
        self, trajectory: Trajectory, request: web.Request, chat: ChatRequest, raw: bytes
    ) -> web.StreamResponse:
        lock = self._locks.setdefault(trajectory.id, asyncio.Lock())
        async with lock:
            if not trajectory.is_open:
                return _error(f"trajectory is {trajectory.status}", 410, code="trajectory_closed")
            return await self._turn(trajectory, request, chat)

    async def _turn(self, trajectory: Trajectory, request: web.Request, chat: ChatRequest) -> web.StreamResponse:
        started = time.time()
        model = self.model or chat.model
        graph = trajectory.graph
        tools_key = hashing.tools_hash(chat.tools)
        if tools_key and chat.tools is not None:
            graph.tools.setdefault(tools_key, [dict(tool) for tool in chat.tools])
        matches = turn.match_hashes(chat.messages, tools_key, model)
        try:
            planned = await asyncio.to_thread(turn.plan, graph, self.renderer, chat.messages, chat.tools, matches)
        except turn.TokenError as error:
            return _error(str(error), 400)

        # `max_completion_tokens` is an alias of `max_tokens`; resolving it on each side before
        # merging is what lets an override win over a caller's alias.
        sampling = {**_resolve_max_tokens(chat.sampling), **_resolve_max_tokens(self.sampling_overrides)}
        max_tokens = sampling.get("max_tokens")
        if self.max_model_len is not None:
            room = self.max_model_len - len(planned.prompt_ids)
            if room <= 0:
                return _error(
                    f"prompt of {len(planned.prompt_ids)} tokens leaves no room within max_model_len="
                    f"{self.max_model_len}",
                    400,
                    code="context_length_exceeded",
                )
            max_tokens = min(max_tokens, room) if max_tokens else room
        if max_tokens:
            sampling["max_tokens"] = max_tokens
        sampling.setdefault("stop_token_ids", self.renderer.stop_token_ids())

        body = self.engine.request(
            prompt_ids=planned.prompt_ids,
            sampling=sampling,
            model=model,
            cache_salt=chat.body.get("cache_salt"),
            sampling_mask=self.sampling_mask,
        )
        try:
            async with self.session.post(
                f"{self.engine_url}{self.engine.generate_path}",
                data=orjson.dumps(body),
                headers={
                    **self._headers(),
                    "Content-Type": "application/json",
                    "X-Session-ID": trajectory.id,
                },
            ) as up:
                raw = await up.read()
                if up.status != 200:
                    self._fail(trajectory, up.status, raw.decode(errors="replace")[:2000])
                    return _error(
                        f"engine returned HTTP {up.status}",
                        502 if up.status < 500 else up.status,
                        kind="api_error",
                    )
                output = self.engine.parse(orjson.loads(raw))
        except (aiohttp.ClientError, EngineError, orjson.JSONDecodeError) as error:
            self._fail(trajectory, None, f"engine: {error}")
            return _error(f"engine: {error}", 502, kind="api_error")

        reply = await asyncio.to_thread(self.renderer.parse, output.completion_ids, chat.tools)
        reason = response.finish_reason(output.finish_reason, reply)
        call = CallInfo(
            t_start=started,
            t_end=time.time(),
            model=model,
            sampling={k: v for k, v in sampling.items() if k != "stop_token_ids"},
            usage={"prompt_tokens": len(planned.prompt_ids), "completion_tokens": len(output.completion_ids)},
            finish_reason=reason,
            tools=tools_key or None,
        )
        status = "ok"
        if trajectory.is_open:
            try:
                turn.commit(
                    graph,
                    planned,
                    messages=chat.messages,
                    matches=matches,
                    reply=reply,
                    reply_match=hashing.match_hash(reply, tools=tools_key, model=model),
                    output=output,
                    call=call,
                )
            except turn.TokenError as error:
                logger.warning("trajectory %s failed: %s", trajectory.id, error)
                self._fail(trajectory, None, f"token attribution: {error}")
                trajectory.seal("failed")
                status = "failed"
        body_out = response.completion(
            reply,
            model=chat.model or model,
            reason=reason,
            prompt_tokens=len(planned.prompt_ids),
            completion_tokens=len(output.completion_ids),
        )
        headers = {STATUS_HEADER: status}
        if not chat.stream:
            return web.Response(body=orjson.dumps(body_out), content_type="application/json", headers=headers)
        stream = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", **headers}
        )
        await stream.prepare(request)
        for frame in response.stream_frames(body_out):
            await stream.write(frame)
        await stream.write_eof()
        return stream

    @staticmethod
    def _fail(trajectory: Trajectory, status: int | None, error: str) -> None:
        if trajectory.is_open:
            trajectory.failures.append(Failure(t=time.time(), status=status, error=error))


def _resolve_max_tokens(sampling: Mapping[str, Any]) -> dict[str, Any]:
    """``sampling`` with ``max_completion_tokens`` folded into ``max_tokens``, which it takes precedence over."""
    resolved = dict(sampling)
    alias = resolved.pop("max_completion_tokens", None)
    if alias is not None:
        resolved["max_tokens"] = alias
    return resolved
