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

from skycap import hashing, retry
from skycap.dialects.base import CallRequest, Dialect, Turn, error_body
from skycap.graph import CallInfo
from skycap.tokens import turn
from skycap.tokens.developer import fold_developer
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
        use_raw_content: bool = False,
    ) -> None:
        """
        ``use_raw_content`` answers with the completion's own text as ``content``, as vLLM does with no
        reasoning or tool-call parser: a thinking model's reasoning stays inline, and tool calls
        stay text. For a harness written against such a server, which replays ``content`` and
        drops ``reasoning_content``; with parsed replies it would edit every turn it replays.
        """
        self.engine_url = engine_url.rstrip("/")
        self.renderer = renderer
        self.engine = engine or VLLMEngine()
        self.api_key = api_key
        self.model = model
        self.max_model_len = max_model_len
        self.sampling_overrides = dict(sampling_overrides or {})
        self.sampling_mask = sampling_mask
        self.logprobs_mode = logprobs_mode
        self.use_raw_content = use_raw_content
        #: Why the renderer's images can't be sent to the engine, found by the first call with images (``_check_media``).
        self.media_error: str | None = None
        #: The chat template ``developer`` messages are folded for, if the renderer exposes one.
        self.chat_template: str | None = getattr(renderer, "chat_template", None)
        self._locks: dict[str, asyncio.Lock] = {}
        self._session: aiohttp.ClientSession | None = None
        #: Whether the first call with images has checked that images can be sent to the engine (``_check_media``).
        self._media_checked = False
        self._media_check_lock = asyncio.Lock()

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "tokens",
            "engine": self.engine.name,
            "tokenizer": self.renderer.name,
            "logprobs_mode": self.logprobs_mode,
            "sampling_overrides": self.sampling_overrides,
            "use_raw_content": self.use_raw_content,
        }

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30),
            connector=aiohttp.TCPConnector(limit=0),
        )

    async def _check_media(self) -> None:
        """Once, before the first call with images: refuse images the engine can't take as rendered here.

        Two checks, both of which would otherwise fail per call or not at all: that the renderer can encode
        images for vLLM (``validate_media_encoding``; not every family can, e.g. Kimi K2.5), and that the
        engine processes images as the renderer does (``_compare_engine_processor``). A failure sets
        ``media_error``, so calls with images get a 400 and text-only calls still work. Not at ``start``, when
        the engine may not answer yet (SkyRL's colocated engines are asleep then). Concurrent first calls wait
        for it, and the flag is set only once it's done, so none sends images unchecked.
        """
        async with self._media_check_lock:
            if self._media_checked:
                return
            try:
                check = getattr(self.renderer, "validate_media_encoding", None)
                error = await asyncio.to_thread(check) if check is not None else None
                if error is None:
                    error = await self._compare_engine_processor()
                if error is not None:
                    logger.error("%s: refusing calls with images: %s", self.renderer.name, error)
                    self.media_error = error
            finally:
                self._media_checked = True

    async def _compare_engine_processor(self) -> str | None:
        """Whether the engine processes images as the renderer does: why not, or None.

        Each of the renderer's ``processor_probes`` is rendered here and sent to the engine's
        ``/v1/chat/completions`` (one token), and the prompt token counts must agree, or the model would see
        images at another resolution than the engine is configured for. A text-only difference is the chat
        template's, and an engine that doesn't answer can't be compared: both only warn. Equal counts don't
        prove equal pixels, but catch different min/max pixels.
        """
        probes = getattr(self.renderer, "processor_probes", None)
        if probes is None or self.media_error:
            return None
        name = self.renderer.name
        try:
            probes = await asyncio.to_thread(probes)
        except Exception as error:  # noqa: BLE001 - e.g. Pillow missing; images then fail per request, text works
            logger.warning("%s: couldn't build the probes to check image processing: %s", name, error)
            return None
        if not probes:
            return None
        if not self.model:
            logger.warning("%s: no model name, so image processing isn't checked against the engine", name)
            return None
        for label, messages in probes:
            try:
                ours = len((await asyncio.to_thread(self.renderer.render, messages, None)).token_ids)
                async with self.session.post(
                    f"{self.engine_url}/v1/chat/completions",
                    json={"model": self.model, "messages": messages, "max_tokens": 1},
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as up:
                    body = await up.json(content_type=None)
                    if up.status != 200:
                        raise RuntimeError(f"HTTP {up.status}: {str(body)[:300]}")
                theirs = int(body["usage"]["prompt_tokens"])
            except Exception as error:  # noqa: BLE001 - whatever the engine or the request raises
                logger.warning("%s: couldn't check image processing against the engine (%s): %r", name, label, error)
                return None
            if ours == theirs:
                continue
            if label == "text":
                logger.warning(
                    "%s: a text-only prompt is %d tokens here and %d in the engine (the chat templates differ), "
                    "so image processing isn't checked against the engine",
                    name,
                    ours,
                    theirs,
                )
                return None
            return (
                f"a {label} prompt is {ours} tokens in skycap and {theirs} in the engine, so the image "
                "processor settings differ. Give skycap the engine's mm_processor_kwargs (processor_kwargs, "
                "--processor-kwargs)."
            )
        return None

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
        self, trajectory: Trajectory, request: web.Request, chat: CallRequest, raw: bytes
    ) -> web.StreamResponse:
        dialect = chat.dialect.get_dialect()
        lock = self._locks.setdefault(trajectory.id, asyncio.Lock())
        async with lock:
            if not trajectory.is_open:
                return dialect.error(chat, f"trajectory is {trajectory.status}", 410, code="trajectory_closed")
            return await self._turn(trajectory, request, chat, dialect)

    async def _turn(
        self, trajectory: Trajectory, request: web.Request, chat: CallRequest, dialect: Dialect
    ) -> web.StreamResponse:
        started = time.time()
        if self.use_raw_content and not dialect.raw_content:
            return dialect.error(chat, "this server answers with raw content, which this API can't carry", 400)
        refused = dialect.unsupported_in_token_mode(chat)
        if refused is not None:
            return dialect.error(chat, refused, 400)
        # The turns as the template renders them: matched, rendered and committed in this form.
        messages = fold_developer(chat.messages, self.chat_template)
        model = self.model or chat.model
        graph = trajectory.graph
        tools_key = hashing.tools_hash(chat.tools)
        if tools_key and chat.tools is not None:
            graph.tools.setdefault(tools_key, [dict(tool) for tool in chat.tools])
        key = hashing.MatchKey.tokens(tools_key, model)
        try:
            planned = await asyncio.to_thread(turn.plan, graph, self.renderer, messages, chat.tools, key)
            # A first call has nothing to extend; after that, not extending is worth reporting.
            # A call commits its messages and its reply together (``turn.commit``), so the graph
            # has nodes exactly when an earlier call went through.
            follows_a_call = bool(graph.nodes)
        except turn.TokenError as error:
            return dialect.error(chat, str(error), 400)

        # `max_completion_tokens` is an alias of `max_tokens`; resolving it on each side before
        # merging is what lets an override win over a caller's alias.
        sampling = {**_resolve_max_tokens(chat.sampling), **_resolve_max_tokens(self.sampling_overrides)}
        max_tokens = sampling.get("max_tokens")
        if self.max_model_len is not None:
            room = self.max_model_len - len(planned.prompt_ids)
            if room <= 0:
                return dialect.error(
                    chat,
                    f"prompt of {len(planned.prompt_ids)} tokens leaves no room within max_model_len="
                    f"{self.max_model_len}",
                    400,
                    code="context_length_exceeded",
                )
            max_tokens = min(max_tokens, room) if max_tokens else room
        if max_tokens:
            sampling["max_tokens"] = max_tokens
        sampling.setdefault("stop_token_ids", self.renderer.stop_token_ids())

        routes_from = turn.routes_from(graph, planned)
        features = None
        if planned.media and not self._media_checked:
            await self._check_media()
        if planned.media and self.media_error:
            return _error(f"{self.renderer.name}'s images can't be sent to the engine: {self.media_error}", 400)
        if planned.media:
            try:
                features = await asyncio.to_thread(self.renderer.features, planned.media)
            except Exception as error:  # noqa: BLE001 - the renderer's encoder, whatever it raises
                logger.exception("trajectory %s: encoding the prompt's media failed", trajectory.id)
                return _error(f"encoding the prompt's media: {error}", 500, kind="api_error")
        body = self.engine.request(
            prompt_ids=planned.prompt_ids,
            sampling=sampling,
            model=model,
            cache_salt=chat.body.get("cache_salt"),
            sampling_mask=self.sampling_mask,
            routes_from=routes_from,
            features=features,
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
                    return dialect.error(
                        chat,
                        f"engine returned HTTP {up.status}",
                        502 if up.status < 500 else up.status,
                        kind="api_error",
                    )
                output = self.engine.parse(orjson.loads(raw), routes_from=routes_from)
        except (aiohttp.ClientError, EngineError, orjson.JSONDecodeError) as error:
            self._fail(trajectory, None, f"engine: {error}")
            return dialect.error(chat, f"engine: {error}", 502, kind="api_error")

        if self.use_raw_content:
            reply = await asyncio.to_thread(self._raw_reply, output.completion_ids, planned.prompt_ids)
        else:
            reply = await asyncio.to_thread(
                self.renderer.parse, output.completion_ids, chat.tools, prompt_ids=planned.prompt_ids
            )
        reason = finish_reason(output.finish_reason, reply)
        sampled = Turn(
            message=reply,
            finish_reason=reason,
            model=chat.model or model,
            prompt_tokens=len(planned.prompt_ids),
            completion_tokens=len(output.completion_ids),
            key=hashing.digest([len(planned.prompt_ids), output.completion_ids]),
            raw_content=self.use_raw_content,
        )
        # The reply as the client's replay of it will be read back, which is what the next call
        # matches: stored as the node's message and hashed for its match hash.
        replay = dialect.replayed(sampled)
        call = CallInfo(
            t_start=started,
            t_end=time.time(),
            model=model,
            sampling={k: v for k, v in sampling.items() if k != "stop_token_ids"},
            usage={"prompt_tokens": len(planned.prompt_ids), "completion_tokens": len(output.completion_ids)},
            finish_reason=reason,
            tools=tools_key or None,
            bridged=planned.bridged if follows_a_call else None,
            dialect=chat.dialect.value,
        )
        status = "ok"
        recorded = False
        if trajectory.is_open:
            try:
                turn.commit(
                    graph,
                    planned,
                    messages=messages,
                    reply=replay,
                    reply_match=key(replay),
                    output=output,
                    call=call,
                )
                recorded = True
            except turn.TokenError as error:
                logger.warning("trajectory %s failed: %s", trajectory.id, error)
                self._fail(trajectory, None, f"token attribution: {error}")
                trajectory.seal("failed", cancel=False)
                status = "failed"
        body_out = dialect.reply(chat, sampled)
        headers = {STATUS_HEADER: status}
        if not chat.stream:
            out = web.Response(body=orjson.dumps(body_out), content_type="application/json", headers=headers)
        else:
            # The whole completion exists already, so the stream is one buffered
            # body: identical to the client, and replayable for a retry.
            out = web.Response(
                body=b"".join(dialect.stream(body_out)),
                content_type="text/event-stream",
                headers={"Cache-Control": "no-cache", **headers},
            )
        return retry.committed(out) if recorded else out

    def _raw_reply(self, completion_ids: list[int], prompt_ids: list[int]) -> dict[str, Any]:
        """The completion as text, without its stop token: what vLLM returns with no parsers.

        The harness replays this ``content`` as the assistant message of its next request, so it
        has to be the text that renders back to exactly the sampled tokens. The chat template
        closes every assistant turn with its own end-of-turn token, so the text must not carry
        one. With Qwen3, a completion sampled as ``done<|im_end|>``, ids ``[10438, 151645]``:

        * returned as ``"done"``, the replayed turn renders as ``done <|im_end|>``,
          ``[10438, 151645]``, the sampled tokens;
        * returned as ``"done<|im_end|>"``, the literal ``<|im_end|>`` encodes to 151645 too, and
          the template adds its own: ``[10438, 151645, 151645]``, an end token the model never
          sampled. Wherever skycap renders that history rather than reusing the sampled tokens, the
          turn no longer matches what was sampled and the trajectory forks. The harness also sees
          a stray ``<|im_end|>`` that vLLM would never have returned.

        So trailing stop tokens are dropped before decoding. Only trailing ones: generation ends at
        the first stop token, so none can sit mid-completion. The recorded tokens keep theirs.
        """
        stops = set(self.renderer.stop_token_ids())
        end = len(completion_ids)
        while end and completion_ids[end - 1] in stops:
            end -= 1
        text, _ = self.renderer.decode_spans(completion_ids[:end], prompt_ids[-self.DECODE_CONTEXT :])
        return {"role": "assistant", "content": text}

    @staticmethod
    def _fail(trajectory: Trajectory, status: int | None, error: str) -> None:
        if trajectory.is_open:
            trajectory.failures.append(Failure(t=time.time(), status=status, error=error))


def finish_reason(engine_reason: str, message: dict[str, Any]) -> str:
    """The engine never says ``tool_calls``; a parsed tool call makes a stop one."""
    if engine_reason == "stop" and message.get("tool_calls"):
        return "tool_calls"
    return engine_reason


def _resolve_max_tokens(sampling: Mapping[str, Any]) -> dict[str, Any]:
    """``sampling`` with ``max_completion_tokens`` folded into ``max_tokens``, which it takes precedence over."""
    resolved = dict(sampling)
    alias = resolved.pop("max_completion_tokens", None)
    if alias is not None:
        resolved["max_tokens"] = alias
    return resolved
