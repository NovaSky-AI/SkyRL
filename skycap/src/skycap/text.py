"""Text mode: forward the call to an OpenAI-compatible server and record what came back.

The request bytes are forwarded unchanged apart from the credential. The reply
is committed to the graph before the response (or, when streamed, its end)
reaches the harness, so a harness that calls ``finish`` right after its last
reply always finds that reply recorded.
"""

from __future__ import annotations

import time

import aiohttp
import orjson
from aiohttp import web

from skycap.graph import CallInfo
from skycap.openai_chat import (
    ChatReply,
    ChatRequest,
    StreamAssembler,
    error_body,
    parse_response,
)
from skycap.trajectory import Failure, Trajectory

_HOP_HEADERS = {"authorization", "host", "content-length", "transfer-encoding", "connection"}


class TextBackend:
    def __init__(self, upstream_url: str, *, api_key: str | None = None) -> None:
        self.upstream_url = upstream_url.rstrip("/")
        self.api_key = api_key
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30),
            connector=aiohttp.TCPConnector(limit=0),
        )

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()

    async def release(self, trajectory: Trajectory) -> None:
        """Nothing is held upstream per trajectory in text mode."""

    @property
    def session(self) -> aiohttp.ClientSession:
        assert self._session is not None, "backend not started"
        return self._session

    def _headers(self, request: web.Request) -> dict[str, str]:
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def models(self, request: web.Request) -> web.Response:
        async with self.session.get(f"{self.upstream_url}/models", headers=self._headers(request)) as up:
            return web.Response(body=await up.read(), status=up.status, content_type=up.content_type)

    async def chat(
        self, trajectory: Trajectory, request: web.Request, chat: ChatRequest, raw: bytes
    ) -> web.StreamResponse:
        started = time.time()
        try:
            async with self.session.post(
                f"{self.upstream_url}/chat/completions", data=raw, headers=self._headers(request)
            ) as up:
                if chat.stream and up.status == 200 and up.content_type == "text/event-stream":
                    return await self._relay(trajectory, request, chat, up, started)
                body = await up.read()
                response = web.Response(body=body, status=up.status, content_type=up.content_type)
                if up.status != 200:
                    _fail(trajectory, up.status, body.decode(errors="replace")[:2000])
                    return response
                try:
                    reply = parse_response(orjson.loads(body))
                except ValueError as error:
                    _fail(trajectory, up.status, f"unreadable reply: {error}")
                    return response
                commit(trajectory, chat, reply, started)
                return response
        except aiohttp.ClientError as error:
            _fail(trajectory, None, f"upstream: {error}")
            return web.Response(
                body=error_body(f"upstream unavailable: {error}", kind="api_error"),
                status=502,
                content_type="application/json",
            )

    async def _relay(
        self,
        trajectory: Trajectory,
        request: web.Request,
        chat: ChatRequest,
        up: aiohttp.ClientResponse,
        started: float,
    ) -> web.StreamResponse:
        response = web.StreamResponse(
            status=up.status, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        await response.prepare(request)
        assembler = StreamAssembler()
        # From the event carrying the finish_reason on, chunks are held back until the reply
        # is committed: a client that stops at the end of the stream and finishes at once
        # must find its last reply recorded.
        held: list[bytes] = []
        try:
            async for chunk in up.content.iter_any():
                assembler.feed(chunk)
                if held or assembler.finish_reason is not None:
                    held.append(chunk)
                    if assembler.saw_done:
                        break
                    continue
                await response.write(chunk)
        except aiohttp.ClientError as error:
            _fail(trajectory, up.status, f"upstream stream interrupted: {error}")
            # The response has started, so it can't become an error response. Dropping the
            # connection makes the client see a failed stream rather than a short one.
            raise ConnectionResetError("upstream stream interrupted") from error
        assembler.feed(b"\n\n")
        try:
            commit(trajectory, chat, assembler.reply(), started)
        except ValueError as error:
            _fail(trajectory, up.status, f"unreadable stream: {error}")
        for chunk in held:
            await response.write(chunk)
        await response.write_eof()
        return response


def commit(trajectory: Trajectory, chat: ChatRequest, reply: ChatReply, started: float) -> None:
    """Record one successful call, unless the trajectory was sealed meanwhile."""
    if not trajectory.is_open:
        return
    call = CallInfo(
        t_start=started,
        t_end=time.time(),
        model=chat.model,
        sampling=chat.sampling,
        usage=reply.usage,
        finish_reason=reply.finish_reason,
    )
    trajectory.graph.commit_text(chat.messages, reply.message, tools=chat.tools, model=chat.model, call=call)


def _fail(trajectory: Trajectory, status: int | None, error: str) -> None:
    if trajectory.is_open:
        trajectory.failures.append(Failure(t=time.time(), status=status, error=error))
