"""The data plane: find the trajectory, read the body, hand it to the proxy.

This is a hand-written ASGI application rather than a framework route, because
it runs once per inference request. It does three things and nothing else:
resolve the trajectory named in the route, read the body within the size
limit, and hand the request to the one capture proxy this process was started
with. That proxy owns what happens next, and it cannot reach back here.

Resolving is a dictionary lookup for a trajectory this process is already hot
for, which is every request after the first. A cold one is recovered from its
journal once, here, rather than at startup -- a replacement process does not
read a directory of a hundred thousand journals to find the four it is about
to serve.

**There is no credential here.** The trajectory id in the route says which
trajectory a request belongs to; it is correlation, not authorization, and a
deployment that needs authentication puts it in front -- an authenticating
reverse proxy, a service mesh, a network policy. Capture is a sidecar beside
the inference server it captures, reachable from the same places, and issuing
its own bearer token only made that look like a security boundary it never
was. An inbound `Authorization` header is stripped and never recorded; the
credential this process does hold is the upstream's, and it is applied on the
way out.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from skyrl_capture.routes import DATA_PLANE_PREFIX
from skyrl_capture.transport.response import send_response

logger = logging.getLogger(__name__)

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]


class DataPlane:
    """ASGI app serving ``/route/{trajectory_id}/...`` capture routes."""

    def __init__(self, *, registry: Any, proxy: Any, max_request_bytes: int) -> None:
        self._registry = registry
        # One proxy, chosen at startup. Not a dispatcher holding both: a
        # process is text or token capture for its whole life, and the one it
        # is not was never constructed.
        self._proxy = proxy
        self._max_request_bytes = max_request_bytes
        self.requests_served = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # pragma: no cover - no websocket routes
            await self._error(send, 404, "not found")
            return

        # Route shape is /route/{trajectory_id}/{suffix}. Drop the prefix that
        # got us here, leaving /{trajectory_id}/{suffix}, then split once; do
        # not build a regex match or a router table for this.
        path: str = scope["path"][len(DATA_PLANE_PREFIX) - 1 :]
        slash = path.find("/", 1)
        if slash == -1:
            trajectory_id, suffix = path[1:], "/"
        else:
            trajectory_id, suffix = path[1:slash], path[slash:]

        # A dictionary lookup for a hot trajectory; a journal read for a cold
        # one, once.
        active = self._registry.hot(trajectory_id) or await self._registry.resolve(trajectory_id)
        if active is None:
            # Either it finished -- its journal is gone and its record is
            # committed -- or it never existed. 410 says the route existed and
            # has closed, which 404 would not.
            if await self._registry.is_committed(trajectory_id):
                await self._error(
                    send,
                    410,
                    f"trajectory {trajectory_id} has finished; its record is committed "
                    "and it takes no further requests",
                )
            else:
                await self._error(send, 404, f"unknown trajectory {trajectory_id!r}")
            return
        if not active.accepting:
            # Finalizing or poisoned. Either way the route has closed.
            detail = (
                "a turn could not be attributed exactly, so this trajectory's graph "
                "cannot be extended without recording model tokens as client-authored"
                if active.status == "poisoned"
                else f"status={active.status}"
            )
            await self._error(
                send, 410, f"trajectory {trajectory_id} is no longer accepting requests ({detail})"
            )
            return

        request_start_wall = time.time_ns()
        request_start_mono = time.perf_counter_ns()

        body, too_large = await _read_body(receive, self._max_request_bytes)
        if too_large:
            await self._error(send, 413, "request body is too large")
            return

        self.requests_served += 1
        # Counted open for the whole turn, so `finish` can wait for it. Here
        # rather than in the proxy: this is the one place that knows a turn
        # started, and it is the only place that can promise it is closed.
        active.open_turn()
        try:
            await self._proxy.handle(
                active=active,
                scope=scope,
                send=send,
                suffix=suffix,
                body=body,
                request_start_wall=request_start_wall,
                request_start_mono=request_start_mono,
            )
        finally:
            active.close_turn()

    async def _error(self, send: Send, status: int, message: str) -> None:
        """In the shape the selected proxy's clients expect.

        The proxy owns its wire, so an OpenAI client gets an OpenAI error and
        an Anthropic client an Anthropic one -- without this file knowing
        which upstream is configured.
        """
        await send_response(
            send, status, [(b"content-type", b"application/json")], self._proxy.error_body(status, message)
        )

    def stats(self) -> dict[str, Any]:
        return {"requests_served": self.requests_served}


async def _read_body(receive: Receive, limit: int) -> tuple[bytes, bool]:
    """Read the full request body, refusing anything over ``limit``."""
    parts: list[bytes] = []
    total = 0
    while True:
        message = await receive()
        if message["type"] != "http.request":
            break
        chunk = message.get("body") or b""
        total += len(chunk)
        if total > limit:
            return b"", True
        parts.append(chunk)
        if not message.get("more_body"):
            break
    return b"".join(parts), False
