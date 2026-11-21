"""OpenAI-compatible ingress for endpoint-based rollout generators.

The proxy is a transport adapter only. It owns no policy or queue: every
request is converted to ``SamplingRequest`` and submitted to the run-scoped
``SamplingService`` that also serves direct Python callers.
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import TYPE_CHECKING, Any

from aiohttp import web
from loguru import logger

from skyrl.train.sampling_service import SamplingOperation, SamplingRequest

if TYPE_CHECKING:
    from skyrl.train.sampling_service import SamplingService


class SamplingHTTPProxy:
    """Small background HTTP transport into one ``SamplingService``."""

    def __init__(self, service: "SamplingService") -> None:
        self._service = service
        self._loop = asyncio.new_event_loop()
        self._runner: web.AppRunner | None = None
        self._startup_error: BaseException | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="skyrl-sampling-http", daemon=True)

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("0.0.0.0", 0))
        self._socket.listen(256)
        self._socket.setblocking(False)
        port = int(self._socket.getsockname()[1])
        self.endpoint_url = f"http://{_advertised_host()}:{port}"

        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            self._socket.close()
            raise RuntimeError("timed out starting sampling HTTP proxy")
        if self._startup_error is not None:
            raise RuntimeError("failed to start sampling HTTP proxy") from self._startup_error

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start())
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            self._socket.close()
            return
        self._ready.set()
        self._loop.run_forever()
        self._loop.run_until_complete(self._cleanup())
        self._loop.close()

    async def _start(self) -> None:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._chat_completion)
        app.router.add_post("/v1/completions", self._completion)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.SockSite(self._runner, self._socket).start()

    async def _chat_completion(self, request: web.Request) -> web.Response:
        return await self._forward(request, operation="chat_completion")

    async def _completion(self, request: web.Request) -> web.Response:
        return await self._forward(request, operation="completion")

    async def _forward(self, request: web.Request, *, operation: SamplingOperation) -> web.Response:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("OpenAI request body must be a JSON object")
            payload: dict[str, Any] = {"json": body, "headers": dict(request.headers)}
            result = await self._service.submit_from_proxy(
                SamplingRequest(
                    operation=operation,
                    payload=payload,
                    # HTTP clients cannot unwind and reconstruct a Python
                    # trajectory. They participate in admission, while active
                    # trajectory shedding remains opt-in through the direct SDK.
                    attempt_id=None,
                )
            )
            return web.json_response(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Sampling HTTP proxy {} failed: {!r}", operation, exc)
            return web.json_response(
                {"error": {"message": str(exc), "type": type(exc).__name__}},
                status=503,
            )

    async def _cleanup(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def close(self) -> None:
        if not self._thread.is_alive():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("timed out stopping sampling HTTP proxy")


def _advertised_host() -> str:
    """Return an address reachable by other workers in the Ray cluster."""

    try:
        from ray.util import get_node_ip_address

        return get_node_ip_address()
    except Exception:
        return "127.0.0.1"
