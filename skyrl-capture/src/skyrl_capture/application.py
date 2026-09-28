"""The ASGI application: routing, and one ``Runtime`` lifespan.

Three applications share a port and no code path:

* anything under the data-plane prefix is a trajectory capture route and goes
  to the raw ASGI data plane, with no framework in the way;
* the reads and the bulk exports go to the viewer app, when this replica
  serves one;
* everything else -- create, finish, metadata, health -- goes to the lifecycle
  app.

The split between the last two is by method and path rather than by mounting,
because they both live at the root of `/v1` and only one of them is optional.
When the viewer is disabled its routes are not registered anywhere, so a read
gets a `404` or a `405` from the lifecycle app, which is exactly what
`--disable-viewer` should mean.

The ``Runtime`` is built here, during ASGI lifespan startup, and nowhere else.
That is what puts it inside the event loop that will use it -- it starts tasks,
and a task belongs to the loop that created it.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from skyrl_capture.config import Config, StartupError
from skyrl_capture.routes import DATA_PLANE_PREFIX
from skyrl_capture.runtime import Runtime, build_runtime

logger = logging.getLogger(__name__)

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]


class CaptureApplication:
    """Route capture and control requests and own one Runtime lifespan.

    This object is an ASGI application, not a server. It does not configure
    Uvicorn, create threads, or decide whether serving is blocking.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        # Kept so `CaptureService` can report why a start failed, rather than
        # leaving a caller with a timeout and nothing to read.
        self.startup_error: BaseException | None = None
        self._runtime: Runtime | None = None
        self._lifecycle: Any = None
        self._viewer: Any = None
        self._data_plane: Any = None
        self._started = False

    @property
    def runtime(self) -> Runtime:
        """The running runtime. Only after lifespan startup has completed."""
        if self._runtime is None:
            raise RuntimeError("the capture application has not started")
        return self._runtime

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if self._lifecycle is None:  # pragma: no cover - startup failed
            raise RuntimeError("the capture application has not started")
        # Anything under this prefix is a trajectory route; everything else at
        # the root belongs to the control API. `skyrl_capture.routes` says why
        # it is keyed this way round.
        if scope["type"] == "http" and scope["path"].startswith(DATA_PLANE_PREFIX):
            await self._data_plane(scope, receive, send)
            return
        if self._viewer is not None and reads_captured_data(
            scope.get("method", "GET"), scope.get("path", "")
        ):
            await self._viewer(scope, receive, send)
            return
        await self._lifecycle(scope, receive, send)

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                try:
                    runtime = await build_runtime(self.config)
                    self._wire(runtime)
                    await runtime.start()
                    self._started = True
                except StartupError as error:
                    # The message is the whole report; a traceback would bury it.
                    self.startup_error = error
                    logger.error("%s", error)
                    await send({"type": "lifespan.startup.failed", "message": str(error)})
                    return
                except Exception as error:
                    self.startup_error = error
                    logger.exception("startup failed")
                    await send({"type": "lifespan.startup.failed", "message": str(error)})
                    return
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if self._started and self._runtime is not None:
                    await self._runtime.stop()
                    self._started = False
                await send({"type": "lifespan.shutdown.complete"})
                return

    def _wire(self, runtime: Runtime) -> None:
        self._runtime = runtime
        self._lifecycle = runtime.lifecycle
        self._viewer = runtime.viewer
        self._data_plane = runtime.data_plane


def reads_captured_data(method: str, path: str) -> bool:
    """Whether the viewer app owns this request.

    Everything under `/v1/exports` -- a bulk export is a viewer concern whether
    it is being created, polled or downloaded -- plus the reads of trajectories
    and runs. `POST /v1/trajectories` and `PATCH .../metadata` are lifecycle,
    which is why the method is part of the test.
    """
    if path.startswith("/v1/exports"):
        return True
    if method not in ("GET", "HEAD"):
        return False
    return path == "/v1/runs" or path.startswith("/v1/trajectories")
