"""Run a complete capture service.

Ownership:

    CaptureService
      owns the serving thread, Uvicorn configuration, blocking behavior,
      startup error propagation, and stop signal

    CaptureApplication
      owns ASGI routing and the Runtime lifespan

    Runtime
      owns capture resources and background tasks

Startup flows in one direction:

    CaptureService.start()
      -> uvicorn.Server.run()
      -> CaptureApplication lifespan startup
      -> build_runtime()
      -> Runtime.start()

Shutdown runs the same chain in reverse. CaptureService never starts or stops
Runtime directly; CaptureApplication never creates threads or configures
Uvicorn.

The CLI and embedded Python both use CaptureService.start(). The only
difference is whether ``blocking`` is true:

    service = CaptureService(config=config, host=host, port=port)
    service.start(blocking=True)           # `skyrl-capture serve`

    service = CaptureService(config=config, host=host, port=port)
    service.start(blocking=False)          # a training script
    try:
        run_training(service.base_url)
    finally:
        service.stop()

The upstream is part of that config, so there is no second call to make after
start(): what the process captures is decided before it is running.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from pathlib import Path

import uvicorn

from skyrl_capture.application import CaptureApplication
from skyrl_capture.config import Config, load_config
from skyrl_capture.runtime import Runtime

logger = logging.getLogger(__name__)


def _reachable(host: str) -> str:
    """A server bound to every interface is not an address a client can use."""
    return "127.0.0.1" if host in ("0.0.0.0", "::") else host


class CaptureServiceError(RuntimeError):
    pass


class CaptureService:
    """Own and serve one CaptureApplication.

    This is the public process-level API. It is the only place that knows about
    Uvicorn or background threads. Runtime construction and cleanup happen
    through the application's ASGI lifespan.

    A service is single-use: `start()` refuses a second call, `stop()` is safe
    to repeat, and restarting means building another one.
    """

    def __init__(
        self,
        *,
        config: Config | None = None,
        data_dir: str | Path | None = None,
        port: int | None = None,
        host: str | None = None,
    ) -> None:
        config = config or load_config()
        if data_dir is not None:
            config = replace(config, data_dir=Path(data_dir))
        if host is not None or port is not None:
            resolved_host = host or config.proxy.host
            resolved_port = config.proxy.port if port is None else port
            # `public_url` is what a trajectory's `base_url` is built from, so
            # moving the port without moving it hands every caller a URL
            # pointing at the old one -- which looks perfectly ordinary and
            # answers nothing. Only rewritten when it was still the default for
            # the *previous* address; an explicit PUBLIC_URL is left alone,
            # because behind a load balancer it is deliberately not the bind
            # address.
            default_before = f"http://{_reachable(config.proxy.host)}:{config.proxy.port}"
            public_url = config.proxy.public_url
            if public_url.rstrip("/") == default_before:
                public_url = f"http://{_reachable(resolved_host)}:{resolved_port}"
            config = config.with_overrides(
                proxy=replace(
                    config.proxy,
                    host=resolved_host,
                    port=resolved_port,
                    public_url=public_url,
                )
            )
        self.config = config

        self._application = CaptureApplication(self.config)
        # Uvicorn owns the event loop. CaptureApplication's lifespan builds
        # Runtime inside that loop and shuts it down before the loop exits.
        self._server = uvicorn.Server(
            uvicorn.Config(
                self._application,
                host=self.config.proxy.host,
                port=self.config.proxy.port,
                # "auto" picks uvloop and httptools when the `fast` extra is
                # installed, so there is nothing to detect here.
                loop="auto",
                http="auto",
                log_level="warning",
                access_log=False,
                # Capture payloads can be large; keep the frame limits generous.
                h11_max_incomplete_event_size=self.config.proxy.max_request_bytes,
                timeout_keep_alive=75,
            )
        )
        self._thread: threading.Thread | None = None
        self._thread_error: BaseException | None = None
        self._start_called = False

    # -- addresses ---------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://{_reachable(self.config.proxy.host)}:{self.config.proxy.port}"

    @property
    def runtime(self) -> Runtime:
        """The running runtime, once startup has completed."""
        try:
            return self._application.runtime
        except RuntimeError as error:
            raise CaptureServiceError(str(error)) from error

    # -- lifecycle ---------------------------------------------------------
    def start(self, *, blocking: bool = False, timeout: float = 180.0) -> None:
        """Serve.

        ``blocking=True`` serves on the calling thread and returns when the
        server is asked to stop, which is what ``skyrl-capture serve`` wants:
        Uvicorn installs the signal handlers and owns the process.

        ``blocking=False`` serves on a thread this object owns and returns once
        the socket is bound *and* lifespan startup has finished -- so a caller
        can hand out `base_url` on the next line without racing its own
        harness. ``timeout`` is generous because a tokens deployment loads a
        tokenizer before it can serve.

        Read the address from `base_url`; this returns nothing.
        """
        if self._start_called:
            raise CaptureServiceError(
                "this capture service has already been started; build another one"
            )
        self._start_called = True
        if blocking:
            # The same `_serve` the thread runs, on this thread. A lifespan
            # failure or a bound port stops the server rather than raising out
            # of it, so a blocking caller would otherwise see a clean exit.
            self._serve()
            if self._application.startup_error is not None or self._thread_error is not None:
                raise CaptureServiceError(self._startup_failure())
            return
        self._thread = threading.Thread(target=self._serve, name="capture-service", daemon=True)
        self._thread.start()
        self._wait_until_serving(timeout)

    def stop(self, *, timeout: float = 60.0) -> None:
        """Ask the server to stop, and wait for it. Safe to call twice.

        The runtime is not stopped here. Uvicorn's shutdown runs ASGI lifespan
        shutdown, and that is what owns runtime cleanup -- one teardown path,
        whichever way the server was told to stop.
        """
        self._server.should_exit = True
        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout)
        if thread.is_alive():
            raise CaptureServiceError(f"capture service did not stop within {timeout}s")
        self._thread = None

    def __enter__(self) -> CaptureService:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- internals ---------------------------------------------------------
    def _serve(self) -> None:
        """Serve until told to stop. The one call to `uvicorn.Server.run`.

        ``BaseException``, because Uvicorn reports a bound port or a failed
        lifespan by calling `sys.exit`: on a thread that would be a silent
        death, and on the calling thread it would bypass the error `start`
        wants to raise.
        """
        try:
            self._server.run()
        except BaseException as error:  # surfaced by `start`
            self._thread_error = error

    def _wait_until_serving(self, timeout: float) -> None:
        assert self._thread is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # `started` goes true after lifespan startup *and* the bind, which
            # is exactly what a caller is waiting for. Polling `/healthz` would
            # prove less: it answers 200 whatever capture is doing.
            if self._server.started:
                return
            if (
                self._application.startup_error is not None
                or self._thread_error is not None
                or not self._thread.is_alive()
            ):
                raise CaptureServiceError(self._startup_failure())
            time.sleep(0.01)
        # Do not leave a half-started server running behind a failed `start`.
        self.stop(timeout=timeout)
        raise CaptureServiceError(f"capture service did not start within {timeout}s")

    def _startup_failure(self) -> str:
        error = self._application.startup_error
        if error is not None:
            return f"capture service failed to start: {error}"
        error = self._thread_error
        if isinstance(error, SystemExit):
            # Uvicorn logs the cause -- almost always a bound port -- and exits.
            return (
                f"capture service exited during startup (uvicorn status {error.code}). "
                f"Is {self.base_url} already in use?"
            )
        if error is not None:
            return f"capture service failed to start: {error!r}"
        return "capture service exited during startup"
