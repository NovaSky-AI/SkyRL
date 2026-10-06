"""A capture server in this process, on its own thread and event loop.

``CaptureService`` is for embedding: a trainer starts one next to its engine,
from the same options ``skycap serve`` takes, and hands its URL to a
``CapturePool``.

The server gets a thread of its own because the embedding process may run each
batch on a new event loop. ``stop`` shuts it down gracefully, which writes every
trajectory still in memory to the record directory.

With an ``exposure`` (``skycap.exposure``), the server also serves its harness
routes on a listener the exposure makes reachable from outside this network:
``exposed_url`` is where, and each trajectory's ``exposed_base_url`` is its
route there.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aiohttp import web

from skycap.exposure import Exposure
from skycap.paths import PathRule
from skycap.server import Backend, CaptureServer
from skycap.text import TextBackend

if TYPE_CHECKING:
    from skycap.tokens.engine import VLLMEngine
    from skycap.tokens.renderer import TokenRenderer

logger = logging.getLogger(__name__)


def build_backend(
    upstream_url: str,
    *,
    mode: str = "text",
    api_key: str | None = None,
    tokenizer: str | None = None,
    renderer: TokenRenderer | None = None,
    renderer_pool_size: int = 8,
    engine: VLLMEngine | None = None,
    model: str | None = None,
    max_model_len: int | None = None,
    sampling_overrides: Mapping[str, Any] | None = None,
    sampling_mask: bool = False,
    logprobs_mode: str = "processed_logprobs",
    use_raw_content: bool = False,
) -> Backend:
    """What ``skycap serve`` and ``CaptureService`` run, from the options they share.

    Token mode renders with ``tokenizer`` through ``renderers``, or with ``renderer`` when one is given
    (e.g. a test's). ``engine`` is the engine's wire, vLLM's by default.
    """
    if mode == "text":
        return TextBackend(upstream_url, api_key=api_key)
    if mode != "tokens":
        raise ValueError(f"mode must be 'text' or 'tokens', not {mode!r}")
    if (tokenizer is None) == (renderer is None):
        raise ValueError("token mode needs exactly one of tokenizer and renderer")
    from skycap.tokens.backend import TokensBackend

    if renderer is None:
        from skycap.tokens.renderer import RenderersRenderer

        renderer = RenderersRenderer(tokenizer, size=renderer_pool_size)
    return TokensBackend(
        upstream_url,
        renderer,
        engine=engine,
        api_key=api_key,
        model=model,
        max_model_len=max_model_len,
        sampling_overrides=sampling_overrides,
        sampling_mask=sampling_mask,
        logprobs_mode=logprobs_mode,
        use_raw_content=use_raw_content,
    )


class CaptureService:
    """The model options are ``build_backend``'s; the rest place and persist the server.

    ``port=0`` lets the OS pick a free port. ``advertise_host`` is the address clients use to reach
    this server, which ``url`` carries once started. ``path_rules`` are the custom path rules
    ``finish`` may name besides ``all`` and ``final``, each a function or its ``"pkg.module:function"``
    import path (``skycap.paths``). ``exposure`` makes the harness routes reachable from outside this
    network (``skycap.exposure``); opening it can take a while (a tunnel), which ``start`` waits for.
    """

    def __init__(
        self,
        upstream_url: str,
        *,
        mode: str = "text",
        api_key: str | None = None,
        tokenizer: str | None = None,
        renderer: TokenRenderer | None = None,
        renderer_pool_size: int = 8,
        engine: VLLMEngine | None = None,
        model: str | None = None,
        max_model_len: int | None = None,
        sampling_overrides: Mapping[str, Any] | None = None,
        sampling_mask: bool = False,
        logprobs_mode: str = "processed_logprobs",
        use_raw_content: bool = False,
        record_dir: str | None = None,
        ttl: float = 3600.0,
        path_rules: Mapping[str, PathRule | str] | None = None,
        host: str = "0.0.0.0",
        port: int = 0,
        advertise_host: str = "127.0.0.1",
        exposure: Exposure | None = None,
    ) -> None:
        backend = build_backend(
            upstream_url,
            mode=mode,
            api_key=api_key,
            tokenizer=tokenizer,
            renderer=renderer,
            renderer_pool_size=renderer_pool_size,
            engine=engine,
            model=model,
            max_model_len=max_model_len,
            sampling_overrides=sampling_overrides,
            sampling_mask=sampling_mask,
            logprobs_mode=logprobs_mode,
            use_raw_content=use_raw_content,
        )
        self.server = CaptureServer(backend, record_dir=record_dir, ttl=ttl, path_rules=path_rules)
        self._host, self._port = host, port
        self._advertise_host = advertise_host
        self._exposure = exposure
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopping: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        #: Where clients reach the server, set by ``start``.
        self.url: str | None = None
        #: Where this node reaches the harness listener, set by ``start`` when there is an exposure.
        self.harness_url: str | None = None

    @property
    def exposed_url(self) -> str | None:
        """Where the harness routes are reached from outside, set by ``start`` when there is an exposure."""
        return self.server.exposed_url

    def start(self, timeout: float | None = None) -> str:
        """Start serving and return the server's URL. Returns once it accepts connections and any exposure
        is open. ``timeout`` defaults to 60 s, or 600 s with an exposure."""
        if timeout is None:
            timeout = 600.0 if self._exposure is not None else 60.0
        self._thread = threading.Thread(target=self._run, name="skycap", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError(f"skycap did not start within {timeout}s")
        if self._error is not None:
            raise RuntimeError("skycap failed to start") from self._error
        logger.info("skycap serving at %s", self.url)
        return self.url

    def stop(self, timeout: float = 120.0) -> bool:
        """Stop serving, writing what is still in memory. Returns whether that finished in time."""
        thread = self._thread
        if thread is None:
            return True
        if self._loop is not None and self._stopping is not None and not self._stopping.is_set():
            try:
                self._loop.call_soon_threadsafe(self._stopping.set)
            except RuntimeError:
                pass  # the loop already closed: the server stopped on its own
        thread.join(timeout)
        if thread.is_alive():
            logger.warning("skycap did not stop within %ss", timeout)
            return False
        self._thread = None
        return True

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as error:  # noqa: BLE001 - reported by start, or logged
            self._error = error
            self._ready.set()
            logger.exception("skycap stopped with an error")

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stopping = asyncio.Event()

        def ready(url: str, harness_url: str | None) -> None:
            self.url, self.harness_url = url, harness_url
            self._ready.set()

        await serve(
            self.server,
            host=self._host,
            port=self._port,
            advertise_host=self._advertise_host,
            exposure=self._exposure,
            stopping=self._stopping,
            ready=ready,
        )


async def serve(
    server: CaptureServer,
    *,
    host: str,
    port: int,
    advertise_host: str,
    exposure: Exposure | None,
    stopping: asyncio.Event,
    ready: Callable[[str, str | None], None],
) -> None:
    """Serve ``server`` on ``host:port`` until ``stopping`` is set; ``CaptureService`` and ``skycap serve``.

    With ``exposure``, the harness routes are also served alone where it binds them, and it is opened
    on that listener. ``ready`` gets the server's URL and the harness listener's, once both accept
    connections and the exposure is open. On the way out the exposure closes first, then the harness
    listener, then the server, which writes what it holds.
    """
    runner = web.AppRunner(server.app())
    await runner.setup()
    harness: web.AppRunner | None = None
    opened = False
    try:
        await web.TCPSite(runner, host, port).start()
        url = _http_url(advertise_host, runner.addresses[0][1])
        harness_url = None
        if exposure is not None:
            harness = web.AppRunner(server.harness_app())
            await harness.setup()
            bind_host, bind_port = exposure.bind()
            await web.TCPSite(harness, bind_host, bind_port).start()
            harness_url = _http_url(_LOOPBACK.get(bind_host, bind_host), harness.addresses[0][1])
            opened = True
            # Off the loop: a tunnel can take a minute to come up, and the control plane serves meanwhile.
            server.exposed_url = (await asyncio.to_thread(exposure.start, harness_url)).rstrip("/")
            logger.info("skycap harness routes exposed at %s", server.exposed_url)
        ready(url, harness_url)
        await stopping.wait()
    finally:
        if opened:
            assert exposure is not None
            try:
                await asyncio.to_thread(exposure.stop)
            except Exception:
                logger.exception("closing the exposure failed")
            server.exposed_url = None
        if harness is not None:
            await harness.cleanup()
        await runner.cleanup()


#: Where a listener bound on a wildcard address is reached from this node.
_LOOPBACK = {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "::1"}


def _http_url(host: str, port: int) -> str:
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"
