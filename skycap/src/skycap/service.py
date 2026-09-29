"""A capture server in this process, on its own thread and event loop.

``CaptureService`` is for embedding: a trainer starts one next to its engine,
from the same options ``skycap serve`` takes, and hands its URL to a
``CapturePool``. How a call reaches the model (text or tokens mode) is built
inside from those options; the caller never constructs it.

The server gets a thread of its own because the embedding process may run each
batch on a new event loop. ``stop`` shuts it down gracefully, which writes every
trajectory still in memory to the record directory.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aiohttp import web

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
    """``upstream_url`` and ``options`` are ``build_backend``'s; the rest place and persist the server.

    ``port=0`` lets the OS pick a free port. ``advertise_host`` is the address clients use to reach
    this server, which ``url`` carries once started.
    """

    def __init__(
        self,
        upstream_url: str,
        *,
        record_dir: str | None = None,
        ttl: float = 3600.0,
        host: str = "0.0.0.0",
        port: int = 0,
        advertise_host: str = "127.0.0.1",
        **options: Any,
    ) -> None:
        self.server = CaptureServer(build_backend(upstream_url, **options), record_dir=record_dir, ttl=ttl)
        self._host, self._port = host, port
        self._advertise_host = advertise_host
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopping: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        #: Where clients reach the server, set by ``start``.
        self.url: str | None = None

    def start(self, timeout: float = 60.0) -> str:
        """Start serving and return the server's URL. Returns once it accepts connections."""
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
        runner = web.AppRunner(self.server.app())
        await runner.setup()
        try:
            site = web.TCPSite(runner, self._host, self._port)
            await site.start()
            port = runner.addresses[0][1]
            host = f"[{self._advertise_host}]" if ":" in self._advertise_host else self._advertise_host
            self.url = f"http://{host}:{port}"
            self._ready.set()
            await self._stopping.wait()
        finally:
            await runner.cleanup()
