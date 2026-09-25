"""A skycap server in this process, on its own thread and event loop.

The trainer may run each ``generate`` on a new event loop, so the server can't
live on the caller's: it gets a thread of its own for the run. ``stop`` shuts
it down gracefully, which writes every trajectory still in memory to the record
directory and releases the engine sessions of those that never ended.
"""

import asyncio
import threading
from typing import Optional

from aiohttp import web
from loguru import logger
from skycap.server import Backend, CaptureServer

from skyrl.backends.skyrl_train.inference_servers.common import format_http_url


class SkycapService:
    def __init__(
        self,
        backend: Backend,
        *,
        record_dir: Optional[str],
        ttl: float = 3600.0,
        host: str = "0.0.0.0",
        port: int = 0,
        advertise_host: str = "127.0.0.1",
    ) -> None:
        self.server = CaptureServer(backend, record_dir=record_dir, ttl=ttl)
        self._host, self._port = host, port
        self._advertise_host = advertise_host
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stopping: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        #: Where clients reach the server, set by ``start``.
        self.url: Optional[str] = None

    def start(self, timeout: float = 60.0) -> str:
        """Start serving and return the server's URL. Returns once it accepts connections."""
        self._thread = threading.Thread(target=self._run, name="skycap", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError(f"skycap did not start within {timeout}s")
        if self._error is not None:
            raise RuntimeError("skycap failed to start") from self._error
        logger.info(f"skycap serving at {self.url}")
        return self.url

    def stop(self, timeout: float = 120.0) -> bool:
        """Stop serving, writing what is still in memory. Returns whether that finished in time."""
        thread = self._thread
        if thread is None:
            return True
        if self._loop is not None and self._stopping is not None and not self._stopping.is_set():
            self._loop.call_soon_threadsafe(self._stopping.set)
        thread.join(timeout)
        if thread.is_alive():
            logger.warning(f"skycap did not stop within {timeout}s")
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
            self.url = format_http_url(self._advertise_host, port)
            self._ready.set()
            await self._stopping.wait()
        finally:
            await runner.cleanup()
