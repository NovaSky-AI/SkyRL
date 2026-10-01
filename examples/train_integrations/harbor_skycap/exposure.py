"""How agents that call the model from inside a remote sandbox reach skycap.

Terminus-2 calls the model from this machine, so it reaches skycap at the
server's own URL and nothing is exposed. An agent that runs inside its sandbox
(Daytona, Modal, ...) calls the model from there, so each skycap server's
harness routes have to be reachable from the sandbox network. Per server:

* ``HarnessGateway`` forwards only the harness routes,
  ``/t/{trajectory id}/v1/chat/completions`` and ``/models``, to the server.
  The control plane (create, finish, read) stays private. The random
  trajectory id in the path is what a caller must know.
* An ``Exposure`` makes the gateway reachable and returns the URL agents use
  instead of the server's. Built in, by ``skycap.exposure.type``:
  ``external_host`` (an address the sandboxes route to: the node's own, or a
  relay's such as frp on a public VM) and ``cloudflare`` (a Cloudflare quick
  tunnel, see ``tunnel.py``). Any other way in is an ``Exposure`` subclass
  named by import path, ``module:Class``.
"""

import asyncio
import importlib
import inspect
import re
import threading
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Type
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

from skyrl.backends.skyrl_train.inference_servers.common import (
    default_bind_host,
    format_http_url,
)

#: The routes an agent in a sandbox may reach. Everything else is the control plane.
HARNESS_ROUTE = re.compile(r"^/t/[^/]+/v1/(chat/completions|models)$")
_HOP_HEADERS = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "content-encoding",
}
#: Where a gateway bound on a wildcard address is reached from this node.
_LOOPBACK = {"0.0.0.0": "127.0.0.1", "::": "::1"}


class HarnessGateway:
    """Forwards the harness routes to one skycap server, on its own thread and event loop."""

    def __init__(self, upstream_url: str) -> None:
        self.upstream_url = upstream_url.rstrip("/")
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._runner: Optional[web.AppRunner] = None
        self._session: Optional[aiohttp.ClientSession] = None

    def start(self, host: str = "127.0.0.1", port: int = 0, timeout: float = 30.0) -> int:
        """Start serving on ``host:port`` (``port=0`` picks a free one). Returns the bound port."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="skycap-gateway", daemon=True)
        self._thread.start()
        return asyncio.run_coroutine_threadsafe(self._serve(host, port), self._loop).result(timeout)

    def stop(self, timeout: float = 30.0) -> None:
        if self._loop is None or self._thread is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(timeout)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout)
            self._loop.close()
            self._loop = self._thread = None

    async def _serve(self, host: str, port: int) -> int:
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=30))
        app = web.Application(client_max_size=1024**3)
        app.router.add_route("*", "/{tail:.*}", self._forward)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()
        return self._runner.addresses[0][1]

    async def _shutdown(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        if self._session is not None:
            await self._session.close()

    async def _forward(self, request: web.Request) -> web.StreamResponse:
        if not HARNESS_ROUTE.match(request.path):
            raise web.HTTPNotFound()
        assert self._session is not None
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
        async with self._session.request(
            request.method, f"{self.upstream_url}{request.path_qs}", data=await request.read(), headers=headers
        ) as upstream:
            response = web.StreamResponse(
                status=upstream.status,
                headers={k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_HEADERS},
            )
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response


class Exposure:
    """Makes one skycap server's harness routes reachable from sandboxes. Subclass it for a new way in.

    One instance per server, built in the server's Ray actor as ``cls(**skycap.exposure.kwargs)``, so the
    class must be importable there and its constructor should only store its arguments. The actor serves
    the harness routes on a gateway bound at ``bind(index)``, then calls ``start`` with the gateway's
    local URL, and ``stop`` before the server stops.
    """

    def bind(self, index: int) -> Tuple[str, int]:
        """Host and port of server ``index``'s gateway. Default: a free loopback port, for a way in
        that dials out from this node (a tunnel)."""
        return "127.0.0.1", 0

    def start(self, gateway_url: str, index: int) -> str:
        """Make ``gateway_url`` reachable from sandboxes. Returns the URL they reach it at, which stands
        in for the server's URL: agents get ``{url}/t/{trajectory id}/v1``."""
        raise NotImplementedError

    def stop(self) -> None:
        """Release what ``start`` opened. Also called when opening failed, so ``start`` may not have run."""

    def open(self, server_url: str, index: int) -> str:
        """Start a gateway to ``server_url`` and expose it. Returns the exposed URL."""
        host, port = self.bind(index)
        self._gateway = HarnessGateway(server_url)
        try:
            bound = self._gateway.start(host=host, port=port)
            return self.start(format_http_url(_LOOPBACK.get(host, host), bound), index).rstrip("/")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close the way in, then the gateway behind it. Does nothing unless opened, and only once."""
        gateway: Optional[HarnessGateway] = getattr(self, "_gateway", None)
        if gateway is None:
            return
        self._gateway = None
        try:
            self.stop()
        finally:
            gateway.stop()


class ExternalHost(Exposure):
    """Sandboxes reach this node at ``host``, server ``i`` on ``port + i``; the gateways bind all interfaces.

    ``host`` is an address the sandboxes route to: the node's public or peered address (like Miles'
    ``--session-server-external-host``), or a relay's that forwards each port here, such as an frp
    server on a public VM with one TCP forward per server.
    """

    def __init__(self, host: str, port: int = 11500) -> None:
        self.host = host
        self.port = port

    def bind(self, index: int) -> Tuple[str, int]:
        return default_bind_host(self.host), self.port + index

    def start(self, gateway_url: str, index: int) -> str:
        return format_http_url(self.host, urlsplit(gateway_url).port)


class CloudflareQuickTunnel(Exposure):
    """A Cloudflare quick tunnel per server: a random ``https://*.trycloudflare.com`` URL, no account needed.

    For development: a quick tunnel takes at most 200 requests in flight, cuts a response that hasn't
    started within about 125 s, and has no SLA.
    """

    def __init__(self, timeout: float = 120.0, attempts: int = 3) -> None:
        self.timeout = timeout
        self.attempts = attempts
        self._tunnel: Any = None

    def start(self, gateway_url: str, index: int) -> str:
        from .tunnel import CloudflareTunnel

        self._tunnel = CloudflareTunnel(gateway_url)
        return self._tunnel.start(timeout=self.timeout, attempts=self.attempts)

    def stop(self) -> None:
        if self._tunnel is not None:
            self._tunnel.stop()
            self._tunnel = None


#: ``skycap.exposure.type``'s built-in values; ``none`` exposes nothing.
BUILT_IN: Dict[str, Optional[Type[Exposure]]] = {
    "none": None,
    "external_host": ExternalHost,
    "cloudflare": CloudflareQuickTunnel,
}


def exposure_factory(
    kind: str,
    *,
    host: Optional[str] = None,
    port: int = 11500,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Callable[[], Exposure]]:
    """``skycap.exposure`` as a factory of per-server exposures, or None for ``none``.

    Checks everything that can be checked before anything starts: the type, the import path, and the
    constructor arguments against the class's signature. Raises ``ValueError`` otherwise.
    """
    kwargs = dict(kwargs or {})
    if kind != "external_host" and host is not None:
        raise ValueError(f"skycap.exposure.host is for type=external_host, not type={kind!r}")
    if kind == "external_host":
        if not host:
            raise ValueError("skycap.exposure.type=external_host needs skycap.exposure.host")
        if not 0 < port < 65536:
            raise ValueError(f"skycap.exposure.port must be a TCP port, got {port}")
        kwargs.update(host=host, port=port)
    cls = BUILT_IN[kind] if kind in BUILT_IN else _import(kind)
    if cls is None:
        if kwargs:
            raise ValueError(f"skycap.exposure.type=none takes no kwargs, got {kwargs}")
        return None
    try:
        inspect.signature(cls).bind(**kwargs)
    except TypeError as error:
        raise ValueError(
            f"skycap.exposure.kwargs {kwargs} don't match {cls.__name__}{inspect.signature(cls)}: {error}"
        ) from None
    return partial(cls, **kwargs)


def _import(path: str) -> Type[Exposure]:
    module, _, name = path.partition(":")
    if not module or not name:
        raise ValueError(
            f"skycap.exposure.type must be one of {sorted(BUILT_IN)} or an import path 'module:Class', got {path!r}"
        )
    try:
        cls = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"skycap.exposure.type={path!r} can't be imported: {error}") from error
    if not (inspect.isclass(cls) and issubclass(cls, Exposure)):
        raise ValueError(f"skycap.exposure.type={path!r} must name a subclass of {Exposure.__module__}.Exposure")
    return cls
