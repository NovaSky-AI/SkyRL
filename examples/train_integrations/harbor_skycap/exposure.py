"""How agents that call the model from inside a remote sandbox reach skycap.

Terminus-2 calls the model from this machine, so it reaches skycap at the
server's own URL and nothing is exposed. An agent that runs inside its sandbox
(Daytona, Modal, ...) calls the model from there, so each skycap server's
harness routes have to be reachable from the sandbox network. Per server:

* The skycap server listens a second time, with the harness routes alone
  (``/t/{trajectory id}/v1/chat/completions`` and ``/models``;
  ``CaptureService(harness_host=...)``). The control plane (create, finish,
  read) is not routed there, so it stays private. The random trajectory id in
  the path is what a caller must know.
* An ``Exposure`` says where that listener binds, makes it reachable, and
  returns the URL agents use instead of the server's. Built in, by
  ``skycap.exposure.type``: ``external_host`` (an address the sandboxes route
  to: the node's own, or a relay's such as frp on a public VM) and
  ``cloudflare`` (a Cloudflare quick tunnel, see ``tunnel.py``). Any other way
  in is an ``Exposure`` subclass named by import path, ``module:Class``.
"""

import importlib
import inspect
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Type
from urllib.parse import urlsplit

from skyrl.backends.skyrl_train.inference_servers.common import (
    default_bind_host,
    format_http_url,
)


class Exposure:
    """Makes one skycap server's harness routes reachable from sandboxes. Subclass it for a new way in.

    One instance per server, built in the server's Ray actor as ``cls(**skycap.exposure.kwargs)``, so the
    class must be importable there and its constructor should only store its arguments. The actor starts
    the server with its harness listener bound at ``bind(index)``, then calls ``start`` with that
    listener's local URL, and ``stop`` before the server stops.
    """

    def bind(self, index: int) -> Tuple[str, int]:
        """Host and port of server ``index``'s harness listener. Default: a free loopback port, for a way
        in that dials out from this node (a tunnel)."""
        return "127.0.0.1", 0

    def start(self, harness_url: str, index: int) -> str:
        """Make ``harness_url`` reachable from sandboxes. Returns the URL they reach it at, which stands
        in for the server's URL: agents get ``{url}/t/{trajectory id}/v1``."""
        raise NotImplementedError

    def stop(self) -> None:
        """Release what ``start`` opened. Also called when opening failed, so ``start`` may not have run."""

    def open(self, harness_url: str, index: int) -> str:
        """Expose the server's harness listener at ``harness_url``. Returns the exposed URL."""
        self._opened = True
        try:
            return self.start(harness_url, index).rstrip("/")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close the way in. Does nothing unless opened, and only once."""
        if not getattr(self, "_opened", False):
            return
        self._opened = False
        self.stop()


class ExternalHost(Exposure):
    """Sandboxes reach this node at ``host``, server ``i`` on ``port + i``; the harness listeners bind all
    interfaces.

    ``host`` is an address the sandboxes route to: the node's public or peered address, or a relay's
    that forwards each port here, such as an frp server on a public VM with one TCP forward per server.
    """

    def __init__(self, host: str, port: int = 11500) -> None:
        self.host = host
        self.port = port

    def bind(self, index: int) -> Tuple[str, int]:
        return default_bind_host(self.host), self.port + index

    def start(self, harness_url: str, index: int) -> str:
        return format_http_url(self.host, urlsplit(harness_url).port)


class CloudflareQuickTunnel(Exposure):
    """A Cloudflare quick tunnel per server: a random ``https://*.trycloudflare.com`` URL, no account needed.

    For development: a quick tunnel takes at most 200 requests in flight, cuts a response that hasn't
    started within about 125 s, and has no SLA.
    """

    def __init__(self, timeout: float = 120.0, attempts: int = 3) -> None:
        self.timeout = timeout
        self.attempts = attempts
        self._tunnel: Any = None

    def start(self, harness_url: str, index: int) -> str:
        from .tunnel import CloudflareTunnel

        self._tunnel = CloudflareTunnel(harness_url)
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
