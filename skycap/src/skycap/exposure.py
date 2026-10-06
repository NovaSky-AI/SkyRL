"""Making a server's harness routes reachable from outside this network.

A harness that runs where the server's own URL doesn't reach (an agent inside a
remote sandbox) needs another way in. With an ``Exposure``, a server listens a
second time with the harness routes alone (``CaptureServer.harness_app``: the
trajectories' chat and models routes, no control plane), and the exposure makes
that listener reachable and says at what URL. ``create`` then answers with the
trajectory's ``exposed_base_url`` beside its ``base_url``.

Built in, by name:

* ``external_host``: the listener binds ``port`` on every interface, and is
  reached at ``host:port`` -- the node's own routable address, or a relay's
  (frp on a public VM) that forwards that port here. Plain HTTP.
* ``cloudflare``: a Cloudflare quick tunnel (``skycap.tunnel``), a random
  ``https://*.trycloudflare.com`` URL dialed out from this node. No account,
  no inbound port. For development: a quick tunnel takes at most 200 requests
  in flight and cuts a response that hasn't started within about 125 s.

Any other way in is an ``Exposure`` subclass, named by import path
(``"pkg.module:Class"``). The random trajectory id in the path is what a caller
must know.
"""

from __future__ import annotations

import importlib
import inspect
import ipaddress
from typing import Any
from urllib.parse import urlsplit


class Exposure:
    """Makes one server's harness listener reachable. Subclass it for a new way in.

    The server binds its harness listener at ``bind()``, then calls ``start`` with the listener's
    local URL, and ``stop`` once, before it stops listening. ``start`` may block (a tunnel coming
    up); the server runs it off its event loop. If the server is stopped while ``start`` runs,
    ``stop`` is called from another thread meanwhile, and should make ``start`` give up soon; the
    server waits a while for it, then stops anyway. An instance serves one server, once.
    """

    def bind(self) -> tuple[str, int]:
        """Host and port of the harness listener. Default: a free loopback port, for a way in that
        dials out from this node (a tunnel)."""
        return "127.0.0.1", 0

    def start(self, harness_url: str) -> str:
        """Make ``harness_url`` reachable. Returns the URL callers reach it at, which stands in for the
        server's URL: trajectories are at ``{url}/t/{trajectory id}/v1``."""
        raise NotImplementedError

    def stop(self) -> None:
        """Release what ``start`` opened. Also called when ``start`` failed, so it may not have run."""


class ExternalHost(Exposure):
    """Callers reach this node at ``host:port``; the listener binds ``port`` on every interface.

    ``host`` is an address callers route to: the node's own public or peered address, or a relay's
    that forwards the port here, such as an frp server on a public VM.
    """

    def __init__(self, host: str, port: int) -> None:
        if not host:
            raise ValueError("external_host needs a host")
        if not 0 < port < 65536:
            raise ValueError(f"external_host needs a TCP port, got {port}")
        self.host = host
        self.port = port

    def bind(self) -> tuple[str, int]:
        return ("::" if _is_ipv6(self.host) else "0.0.0.0"), self.port

    def start(self, harness_url: str) -> str:
        host = f"[{self.host}]" if _is_ipv6(self.host) else self.host
        return f"http://{host}:{urlsplit(harness_url).port}"


class CloudflareQuickTunnel(Exposure):
    """A Cloudflare quick tunnel to the listener: a random ``https://*.trycloudflare.com`` URL."""

    def __init__(self, timeout: float = 120.0, attempts: int = 3) -> None:
        self.timeout = timeout
        self.attempts = attempts
        self._tunnel: Any = None
        self._stopped = False

    def start(self, harness_url: str) -> str:
        from skycap.tunnel import CloudflareTunnel

        tunnel = CloudflareTunnel(harness_url)
        self._tunnel = tunnel
        if self._stopped:  # stopped before the tunnel existed: its start gives up at once
            tunnel.stop()
        return tunnel.start(timeout=self.timeout, attempts=self.attempts)

    def stop(self) -> None:
        self._stopped = True
        if self._tunnel is not None:
            self._tunnel.stop()


#: The exposures named without an import path.
BUILT_IN: dict[str, type[Exposure]] = {"external_host": ExternalHost, "cloudflare": CloudflareQuickTunnel}


def exposure_class(spec: str) -> type[Exposure]:
    """The class ``spec`` names: a built-in name, or an ``Exposure`` subclass's ``"pkg.module:Class"``."""
    if spec in BUILT_IN:
        return BUILT_IN[spec]
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"an exposure is one of {sorted(BUILT_IN)} or 'pkg.module:Class', not {spec!r}")
    try:
        cls = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"exposure {spec!r} can't be imported: {error}") from error
    if not (inspect.isclass(cls) and issubclass(cls, Exposure)):
        raise ValueError(f"exposure {spec!r} must name a subclass of skycap.exposure.Exposure")
    return cls


def load_exposure(spec: str, **kwargs: Any) -> Exposure:
    """``spec``'s exposure, built with ``kwargs``. Refuses arguments its constructor doesn't take."""
    cls = exposure_class(spec)
    try:
        inspect.signature(cls).bind(**kwargs)
    except TypeError as error:
        raise ValueError(f"exposure {spec!r} takes {inspect.signature(cls)}, not {kwargs}: {error}") from None
    return cls(**kwargs)


def _is_ipv6(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).version == 6
    except ValueError:
        return False
