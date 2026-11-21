"""
Common utilities for inference servers.

Uses Ray's public network utilities for consistency with Ray's cluster management.
"""

import ipaddress
import logging
import os
import socket
from dataclasses import dataclass
from typing import Optional, Tuple

import ray

logger = logging.getLogger(__name__)

# The base port for the vLLM server group. This is the lowest port number SkyRL
# assigns vLLM servers that are launched. Each server actor owns a 100-port window
# (`SERVER_PORT_STRIDE`). Overridable because another service on the host
# (e.g. a k8s LoadBalancer) can claim port 8000 and silently hijack /wake_up,
# which then 404s from the other app.
VLLM_START_PORT = int(os.environ.get("SKYRL_VLLM_START_PORT", 8000))

# Stride between successive server actors' (or groups') start_port values.
# Each actor's `find_and_reserve_port` increments by 1 on conflict, so this
# stride must be larger than the max number of conflicts an actor could see
# inside its window.
SERVER_PORT_STRIDE = 100


@dataclass
class ServerInfo:
    """Information about a running inference server."""

    ip: str
    port: int
    # optional, in case the mooncake connector is being used
    mooncake_bootstrap_server_port: Optional[int] = None

    @property
    def url(self) -> str:
        return format_http_url(self.ip, self.port)


def is_ipv6_address(host: str) -> bool:
    """True if ``host`` is a literal IPv6 address (hostnames and IPv4 return False)."""
    try:
        return isinstance(ipaddress.ip_address(host), ipaddress.IPv6Address)
    except ValueError:
        return False


def format_http_url(host: str, port: int, path: str = "") -> str:
    """Build ``http://host:port/path`` with IPv6 literals bracketed (RFC 3986).

    On IPv6-only clusters ``ray.util.get_node_ip_address()`` returns a bare IPv6
    literal; ``f"http://{ip}:{port}"`` then yields e.g. ``http://2602:fb33::5:8100``
    which URL parsers reject (``Invalid port``).
    """
    host_part = f"[{host}]" if is_ipv6_address(host) else host
    return f"http://{host_part}:{port}{path}"


def default_bind_host(node_ip: str) -> str:
    """Wildcard bind address matching the node's address family (``::`` accepts IPv4 too)."""
    return "::" if is_ipv6_address(node_ip) else "0.0.0.0"


def get_node_ip() -> str:
    """
    Get the IP address of the current node.

    Returns the node IP from Ray's global worker if Ray is initialized
    """
    return ray.util.get_node_ip_address()


def get_open_port(start_port: int | None = None) -> int:
    """
    Get an available port.

    Args:
        start_port: If provided, search for an available port starting from this value.
                   If None, let the OS assign a free port.

    Returns:
        An available port number.
    """
    if start_port is not None:
        # Search for available port starting from start_port
        port = start_port
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1
                if port > 65535:
                    raise RuntimeError(f"No available port found starting from {start_port}")

    # Let OS assign a free port
    # Try IPv4 first
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        pass

    # Try IPv6
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def find_and_reserve_port(start_port: int, *, host: str = "0.0.0.0") -> Tuple[int, socket.socket]:
    """Reserve a port on the server's bind address until the caller closes it.

    IPv6 sockets retain the OS default, matching the native server listeners.
    Closing the reservation before server startup still leaves a rebind race.

    Returns:
        (port, socket) -- caller must close the socket before rebinding.
    """
    port = start_port
    end_port = start_port + SERVER_PORT_STRIDE
    addresses = socket.getaddrinfo(host, start_port, type=socket.SOCK_STREAM)
    sock: socket.socket | None = None
    while port < end_port:
        for family, socktype, proto, _, address in addresses:
            sock = None
            try:
                sock = socket.socket(family, socktype, proto)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((address[0], port, *address[2:]))
                sock.listen(1)
                return port, sock
            except OSError:
                if sock is not None:
                    sock.close()
        port += 1
    raise RuntimeError(
        f"No available port found in [{start_port}, {end_port}). "
        f"Free up the port range or raise SERVER_PORT_STRIDE."
    )
