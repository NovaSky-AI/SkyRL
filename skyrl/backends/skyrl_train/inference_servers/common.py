"""
Common utilities for inference servers.

Uses Ray's public network utilities for consistency with Ray's cluster management.
"""

import logging
import socket
from dataclasses import dataclass
from typing import Tuple

import ray

logger = logging.getLogger(__name__)

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
    slot: int = 0
    """Deployment ordinal assigned at provision -- the server's STABLE identity.

    ``ip``/``port`` are not identity: a restarted engine re-reserves its port
    inside its own ``SERVER_PORT_STRIDE`` window and may land on a different one,
    so its URL changes while the thing it *is* does not. Everything that must
    survive a restart keys on the slot instead: RDT stamps it as ``replica_rank``
    (which fixes the consumer-id block the engine owns), and the supervisor
    relaunches a dead deployment with its old slot.

    Data-parallel peers of one deployment SHARE a slot -- they share a parallel
    config in which vLLM's own ``data_parallel_index`` already separates them, so
    one ``replica_rank`` per deployment is exactly right (see
    ``rdt_control_protocol.build_rdt_init_payloads``).

    The default of 0 is only correct for a single deployment; slots are stamped
    driver-side by ``ServerActorPool``, which is the layer that knows a group's
    ordinal within the fleet. Actors never learn their own slot.
    """

    @property
    def url(self) -> str:
        return f"http://{self.ip}:{self.port}"


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


def find_and_reserve_port(start_port: int) -> Tuple[int, socket.socket]:
    """Find an available port and hold the socket to prevent race conditions.

    This keeps the socket bound so no other process can claim the same port
    between discovery and actual server startup.

    Returns:
        (port, socket) -- caller must close the socket before rebinding.
    """
    port = start_port
    end_port = start_port + SERVER_PORT_STRIDE
    sock: socket.socket | None = None
    while port < end_port:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            sock.listen(1)
            return port, sock
        except OSError:
            if sock:
                sock.close()
            port += 1
    raise RuntimeError(
        f"No available port found in [{start_port}, {end_port}). "
        f"Free up the port range or raise SERVER_PORT_STRIDE."
    )
