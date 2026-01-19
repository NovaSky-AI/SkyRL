"""
Common utilities for inference servers.
"""

import socket
from dataclasses import dataclass

import ray


@dataclass
class ServerInfo:
    """Information about a running inference server."""

    ip: str
    port: int

    @property
    def url(self) -> str:
        return f"http://{self.ip}:{self.port}"


def get_node_ip() -> str:
    """Get the IP address of the current node."""
    return ray._private.services.get_node_ip_address().strip("[]")


def get_free_port(start_port: int = 8000) -> int:
    """Find an available port starting from start_port."""
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("", port))
                s.listen(1)
                return port
            except OSError:
                port += 1
