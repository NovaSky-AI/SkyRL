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

# vLLM's `RayExecutorV2` picks the engine's worker `torch.distributed` TCPStore
# port by probing `[VLLM_DP_MASTER_PORT + 100, VLLM_DP_MASTER_PORT + 100 + 32)`
# and taking the first port whose bind succeeds -- see
# `_select_tcpstore_port` in `vllm/v1/executor/ray_executor_v2.py`. Both numbers
# mirror vLLM internals; recheck them on a vLLM version bump.
# TODO: delete along with `compute_dp_master_port` below once vllm#50969 lands --
# see the removal checklist on that function.
DP_TCPSTORE_PROBE_OFFSET = 100
DP_TCPSTORE_WINDOW = 32

# Offset of the TCPStore probe window inside a server actor's own
# `SERVER_PORT_STRIDE`-wide port window. Placed in the upper half so it clears
# the HTTP port, which `find_and_reserve_port` hands out from the bottom.
DP_TCPSTORE_WINDOW_OFFSET = 64
assert DP_TCPSTORE_WINDOW_OFFSET + DP_TCPSTORE_WINDOW <= SERVER_PORT_STRIDE, (
    "The TCPStore probe window must fit inside one server actor's port window, "
    "otherwise adjacent actors probe overlapping ranges."
)


@dataclass
class ServerInfo:
    """Information about a running inference server."""

    ip: str
    port: int

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


# TODO: Once https://github.com/vllm-project/vllm/pull/50969 lands, this whole
# workaround goes away (that PR was still open against vLLM main on 2026-08-12,
# while we pin vllm==0.26.0). It deletes `_select_tcpstore_port` and has rank 0
# create the TCPStore with `port=0`, publishing the kernel-assigned port while
# still holding the bound socket -- so `VLLM_DP_MASTER_PORT` stops influencing
# TCPStore selection and there is no probe/bind race left to work around.
#
# To check on a vLLM version bump: grep the installed vLLM for
# `_select_tcpstore_port`. If it is gone, delete all of the following:
#   - `DP_TCPSTORE_*` constants above, `compute_dp_master_port`,
#     `dp_tcpstore_probe_window`
#   - the `VLLM_DP_MASTER_PORT` assignment in `VLLMServerActor.__init__`
#   - `_seed_dp_master_port` and its call in `_build_and_serve_vllm_server`
#     (vllm_server_actor.py)
#   - `TestDPMasterPort` in tests/.../inference_servers/test_common.py
#
# Do not leave it in place as a no-op: vLLM's `get_open_port()` excludes 10 ports
# around `VLLM_DP_MASTER_PORT` whenever the variable is merely *present* in the
# environment, and the DP-enabled path still uses it for the DP process group.
def compute_dp_master_port(start_port: int) -> int:
    """Return the ``VLLM_DP_MASTER_PORT`` for a server whose port window begins
    at *start_port*.

    This is a *window base*, not a port we bind: vLLM adds
    ``DP_TCPSTORE_PROBE_OFFSET`` to it and probes ``DP_TCPSTORE_WINDOW`` ports
    from there for the engine's TCPStore. Reserving a single port would
    therefore guarantee nothing -- what matters is that co-located engines probe
    *disjoint* windows, so derive the base from the caller's assigned
    ``start_port`` (unique per server actor, ``SERVER_PORT_STRIDE`` apart) rather
    than from an ephemeral port.

    Deriving from ``start_port`` also keeps the window out of the ephemeral
    range, where vLLM's own ``get_open_port()`` (which only excludes 10 ports
    around ``VLLM_DP_MASTER_PORT``) and every other transient socket on the node
    would compete for it.

    Note the returned base itself sits ``DP_TCPSTORE_PROBE_OFFSET`` *below* the
    window, i.e. inside the preceding server's block. That is only safe because
    nothing binds the base when DP is disabled -- every vLLM caller of
    ``get_next_dp_init_port()`` is guarded on ``data_parallel_size > 1``, and on
    that path vLLM overwrites the master port itself. Recheck if that changes.
    """
    return start_port + DP_TCPSTORE_WINDOW_OFFSET - DP_TCPSTORE_PROBE_OFFSET


def dp_tcpstore_probe_window(dp_master_port: int) -> range:
    """Ports vLLM will probe for the TCPStore, given a ``VLLM_DP_MASTER_PORT``."""
    window_start = dp_master_port + DP_TCPSTORE_PROBE_OFFSET
    return range(window_start, window_start + DP_TCPSTORE_WINDOW)
