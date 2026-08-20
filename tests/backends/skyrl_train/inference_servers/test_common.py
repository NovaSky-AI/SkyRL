"""Tests for inference_servers.common module."""

import socket

from skyrl.backends.skyrl_train.inference_servers.common import (
    find_and_reserve_port,
    get_node_ip,
    get_open_port,
)


class TestGetIp:
    """Tests for get_ip function."""

    def test_get_ip_returns_string(self):
        """Test that get_ip returns a string."""
        ip = get_node_ip()
        assert isinstance(ip, str)
        assert len(ip) > 0
        assert ip != ""
        assert "." in ip or ":" in ip


class TestGetOpenPort:
    """Tests for get_open_port function."""

    def test_get_open_port_os_assigned(self):
        """Test that get_open_port returns an available port (OS assigned)."""
        port = get_open_port()
        assert isinstance(port, int)
        assert 1 <= port <= 65535
        self._verify_port_is_free(port)

    def _verify_port_is_free(self, port: int):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)


def _occupy_port(port: int) -> socket.socket:
    """Bind+listen on *port* to simulate another service (e.g. Tinker API)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", port))
    sock.listen(1)
    return sock


class TestFindAndReservePort:
    """
    get_open_port() probes-then-releases, so concurrent callers
    could both claim the same port.  find_and_reserve_port() holds the socket
    open, forcing subsequent callers to skip to the next port.
    """

    def test_sequential_reservations_are_unique(self):
        port_a, sock_a = find_and_reserve_port(15000)
        try:
            port_b, sock_b = find_and_reserve_port(15000)
            try:
                assert port_a != port_b, f"Duplicate port: {port_a}"
            finally:
                sock_b.close()
        finally:
            sock_a.close()

    def test_occupied_base_port_is_skipped(self):
        """If the base port is already taken, the reservation must pick a higher port."""
        base = get_open_port()
        blocker = _occupy_port(base)
        try:
            port, sock = find_and_reserve_port(base)
            try:
                assert port != base, f"Reserved the occupied port {base}"
                assert port > base
            finally:
                sock.close()
        finally:
            blocker.close()

    def test_overlapping_ranges_no_collision(self):
        """When base port N is occupied, reserving from N and N+1 must
        yield different ports even though both scan through N+1."""
        base = get_open_port()
        blocker = _occupy_port(base)
        try:
            port_0, sock_0 = find_and_reserve_port(base)
            try:
                port_1, sock_1 = find_and_reserve_port(base + 1)
                try:
                    assert port_0 != port_1, f"Port collision: both got {port_0}"
                finally:
                    sock_1.close()
            finally:
                sock_0.close()
        finally:
            blocker.close()

    def test_many_reservations_all_unique(self):
        base = get_open_port()
        blocker = _occupy_port(base)
        sockets = []
        try:
            for _ in range(4):
                port, sock = find_and_reserve_port(base)
                sockets.append((port, sock))

            ports = [p for p, _ in sockets]
            assert len(set(ports)) == len(ports), f"Duplicate among {ports}"
            assert base not in ports
        finally:
            for _, sock in sockets:
                sock.close()
            blocker.close()
