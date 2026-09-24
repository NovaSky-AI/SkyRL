"""Tests for inference_servers.common module."""

import socket
from pathlib import Path

import pytest

from skyrl.backends.skyrl_train.inference_servers.common import (
    DP_TCPSTORE_WINDOW,
    SERVER_PORT_STRIDE,
    ServerInfo,
    compute_dp_master_port,
    default_bind_host,
    dp_tcpstore_probe_window,
    find_and_reserve_port,
    format_http_url,
    get_node_ip,
    get_open_port,
    is_ipv6_address,
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


class TestIPv6SafeAddressing:
    """
    On IPv6-only clusters ``ray.util.get_node_ip_address()`` returns a bare
    IPv6 literal, which must be bracketed in URLs (RFC 3986) and bound with a
    matching wildcard address.
    """

    @pytest.mark.parametrize(
        "host, expected",
        [
            ("127.0.0.1", False),
            ("10.0.0.1", False),
            ("localhost", False),
            ("worker-1.internal", False),
            ("::1", True),
            ("2602:fb33::5", True),
            ("fe80::1%eth0", True),
        ],
    )
    def test_is_ipv6_address(self, host, expected):
        assert is_ipv6_address(host) is expected

    def test_format_http_url_ipv4_unchanged(self):
        assert format_http_url("10.0.0.1", 8100) == "http://10.0.0.1:8100"
        assert format_http_url("10.0.0.1", 8100, "/health") == "http://10.0.0.1:8100/health"

    def test_format_http_url_hostname_unchanged(self):
        assert format_http_url("worker-1", 8000, "/metrics") == "http://worker-1:8000/metrics"

    def test_format_http_url_brackets_ipv6(self):
        assert format_http_url("2602:fb33::5", 8100) == "http://[2602:fb33::5]:8100"
        assert format_http_url("::1", 8100, "/health") == "http://[::1]:8100/health"

    def test_server_info_url_brackets_ipv6(self):
        assert ServerInfo(ip="10.0.0.1", port=8000).url == "http://10.0.0.1:8000"
        assert ServerInfo(ip="2602:fb33::5", port=8000).url == "http://[2602:fb33::5]:8000"

    def test_default_bind_host_matches_address_family(self):
        assert default_bind_host("10.0.0.1") == "0.0.0.0"
        assert default_bind_host("worker-1") == "0.0.0.0"
        assert default_bind_host("2602:fb33::5") == "::"
        assert default_bind_host("::1") == "::"


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


class TestDPMasterPort:
    """
    vLLM's RayExecutorV2 probes a 32-port window for the engine's TCPStore
    rather than binding one known port, so there is nothing to reserve --
    correctness means co-located engines probe *disjoint* windows.

    TODO: delete this class once vllm#50969 lands -- see the removal checklist on
    `compute_dp_master_port` in inference_servers/common.py.
    """

    def _server_start_ports(self, count: int, base: int = 8000) -> list[int]:
        """Mirror ServerGroup's per-actor start_port assignment."""
        return [base + i * SERVER_PORT_STRIDE for i in range(count)]

    def test_windows_are_disjoint_across_servers(self):
        windows = [dp_tcpstore_probe_window(compute_dp_master_port(p)) for p in self._server_start_ports(8)]
        seen: set[int] = set()
        for window in windows:
            ports = set(window)
            assert not (ports & seen), f"Overlapping TCPStore window: {window}"
            seen |= ports
        assert len(seen) == 8 * DP_TCPSTORE_WINDOW

    def test_window_stays_inside_own_server_port_window(self):
        """Each engine's probe range must stay within the SERVER_PORT_STRIDE block
        it owns, so it can never reach into the next actor's block."""
        for start_port in self._server_start_ports(4):
            window = dp_tcpstore_probe_window(compute_dp_master_port(start_port))
            assert window.start >= start_port, f"{window} underruns block at {start_port}"
            assert window.stop <= start_port + SERVER_PORT_STRIDE, f"{window} overruns block at {start_port}"

    def test_window_clears_the_privileged_range(self):
        """The bug this guards: an unseeded master port of 0 probes from port 100."""
        for start_port in self._server_start_ports(4):
            assert dp_tcpstore_probe_window(compute_dp_master_port(start_port)).start > 1024

    def test_window_is_below_the_ephemeral_range(self):
        """Windows must not sit where the OS hands out ephemeral ports, since
        vLLM's own get_open_port() only excludes 10 ports around the base."""
        port_range = Path("/proc/sys/net/ipv4/ip_local_port_range")
        if not port_range.exists():
            pytest.skip("no /proc on this platform")
        ephemeral_start = int(port_range.read_text().split()[0])
        for start_port in self._server_start_ports(64):
            assert dp_tcpstore_probe_window(compute_dp_master_port(start_port)).stop <= ephemeral_start

    def test_pd_disagg_groups_do_not_collide(self):
        """Prefill and decode ServerGroups restart server_idx at 0, so the window
        must key off the assigned start_port, not server_idx."""
        prefill = self._server_start_ports(2, base=8000)
        decode = self._server_start_ports(2, base=8000 + 2 * SERVER_PORT_STRIDE)
        bases = [compute_dp_master_port(p) for p in prefill + decode]
        assert len(set(bases)) == len(bases), f"Duplicate master port across groups: {bases}"
