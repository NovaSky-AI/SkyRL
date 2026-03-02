"""Tests for inference_servers.common module."""

import socket

from skyrl_train.inference_servers.common import (
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
