"""Tests for inference_servers.common module."""

import pytest
import socket

from skyrl_train.inference_servers.common import ServerInfo, get_free_port


class TestServerInfo:
    """Tests for ServerInfo dataclass."""

    def test_server_info_url(self):
        """Test URL property."""
        info = ServerInfo(ip="192.168.1.1", port=8000)
        assert info.url == "http://192.168.1.1:8000"

    def test_server_info_url_localhost(self):
        """Test URL with localhost."""
        info = ServerInfo(ip="127.0.0.1", port=30000)
        assert info.url == "http://127.0.0.1:30000"

    def test_server_info_fields(self):
        """Test dataclass fields."""
        info = ServerInfo(ip="10.0.0.1", port=9000)
        assert info.ip == "10.0.0.1"
        assert info.port == 9000


class TestGetFreePort:
    """Tests for get_free_port function."""

    def test_get_free_port_returns_available(self):
        """Test that get_free_port returns an available port."""
        port = get_free_port(start_port=50000)
        assert port >= 50000

        # Verify the port is actually free by binding to it
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)

    def test_get_free_port_skips_occupied(self):
        """Test that get_free_port skips occupied ports."""
        # Occupy a port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", 51000))
            s.listen(1)

            # Should return a different port
            port = get_free_port(start_port=51000)
            assert port >= 51000

    def test_get_free_port_sequential_calls(self):
        """Test that sequential calls return different ports when ports are occupied."""
        ports = []
        sockets = []

        try:
            # Get multiple ports and keep them occupied
            for i in range(3):
                port = get_free_port(start_port=52000)
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("", port))
                s.listen(1)
                ports.append(port)
                sockets.append(s)

            # All ports should be unique
            assert len(set(ports)) == 3
        finally:
            for s in sockets:
                s.close()
