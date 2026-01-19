"""Tests for inference_servers.router module."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from skyrl_train.inference_servers.router import (
    InferenceRouter,
    DATA_PLANE_ROUTES,
    CONTROL_PLANE_ROUTES,
)


class TestRouterRoutingLogic:
    """Tests for router routing logic (no actual HTTP calls)."""

    @pytest.fixture
    def router(self):
        """Create a router with mock backends."""
        server_urls = [
            "http://backend1:8000",
            "http://backend2:8000",
            "http://backend3:8000",
        ]
        return InferenceRouter(server_urls, host="0.0.0.0", port=9999)

    def test_session_hash_consistency(self, router):
        """Test that same session ID always maps to same server."""
        session_id = "user-123-session-456"

        # Multiple calls should return the same server
        server1 = router._get_server_for_session(session_id)
        server2 = router._get_server_for_session(session_id)
        server3 = router._get_server_for_session(session_id)

        assert server1 == server2 == server3

    def test_different_sessions_distribute(self, router):
        """Test that different session IDs distribute across servers."""
        # With enough session IDs, we should hit multiple servers
        servers = set()
        for i in range(100):
            session_id = f"session-{i}"
            server = router._get_server_for_session(session_id)
            servers.add(server)

        # Should hit multiple servers (not all requests to one)
        assert len(servers) >= 2

    def test_round_robin_cycles(self, router):
        """Test that round-robin cycles through all servers."""
        servers = []
        for _ in range(6):  # 2 full cycles
            server = router._get_server_round_robin()
            servers.append(server)

        # First 3 should be unique
        assert len(set(servers[:3])) == 3

        # Should repeat the pattern
        assert servers[0] == servers[3]
        assert servers[1] == servers[4]
        assert servers[2] == servers[5]

    def test_control_plane_route_detection(self, router):
        """Test control plane route detection."""
        # Control plane routes
        assert router._is_control_plane_route("/pause") is True
        assert router._is_control_plane_route("/resume") is True
        assert router._is_control_plane_route("/sleep") is True
        assert router._is_control_plane_route("/wake_up") is True
        assert router._is_control_plane_route("/wakeup") is True
        assert router._is_control_plane_route("/reset_prefix_cache") is True
        assert router._is_control_plane_route("/init_weight_transfer") is True
        assert router._is_control_plane_route("/update_weights") is True
        assert router._is_control_plane_route("/finalize_weight_update") is True

        # Data plane routes should NOT be control plane
        assert router._is_control_plane_route("/v1/completions") is False
        assert router._is_control_plane_route("/v1/chat/completions") is False
        assert router._is_control_plane_route("/health") is False
        assert router._is_control_plane_route("/models") is False
        assert router._is_control_plane_route("/tokenize") is False

    def test_data_plane_routes_list(self):
        """Test that data plane routes list is correct."""
        expected = [
            "/v1/completions",
            "/v1/chat/completions",
            "/tokenize",
            "/detokenize",
            "/health",
            "/models",
            "/version",
        ]
        assert DATA_PLANE_ROUTES == expected

    def test_control_plane_routes_list(self):
        """Test that control plane routes list is correct."""
        expected = [
            "/pause",
            "/resume",
            "/sleep",
            "/wake_up",
            "/wakeup",
            "/reset_prefix_cache",
            "/collective_rpc",
            "/init_weight_transfer",
            "/update_weights",
            "/finalize_weight_update",
        ]
        assert CONTROL_PLANE_ROUTES == expected


class TestRouterRequestRouting:
    """Tests for request routing based on headers."""

    @pytest.fixture
    def router(self):
        """Create a router with mock backends."""
        server_urls = [
            "http://backend1:8000",
            "http://backend2:8000",
        ]
        return InferenceRouter(server_urls, host="0.0.0.0", port=9999)

    def test_request_with_session_id_header(self, router):
        """Test that X-Session-ID header triggers session-aware routing."""
        # Create mock request with session header
        request = MagicMock()
        request.headers = {"X-Session-ID": "test-session-123"}

        server1 = router._get_server_for_request(request)
        server2 = router._get_server_for_request(request)

        # Same session should get same server
        assert server1 == server2

    def test_request_without_session_id_header(self, router):
        """Test that missing X-Session-ID header triggers round-robin."""
        # Create mock request without session header
        request = MagicMock()
        request.headers = {}

        servers = []
        for _ in range(4):
            server = router._get_server_for_request(request)
            servers.append(server)

        # Should alternate between servers (round-robin)
        assert servers[0] == servers[2]
        assert servers[1] == servers[3]
        assert servers[0] != servers[1]


class TestRouterInitialization:
    """Tests for router initialization."""

    def test_router_init_with_servers(self):
        """Test router initialization with server list."""
        urls = ["http://a:8000", "http://b:8000"]
        router = InferenceRouter(urls, host="127.0.0.1", port=8080)

        assert router._server_urls == urls
        assert router._host == "127.0.0.1"
        assert router._port == 8080

    def test_router_start_fails_without_servers(self):
        """Test that start fails with empty server list."""
        router = InferenceRouter([], host="0.0.0.0", port=8080)

        with pytest.raises(ValueError, match="No servers"):
            router.start()
