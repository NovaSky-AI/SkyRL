"""Check that reservations cover the addresses native HTTP listeners bind."""

import errno
import http.server
import multiprocessing
import os
import socket
import threading
from argparse import Namespace
from functools import partial

import httpx
import pytest
from vllm_router.router_args import RouterArgs

from skyrl.backends.skyrl_train.inference_servers import common, vllm_router
from skyrl.backends.skyrl_train.inference_servers.vllm_router import VLLMRouter


@pytest.fixture
def ipv6_blocker():
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    except OSError as error:
        if error.errno == errno.EAFNOSUPPORT:
            pytest.skip("IPv6 sockets are unavailable")
        raise
    with sock:
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        try:
            sock.bind(("::1", 0))
        except OSError as error:
            if error.errno == errno.EADDRNOTAVAIL:
                pytest.skip("IPv6 loopback is unavailable")
            raise
        sock.listen()
        yield sock


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_reservation_uses_bind_address(host, request):
    if host == "::1":
        request.getfixturevalue("ipv6_blocker")
    start = common.get_open_port()
    port, reservation = common.find_and_reserve_port(start, host=host)
    with reservation:
        assert reservation.getsockname()[1] == port
        with socket.socket(reservation.family, socket.SOCK_STREAM) as competing:
            competing.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with pytest.raises(OSError) as error:
                competing.bind(reservation.getsockname())
                competing.listen()
            assert error.value.errno == errno.EADDRINUSE
        address = reservation.getsockname()
        family = reservation.family
    with socket.socket(family, socket.SOCK_STREAM) as replacement:
        replacement.bind(address)
        replacement.listen()


def test_ipv6_conflict_is_skipped(ipv6_blocker):
    start = ipv6_blocker.getsockname()[1]
    port, reservation = common.find_and_reserve_port(start, host="::")
    with reservation:
        assert port > start
        assert reservation.family == socket.AF_INET6
        assert reservation.getsockname()[0] == "::"


def test_ipv6_reservation_preserves_native_socket_policy(ipv6_blocker):
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as native:
        v6only = native.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen()
        start = blocker.getsockname()[1]
        port, reservation = common.find_and_reserve_port(start, host="::")
        with reservation:
            assert reservation.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == v6only
            assert (port == start) is bool(v6only)


def test_hostname_tries_resolved_addresses(monkeypatch, ipv6_blocker):
    start = ipv6_blocker.getsockname()[1]
    addresses = [
        (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("::1", start, 0, 0)),
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", start)),
    ]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: addresses)
    port, reservation = common.find_and_reserve_port(start, host="router.local")
    with reservation:
        assert port == start
        assert reservation.getsockname() == ("127.0.0.1", start)


def test_occupied_range_still_raises(monkeypatch, ipv6_blocker):
    monkeypatch.setattr(common, "SERVER_PORT_STRIDE", 1)
    with pytest.raises(RuntimeError, match="No available port found"):
        common.find_and_reserve_port(ipv6_blocker.getsockname()[1], host="::1")


class HealthyBackend(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"object":"list","data":[{"id":"dummy","object":"model"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def backend_url():
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), HealthyBackend) as backend:
        thread = threading.Thread(target=backend.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{backend.server_port}"
        finally:
            backend.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize("listener", ["router", "metrics"])
def test_native_router_skips_ipv6_conflict(listener, ipv6_blocker, backend_url, monkeypatch, tmp_path):
    start = ipv6_blocker.getsockname()[1]
    args = RouterArgs(
        host="::",
        port=start if listener == "router" else common.get_open_port(),
        prometheus_host="::1",
        prometheus_port=start if listener == "metrics" else common.get_open_port(),
        worker_urls=[backend_url],
        worker_startup_timeout_secs=10,
        worker_startup_check_interval=1,
    )
    monkeypatch.setattr("skyrl.backends.skyrl_train.inference_servers.vllm_router.get_node_ip", lambda: "::1")
    monkeypatch.setattr(vllm_router, "multiprocessing", multiprocessing.get_context("spawn"))
    monkeypatch.setenv("NO_PROXY", "*")
    router = VLLMRouter(args, log_path=str(tmp_path))
    try:
        monkeypatch.setattr(router, "_wait_until_healthy", partial(router._wait_until_healthy, timeout=15))
        selected = args.port if listener == "router" else args.prometheus_port
        url = router.start()
        with httpx.Client(trust_env=False) as client:
            assert client.get(f"{url}/health", timeout=5).status_code == 200
            assert client.get(f"http://[::1]:{args.prometheus_port}/metrics", timeout=5).status_code == 200
        assert selected > start
    finally:
        router.shutdown()


def test_router_metrics_default_matches_native_listener():
    args = RouterArgs(host="127.0.0.1", port=common.get_open_port(), prometheus_port=common.get_open_port())
    router = VLLMRouter(args)
    try:
        assert router._prometheus_port_reservation.getsockname()[0] == "127.0.0.1"
    finally:
        router.shutdown()


def test_failed_metrics_reservation_releases_router_port(ipv6_blocker, monkeypatch):
    monkeypatch.setattr(common, "SERVER_PORT_STRIDE", 1)
    port = common.get_open_port()
    args = RouterArgs(host="127.0.0.1", port=port, prometheus_host="::1", prometheus_port=ipv6_blocker.getsockname()[1])
    with pytest.raises(RuntimeError, match="No available port found") as error:
        VLLMRouter(args)
    # Keep the traceback alive: garbage collection must not release the socket for us.
    assert error.value.__traceback__ is not None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((args.host, port))
        listener.listen()


@pytest.mark.vllm
def test_actor_reserves_native_http_bind_address(ipv6_blocker, monkeypatch):
    # The actor constructor reserves sockets without starting the model or Ray workers.
    from skyrl.backends.skyrl_train.inference_servers import vllm_server_actor

    monkeypatch.setattr(vllm_server_actor, "get_node_ip", lambda: "::1")
    monkeypatch.setattr("skyrl.train.utils.ray_logging.redirect_actor_output_to_file", lambda: None)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    args = Namespace(tensor_parallel_size=1, pipeline_parallel_size=1)
    start = ipv6_blocker.getsockname()[1]
    actor = vllm_server_actor.VLLMServerActor(args, start_port=start)
    try:
        assert args.host == "::"
        assert actor._port_reservation.getsockname()[:2] == (args.host, args.port)
        actor._port_reservation.close()
        with vllm_server_actor.create_server_socket((args.host, args.port), reuse_port=False) as listener:
            listener.listen()
        assert args.port > start
    finally:
        actor._port_reservation.close()
