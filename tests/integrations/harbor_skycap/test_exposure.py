"""Exposing skycap's harness routes to agents inside sandboxes: the config, the built-ins, a custom
exposure named by import path, the gateway, and the URL the generator hands the agent."""

import asyncio
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest

pytest.importorskip("skycap")
pytest.importorskip("harbor")

import aiohttp  # noqa: E402

from examples.train_integrations.harbor_skycap import tunnel  # noqa: E402
from examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap import (  # noqa: E402
    HarborSkycapConfig,
    _exposure,
)
from examples.train_integrations.harbor_skycap.exposure import (  # noqa: E402
    HARNESS_ROUTE,
    CloudflareQuickTunnel,
    Exposure,
    ExternalHost,
    HarnessGateway,
    exposure_factory,
)
from examples.train_integrations.harbor_skycap.harbor_generator import (  # noqa: E402
    HarborSkycapGenerator,
)
from examples.train_integrations.harbor_skycap.tunnel import TUNNEL_URL  # noqa: E402
from skycap import CaptureService  # noqa: E402
from tests.integrations.harbor_skycap.test_harbor_skycap import (  # noqa: E402
    generator_cfg,
    harbor_cfg,
)

pytestmark = pytest.mark.integrations

RECORDING = "tests.integrations.harbor_skycap.test_exposure:RecordingExposure"


class RecordingExposure(Exposure):
    """Exposes the gateway at its own loopback URL, and logs each start and stop to ``log`` (one line each),
    which works across Ray's worker processes."""

    def __init__(self, log: str, fail_on: Optional[int] = None) -> None:
        self.log = log
        self.fail_on = fail_on
        self.index: Optional[int] = None

    def start(self, gateway_url: str, index: int) -> str:
        self.index = index
        self._write(f"start {index} {gateway_url}")
        if index == self.fail_on:
            raise RuntimeError(f"no way in for server {index}")
        return gateway_url + "/"

    def stop(self) -> None:
        self._write(f"stop {self.index}")

    def _write(self, line: str) -> None:
        with open(self.log, "a") as f:
            f.write(line + "\n")


def logged(log: Path) -> List[str]:
    return log.read_text().splitlines() if log.exists() else []


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(url: str) -> tuple[int, str]:
    """Status and body of a GET, or (0, error) when nothing answers."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()
    except urllib.error.URLError as error:
        return 0, str(error)


@pytest.fixture
def service(tmp_path):
    """A skycap server with no engine behind it: enough to tell its 404s and 502s from a gateway's."""
    service = CaptureService("http://127.0.0.1:9/v1", record_dir=str(tmp_path / "record"), host="127.0.0.1")
    service.start()
    yield service
    service.stop()


def test_nothing_is_exposed_by_default() -> None:
    cfg = HarborSkycapConfig.from_cli_overrides([])
    assert cfg.skycap.exposure.type == "none"
    assert _exposure(cfg.skycap.exposure) is None


def test_the_config_names_a_custom_exposure_by_import_path_with_its_kwargs(tmp_path) -> None:
    cfg = HarborSkycapConfig.from_cli_overrides(
        [f"skycap.exposure.type={RECORDING}", f"skycap.exposure.kwargs.log={tmp_path / 'log'}"]
    )
    exposure = _exposure(cfg.skycap.exposure)()
    assert isinstance(exposure, RecordingExposure) and exposure.log == str(tmp_path / "log")
    # Building it starts nothing.
    assert logged(tmp_path / "log") == []


def test_the_built_ins_are_built_from_their_options() -> None:
    external = exposure_factory("external_host", host="203.0.113.7", port=12000)()
    assert isinstance(external, ExternalHost) and (external.host, external.port) == ("203.0.113.7", 12000)
    cloudflare = exposure_factory("cloudflare", kwargs={"timeout": 30.0})()
    assert isinstance(cloudflare, CloudflareQuickTunnel) and cloudflare.timeout == 30.0


@pytest.mark.parametrize(
    "kind, options, match",
    [
        ("cloudfare", {}, "must be one of"),
        ("external_host", {}, "needs skycap.exposure.host"),
        ("external_host", {"host": "203.0.113.7", "port": 70000}, "must be a TCP port"),
        ("cloudflare", {"host": "203.0.113.7"}, "is for type=external_host"),
        ("cloudflare", {"kwargs": {"region": "us"}}, "don't match"),
        ("none", {"kwargs": {"log": "x"}}, "takes no kwargs"),
        (RECORDING, {}, "don't match"),
        ("tests.integrations.harbor_skycap.missing:Exposure", {}, "can't be imported"),
        ("tests.integrations.harbor_skycap.test_exposure:Missing", {}, "can't be imported"),
        ("tests.integrations.harbor_skycap.test_exposure:logged", {}, "subclass of"),
    ],
)
def test_bad_exposure_configs_are_refused_before_anything_starts(kind, options, match) -> None:
    with pytest.raises(ValueError, match=match):
        exposure_factory(kind, **options)


def test_external_host_puts_server_i_on_port_plus_i_on_every_interface() -> None:
    assert ExternalHost("203.0.113.7", 11500).bind(0) == ("0.0.0.0", 11500)
    assert ExternalHost("203.0.113.7", 11500).bind(3) == ("0.0.0.0", 11503)
    assert ExternalHost("relay.example.com", 11500).start("http://127.0.0.1:11502", 2) == (
        "http://relay.example.com:11502"
    )
    ipv6 = ExternalHost("2001:db8::7", 11500)
    assert ipv6.bind(1) == ("::", 11501)
    assert ipv6.start("http://[::1]:11501", 1) == "http://[2001:db8::7]:11501"


def test_external_host_serves_only_the_harness_routes_on_its_port(service) -> None:
    port = free_port()
    exposure = ExternalHost("127.0.0.1", port)
    url = exposure.open(service.url, 0)
    try:
        assert url == f"http://127.0.0.1:{port}"
        status, body = get(f"{url}/t/tr_none/v1/models")
        assert status == 404 and "unknown trajectory" in body  # skycap's own answer, through the gateway
        assert get(f"{url}/healthz")[0] == 404 and get(f"{service.url}/healthz")[0] == 200
    finally:
        exposure.close()
    assert get(f"{url}/t/tr_none/v1/models")[0] == 0


def test_a_custom_exposure_is_started_on_the_gateway_and_stopped_before_it(service, tmp_path) -> None:
    log = tmp_path / "log"
    exposure = exposure_factory(RECORDING, kwargs={"log": str(log)})()
    url = exposure.open(service.url, 3)
    try:
        assert logged(log) == [f"start 3 {url}"] and url.startswith("http://127.0.0.1:")
        status, body = get(f"{url}/t/tr_none/v1/models")
        assert status == 404 and "unknown trajectory" in body
    finally:
        exposure.close()
    assert logged(log) == [f"start 3 {url}", "stop 3"]
    assert get(f"{url}/t/tr_none/v1/models")[0] == 0
    exposure.close()
    assert logged(log) == [f"start 3 {url}", "stop 3"]


def test_a_failed_start_stops_the_exposure_and_its_gateway(service, tmp_path) -> None:
    log = tmp_path / "log"
    exposure = exposure_factory(RECORDING, kwargs={"log": str(log), "fail_on": 0})()
    with pytest.raises(RuntimeError, match="no way in"):
        exposure.open(service.url, 0)
    started, stopped = logged(log)
    assert stopped == "stop 0"
    assert get(f"{started.split()[-1]}/t/tr_none/v1/models")[0] == 0


def test_the_cloudflare_exposure_opens_a_quick_tunnel_to_the_gateway(service, monkeypatch) -> None:
    opened = []

    class FakeTunnel:
        def __init__(self, local_url: str) -> None:
            self.local_url = local_url
            opened.append(self)

        def start(self, timeout: float, attempts: int) -> str:
            self.started = (timeout, attempts)
            return "https://corp-provides-trademark-effective.trycloudflare.com"

        def stop(self) -> None:
            self.stopped = True

    monkeypatch.setattr(tunnel, "CloudflareTunnel", FakeTunnel)
    exposure = exposure_factory("cloudflare", kwargs={"timeout": 5.0, "attempts": 1})()
    url = exposure.open(service.url, 0)
    exposure.close()

    (fake,) = opened
    assert url == "https://corp-provides-trademark-effective.trycloudflare.com"
    assert fake.local_url.startswith("http://127.0.0.1:") and fake.started == (5.0, 1) and fake.stopped


SLEEPER = [sys.executable, "-c", "import os, time; print(os.getpid(), flush=True); time.sleep(600)"]


def alive(pid: int) -> bool:
    """Whether ``pid`` runs: not gone, and not a zombie its new parent hasn't reaped."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def wait_dead(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    return not alive(pid)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="reads /proc")
def test_a_tied_process_stops_when_its_parent_is_killed() -> None:
    # The parent starts the process tied and is then killed with SIGKILL, so nothing of its own runs.
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time; from examples.train_integrations.harbor_skycap.tunnel import spawn_tied; "
            f"p = spawn_tied({SLEEPER!r}); print(p.stdout.readline().strip(), flush=True); time.sleep(600)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    pid = int(parent.stdout.readline())
    assert alive(pid)
    parent.send_signal(signal.SIGKILL)
    parent.wait()
    assert wait_dead(pid)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="reads /proc")
def test_stopping_a_tied_process_stops_it_and_its_wrapper() -> None:
    process = tunnel.spawn_tied(SLEEPER)
    pid = int(process.stdout.readline())
    started = time.monotonic()
    tunnel.stop_tied(process, timeout=10.0)
    assert time.monotonic() - started < 5.0
    assert process.returncode is not None and wait_dead(pid)
    # A command that exits on its own closes the wrapper's stdout, so a reader of it sees the exit.
    quick = tunnel.spawn_tied([sys.executable, "-c", "print('done')"])
    assert quick.stdout.read() == "done\n"
    tunnel.stop_tied(quick, timeout=10.0)


def test_a_failed_tunnel_request_is_not_taken_for_the_tunnel() -> None:
    failed = 'ERR failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel": 429 Too Many Requests'
    assert TUNNEL_URL.search(failed) is None
    banner = "INF |  https://corp-provides-trademark-effective.trycloudflare.com   |"
    assert TUNNEL_URL.search(banner).group(0) == "https://corp-provides-trademark-effective.trycloudflare.com"


def test_only_the_harness_routes_are_exposed() -> None:
    assert HARNESS_ROUTE.match("/t/tr_ab12/v1/chat/completions")
    assert HARNESS_ROUTE.match("/t/tr_ab12/v1/models")
    for private in ("/trajectories", "/trajectories/tr_ab12/finish", "/trajectories/tr_ab12", "/healthz"):
        assert not HARNESS_ROUTE.match(private)


@pytest.mark.asyncio
async def test_the_gateway_forwards_harness_calls_and_hides_the_control_plane(service) -> None:
    gateway = HarnessGateway(service.url)
    gateway_url = f"http://127.0.0.1:{await asyncio.to_thread(gateway.start)}"
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(f"{service.url}/trajectories", json={"meta": {}}) as response:
                trajectory_id = (await response.json())["id"]
            async with http.post(f"{gateway_url}/trajectories", json={"meta": {}}) as response:
                assert response.status == 404
            async with http.post(f"{gateway_url}/trajectories/{trajectory_id}/finish", json={}) as response:
                assert response.status == 404
            # No engine behind skycap: a forwarded chat call comes back as skycap's own 502.
            chat = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            async with http.post(f"{gateway_url}/t/{trajectory_id}/v1/chat/completions", json=chat) as response:
                assert response.status == 502
                assert "upstream unavailable" in (await response.json())["error"]["message"]
    finally:
        await asyncio.to_thread(gateway.stop)


@pytest.mark.asyncio
async def test_the_agent_gets_its_trajectory_on_its_servers_exposed_url() -> None:
    servers = ["http://10.0.0.5:8080", "http://10.0.0.6:8080/"]
    gen = HarborSkycapGenerator(
        generator_cfg(),
        harbor_cfg(),
        servers,
        harness_urls={servers[0]: "https://edge.example/", servers[1]: "http://203.0.113.7:11501"},
    )
    try:
        first = SimpleNamespace(server="http://10.0.0.5:8080", base_url="http://10.0.0.5:8080/t/tr_ab12/v1")
        second = SimpleNamespace(server="http://10.0.0.6:8080", base_url="http://10.0.0.6:8080/t/tr_cd34/v1")
        assert gen._agent_url(first) == "https://edge.example/t/tr_ab12/v1"
        assert gen._agent_url(second) == "http://203.0.113.7:11501/t/tr_cd34/v1"
        config = gen._trial_config("/tasks/t", gen._agent_url(first), cache_salt=None)
        assert config["agent"]["kwargs"]["api_base"] == "https://edge.example/t/tr_ab12/v1"
    finally:
        await gen.close()


@pytest.mark.asyncio
async def test_without_exposure_the_agent_gets_the_servers_own_url() -> None:
    gen = HarborSkycapGenerator(generator_cfg(), harbor_cfg(), ["http://10.0.0.5:8080"])
    try:
        trajectory = SimpleNamespace(server="http://10.0.0.5:8080", base_url="http://10.0.0.5:8080/t/tr_ab12/v1")
        assert gen._agent_url(trajectory) == trajectory.base_url
    finally:
        await gen.close()


def test_the_generator_refuses_exposed_urls_that_miss_a_server() -> None:
    with pytest.raises(ValueError, match="every capture server"):
        HarborSkycapGenerator(
            generator_cfg(),
            harbor_cfg(),
            ["http://10.0.0.5:8080", "http://10.0.0.6:8080"],
            harness_urls={"http://10.0.0.5:8080": "https://edge.example"},
        )
