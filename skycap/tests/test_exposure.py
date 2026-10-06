"""Exposure: the harness routes, alone, reachable from outside this network."""

from __future__ import annotations

import asyncio
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import pytest
import yarl
from aiohttp.test_utils import TestServer

from skycap import CapturePool, CaptureService, record, tunnel
from skycap.cli import build_parser, main
from skycap.exposure import CloudflareQuickTunnel, Exposure, ExternalHost, load_exposure
from tests.mock_openai import API_KEY, MockOpenAI
from tests.test_tokens import client, converse

RECORDING = "tests.test_exposure:RecordingExposure"


class RecordingExposure(Exposure):
    """Exposes the harness listener at its own URL and logs each start and stop, with whether the listener
    still answered then."""

    def __init__(self, log: str, fail: bool = False) -> None:
        self.log = log
        self.fail = fail
        self.harness_url: str | None = None

    def start(self, harness_url: str) -> str:
        self.harness_url = harness_url
        self._write(f"start {harness_url}")
        if self.fail:
            raise RuntimeError("no way in")
        return harness_url + "/"

    def stop(self) -> None:
        self._write(f"stop, listener {'up' if _answers(f'{self.harness_url}/t/tr_x/v1/models') else 'down'}")

    def _write(self, line: str) -> None:
        with open(self.log, "a") as f:
            f.write(line + "\n")


def _answers(url: str) -> bool:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(url, timeout=5)
    except urllib.error.HTTPError:
        return True
    except OSError:
        return False
    return True


def logged(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@asynccontextmanager
async def exposed_service(tmp_path: Path, exposure: Exposure) -> AsyncIterator[CaptureService]:
    upstream = TestServer(MockOpenAI().app())
    await upstream.start_server()
    service = CaptureService(
        str(upstream.make_url("/v1")), api_key=API_KEY, record_dir=str(tmp_path / "record"), host="127.0.0.1",
        exposure=exposure,
    )  # fmt: skip
    try:
        await asyncio.to_thread(service.start)
        yield service
    finally:
        await asyncio.to_thread(service.stop)
        await upstream.close()


# -- building exposures ----------------------------------------------------------
def test_exposures_are_built_by_name_or_import_path(tmp_path: Path) -> None:
    external = load_exposure("external_host", host="203.0.113.7", port=12000)
    assert isinstance(external, ExternalHost) and (external.host, external.port) == ("203.0.113.7", 12000)
    cloudflare = load_exposure("cloudflare", timeout=30.0)
    assert isinstance(cloudflare, CloudflareQuickTunnel) and cloudflare.timeout == 30.0
    custom = load_exposure(RECORDING, log=str(tmp_path / "log"))
    assert isinstance(custom, RecordingExposure)
    assert logged(tmp_path / "log") == []  # building starts nothing


@pytest.mark.parametrize(
    "spec, kwargs, match",
    [
        ("cloudfare", {}, "is one of"),
        ("external_host", {}, "takes"),
        ("external_host", {"host": "203.0.113.7", "port": 70000}, "TCP port"),
        ("cloudflare", {"region": "us"}, "takes"),
        ("tests.missing:Exposure", {}, "can't be imported"),
        ("tests.test_exposure:Missing", {}, "can't be imported"),
        ("tests.test_exposure:logged", {}, "subclass"),
    ],
)
def test_bad_exposures_are_refused(spec: str, kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_exposure(spec, **kwargs)


def test_external_host_binds_its_port_on_every_interface_and_is_reached_at_its_host() -> None:
    assert ExternalHost("203.0.113.7", 11500).bind() == ("0.0.0.0", 11500)
    assert ExternalHost("relay.example.com", 11501).start("http://127.0.0.1:11501") == "http://relay.example.com:11501"
    ipv6 = ExternalHost("2001:db8::7", 11500)
    assert ipv6.bind() == ("::", 11500)
    assert ipv6.start("http://[::1]:11500") == "http://[2001:db8::7]:11500"


def test_the_cli_takes_an_exposure() -> None:
    args = build_parser().parse_args(
        [
            "serve",
            "--upstream-url",
            "http://e/v1",
            "--expose",
            "external_host",
            "--expose-kwargs",
            '{"host": "h", "port": 1}',
        ]
    )
    assert (args.expose, args.expose_kwargs) == ("external_host", {"host": "h", "port": 1})
    assert build_parser().parse_args(["serve", "--upstream-url", "http://e/v1"]).expose is None


# -- serving with an exposure ----------------------------------------------------------
async def test_the_exposed_listener_serves_the_harness_routes_alone(tmp_path: Path) -> None:
    log = tmp_path / "log"
    async with exposed_service(tmp_path, RecordingExposure(str(log))) as service:
        assert service.harness_url is not None and service.harness_url != service.url
        assert service.exposed_url == service.harness_url  # what the exposure returned, without its slash
        async with CapturePool([service.url]) as pool, aiohttp.ClientSession() as http:
            trajectory = await pool.create({"task": "t"})
            assert trajectory.exposed_base_url == f"{service.exposed_url}/t/{trajectory.id}/v1"
            # A harness outside calls the trajectory on the exposed URL ...
            await converse(client(trajectory.exposed_base_url), "q", "r")
            # ... where nothing of the control plane is routed, however the path is spelled.
            private = [
                ("POST", "/trajectories"),
                ("POST", f"/trajectories/{trajectory.id}/finish"),
                ("GET", f"/trajectories/{trajectory.id}"),
                ("GET", "/healthz"),
                ("GET", "/t/%2E%2E/v1/models"),
                ("GET", f"/t/{trajectory.id}/v1/../../../trajectories/{trajectory.id}"),
            ]
            for method, path in private:
                async with http.request(method, yarl.URL(service.exposed_url + path, encoded=True)) as response:
                    assert response.status == 404, path
            result = await trajectory.finish({"reward": 1.0})
    assert result.status == "finished"
    assert len(record.load(tmp_path / "record", trajectory.id).graph) == 4
    # The exposure was closed while the listener still answered, and the listener is gone with the server.
    assert logged(log) == [f"start {service.harness_url}", "stop, listener up"]
    assert not _answers(f"{service.harness_url}/t/tr_x/v1/models")


async def test_without_an_exposure_nothing_more_is_served(tmp_path: Path) -> None:
    upstream = TestServer(MockOpenAI().app())
    await upstream.start_server()
    service = CaptureService(str(upstream.make_url("/v1")), api_key=API_KEY, host="127.0.0.1")
    try:
        url = await asyncio.to_thread(service.start)
        assert service.harness_url is None and service.exposed_url is None
        async with CapturePool([url]) as pool:
            assert (await pool.create()).exposed_base_url is None
    finally:
        await asyncio.to_thread(service.stop)
        await upstream.close()


async def test_a_failed_exposure_fails_the_start_and_closes_everything(tmp_path: Path) -> None:
    log = tmp_path / "log"
    service = CaptureService("http://127.0.0.1:9/v1", host="127.0.0.1", exposure=RecordingExposure(str(log), fail=True))
    with pytest.raises(RuntimeError, match="skycap failed to start"):
        await asyncio.to_thread(service.start)
    harness_url = logged(log)[0].split()[1]
    assert logged(log) == [f"start {harness_url}", "stop, listener up"]
    assert not _answers(f"{harness_url}/t/tr_x/v1/models")


async def test_external_host_serves_on_its_port(tmp_path: Path) -> None:
    port = free_port()
    async with exposed_service(tmp_path, ExternalHost("127.0.0.1", port)) as service:
        assert service.exposed_url == f"http://127.0.0.1:{port}"
        async with CapturePool([service.url]) as pool:
            trajectory = await pool.create()
            await converse(client(trajectory.exposed_base_url), "q")
    assert not _answers(f"http://127.0.0.1:{port}/t/tr_x/v1/models")


async def test_the_cloudflare_exposure_tunnels_to_the_harness_listener(tmp_path: Path, monkeypatch) -> None:
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
    async with exposed_service(tmp_path, load_exposure("cloudflare", timeout=5.0, attempts=1)) as service:
        assert service.exposed_url == "https://corp-provides-trademark-effective.trycloudflare.com"
        async with CapturePool([service.url]) as pool:
            trajectory = await pool.create()
        assert trajectory.exposed_base_url == f"{service.exposed_url}/t/{trajectory.id}/v1"
    (fake,) = opened
    assert fake.local_url == service.harness_url and fake.started == (5.0, 1) and fake.stopped


class BlockingExposure(Exposure):
    """An exposure whose ``start`` doesn't return until it is stopped, like a tunnel that never comes up."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.stops = 0

    def start(self, harness_url: str) -> str:
        self.started.set()
        self.stopped.wait(60)
        raise RuntimeError("stopped before the way in opened")

    def stop(self) -> None:
        self.stops += 1
        self.stopped.set()


async def test_stopping_while_the_exposure_opens_stops_the_server_at_once(tmp_path: Path) -> None:
    exposure = BlockingExposure()
    service = CaptureService("http://127.0.0.1:9/v1", host="127.0.0.1", exposure=exposure)
    starting = asyncio.ensure_future(asyncio.to_thread(service.start))
    assert await asyncio.to_thread(exposure.started.wait, 30)
    stopped_at = time.monotonic()
    assert await asyncio.to_thread(service.stop, 30)
    with pytest.raises(RuntimeError, match="skycap failed to start") as raised:
        await starting
    assert "stopped while it was starting" in str(raised.value.__cause__)
    assert time.monotonic() - stopped_at < 10 and exposure.stops == 1


def test_a_tunnel_stopped_while_it_starts_gives_up(tmp_path: Path, monkeypatch) -> None:
    # A cloudflared that prints its URL and runs, behind which the tunnel never reaches skycap.
    fake = tmp_path / "cloudflared"
    fake.write_text(
        '#!/bin/sh\necho "INF |  https://corp-provides-trademark-effective.trycloudflare.com  |"\nsleep 600\n'
    )
    fake.chmod(0o755)
    monkeypatch.setattr(tunnel, "_cloudflared", lambda stopped=None: str(fake))
    monkeypatch.setattr(tunnel, "_reaches_skycap", lambda url: False)
    opened = tunnel.CloudflareTunnel("http://127.0.0.1:9")
    errors: list[BaseException] = []

    def start() -> None:
        try:
            opened.start(timeout=60.0, attempts=3)
        except BaseException as error:  # noqa: BLE001 - checked below
            errors.append(error)

    thread = threading.Thread(target=start)
    thread.start()
    time.sleep(1.0)
    stopped_at = time.monotonic()
    opened.stop()
    thread.join(10)
    assert not thread.is_alive() and time.monotonic() - stopped_at < 5
    assert len(errors) == 1 and "stopped while it was starting" in str(errors[0])
    # Stopped before it started, a quick tunnel exposure gives up at once.
    quick = CloudflareQuickTunnel()
    quick.stop()
    with pytest.raises(RuntimeError, match="stopped while it was starting"):
        quick.start("http://127.0.0.1:9")


def test_the_cli_refuses_expose_kwargs_that_are_not_an_object_or_without_expose() -> None:
    with pytest.raises(SystemExit, match="must be a JSON object"):
        main(["serve", "--upstream-url", "http://e/v1", "--expose", "cloudflare", "--expose-kwargs", "[]"])
    with pytest.raises(SystemExit, match="without --expose"):
        main(["serve", "--upstream-url", "http://e/v1", "--expose-kwargs", '{"host": "h", "port": 1}'])


async def test_a_start_that_times_out_stops_the_server(tmp_path: Path) -> None:
    exposure = BlockingExposure()
    service = CaptureService("http://127.0.0.1:9/v1", host="127.0.0.1", exposure=exposure)
    with pytest.raises(TimeoutError):
        await asyncio.to_thread(service.start, 1.0)
    # Not left opening its exposure in the background.
    assert exposure.stops == 1 and service._thread is None


# -- the tunnel's process ------------------------------------------------------------
def test_a_tunnel_url_is_told_from_a_failed_tunnel_request() -> None:
    failed = 'ERR failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel": 429 Too Many Requests'
    assert tunnel.TUNNEL_URL.search(failed) is None
    banner = "INF |  https://corp-provides-trademark-effective.trycloudflare.com   |"
    assert tunnel.TUNNEL_URL.search(banner).group(0) == "https://corp-provides-trademark-effective.trycloudflare.com"


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
            "import time; from skycap.tunnel import spawn_tied; "
            f"p = spawn_tied({SLEEPER!r}); print(p.stdout.readline().strip(), flush=True); time.sleep(600)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert parent.stdout is not None
    pid = int(parent.stdout.readline())
    assert alive(pid)
    parent.send_signal(signal.SIGKILL)
    parent.wait()
    assert wait_dead(pid)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="reads /proc")
def test_stopping_a_tied_process_stops_it_and_its_wrapper() -> None:
    process = tunnel.spawn_tied(SLEEPER)
    assert process.stdout is not None
    pid = int(process.stdout.readline())
    started = time.monotonic()
    tunnel.stop_tied(process, timeout=10.0)
    assert time.monotonic() - started < 5.0
    assert process.returncode is not None and wait_dead(pid)
    # A command that exits on its own closes the wrapper's stdout, so a reader of it sees the exit.
    quick = tunnel.spawn_tied([sys.executable, "-c", "print('done')"])
    assert quick.stdout is not None and quick.stdout.read() == "done\n"
    tunnel.stop_tied(quick, timeout=10.0)
