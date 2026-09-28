"""One serving path, driven from both ends.

`skyrl-capture serve` and a training script that brings capture up beside its
inference server now differ in exactly one argument: `blocking`. Everything
below the call is the same code -- one uvicorn configuration, one
`uvicorn.Server.run()`, and a runtime built, started and stopped by ASGI
lifespan.

That is worth asserting rather than assuming, because the two paths used to be
separate and the failure mode of a divergence is nasty: an embedded caller gets
a base URL for a socket that is not listening yet, and the error surfaces
inside the agent as a connection refused.
"""

from __future__ import annotations

import socket
import threading
from typing import Any

import httpx
import pytest
from conftest import free_port

from skyrl_capture.application import CaptureApplication
from skyrl_capture.config import Config, ProxyConfig, RecordConfig, TextUpstream, TitoUpstream
from skyrl_capture.service import CaptureService, CaptureServiceError


def _config(port: int, tmp_path: Any, **proxy: Any) -> Config:
    return Config(
        upstream=TextUpstream(
            type="openai", url="http://127.0.0.1:1/v1", api_key="upstream-secret"
        ),
        record_dir=tmp_path / "traces",
        record=RecordConfig(fsync="never"),
        proxy=ProxyConfig(
            host="127.0.0.1", port=port, public_url=f"http://127.0.0.1:{port}", **proxy
        ),
    )


def _service(tmp_path: Any, **kwargs: Any) -> CaptureService:
    return CaptureService(config=_config(free_port(), tmp_path), **kwargs)


# -- starting ---------------------------------------------------------------
def test_a_nonblocking_start_reaches_a_real_socket(tmp_path):
    """And the whole point: it has finished starting when `start` returns.

    No sleep, no retry, no polling `/healthz` -- the first request a caller
    makes after `start()` must work, because that is what a harness does.
    """
    service = _service(tmp_path)
    service.start(blocking=False)
    try:
        response = httpx.get(f"{service.base_url}/healthz", timeout=5.0)
        assert response.status_code == 200
        assert response.json()["record"] == str(service.config.record_dir)
        # Lifespan startup ran, so the runtime exists and is reachable.
        assert service.runtime.config.upstream.type == "openai"
    finally:
        service.stop()


def test_the_runtime_is_unavailable_before_a_successful_start(tmp_path):
    service = _service(tmp_path)
    with pytest.raises(CaptureServiceError):
        _ = service.runtime


def test_both_planes_answer_on_the_one_port(tmp_path):
    """The dispatcher, through the production application.

    `/tr_.../...` is the raw data plane and everything else is the control
    plane. They share a port and nothing else, and a request for a trajectory
    that does not exist must still be answered *by the data plane* rather than
    404ing out of FastAPI's router.
    """
    service = _service(tmp_path)
    service.start(blocking=False)
    try:
        assert httpx.get(f"{service.base_url}/v1/trajectories", timeout=5.0).status_code == 200
        unknown = httpx.post(
            f"{service.base_url}/route/tr_nope/v1/chat/completions",
            json={"model": "m", "messages": []},
            headers={"authorization": "Bearer nope"},
            timeout=5.0,
        )
        assert unknown.status_code in (401, 404, 410), unknown.text
    finally:
        service.stop()


# -- stopping ---------------------------------------------------------------
def test_stop_closes_the_socket_and_joins_the_thread(tmp_path):
    service = _service(tmp_path)
    service.start(blocking=False)
    url = service.base_url
    service.stop()

    assert service._thread is None, "the serving thread was joined, not abandoned"
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"{url}/healthz", timeout=2.0)


def test_stop_is_safe_to_call_twice(tmp_path):
    service = _service(tmp_path)
    service.start(blocking=False)
    service.stop()
    service.stop()


def test_the_context_manager_stops_the_service(tmp_path):
    with CaptureService(config=_config(free_port(), tmp_path)) as service:
        url = service.base_url
        assert httpx.get(f"{url}/healthz", timeout=5.0).status_code == 200
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"{url}/healthz", timeout=2.0)


def test_a_service_is_single_use(tmp_path):
    """No restart state machine. Build another one."""
    service = _service(tmp_path)
    service.start(blocking=False)
    try:
        with pytest.raises(CaptureServiceError) as caught:
            service.start(blocking=False)
        assert "already been started" in str(caught.value)
    finally:
        service.stop()


# -- failing ----------------------------------------------------------------
def test_an_occupied_port_fails_promptly(tmp_path):
    """Promptly, and not after the full timeout.

    uvicorn reports a failed bind by exiting, which used to look exactly like a
    server that was slow to come up -- so a typo'd port cost three minutes and
    produced "did not start within 180.0s".
    """
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    port = held.getsockname()[1]
    try:
        service = CaptureService(config=_config(port, tmp_path))
        with pytest.raises(CaptureServiceError) as caught:
            service.start(blocking=False, timeout=30.0)
        assert "already in use" in str(caught.value)
    finally:
        held.close()


def test_a_bad_configuration_surfaces_as_a_service_error(tmp_path):
    """A tokens upstream with no tokenizer cannot serve.

    `build_runtime` validates inside lifespan startup, so this is the path a
    lifespan failure takes out to a caller.
    """
    config = _config(free_port(), tmp_path)
    config = config.with_overrides(
        upstream=TitoUpstream(type="tokens", url="http://127.0.0.1:1/generate", model="m")
    )
    service = CaptureService(config=config)
    with pytest.raises(CaptureServiceError) as caught:
        service.start(blocking=False, timeout=30.0)
    assert "tokenizer" in str(caught.value).lower(), str(caught.value)


def test_a_lifespan_failure_is_reported_with_its_own_message(tmp_path, monkeypatch):
    """Not "startup timed out", and not a bare uvicorn exit code."""
    import skyrl_capture.application as application_module

    async def explode(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("the object store was unreachable")

    monkeypatch.setattr(application_module, "build_runtime", explode)
    service = _service(tmp_path)
    with pytest.raises(CaptureServiceError) as caught:
        service.start(blocking=False, timeout=30.0)
    assert "the object store was unreachable" in str(caught.value)


# -- the lifespan owns the runtime -----------------------------------------
def test_the_runtime_starts_and_stops_exactly_once_through_lifespan(tmp_path, monkeypatch):
    """CaptureService must never start or stop a Runtime itself.

    Counted rather than reasoned about, because a second teardown path is the
    kind of thing that works until a shutdown races an in-flight export.
    """
    from skyrl_capture.runtime import Runtime

    counts = {"start": 0, "stop": 0}
    real_start, real_stop = Runtime.start, Runtime.stop

    async def counting_start(self: Runtime) -> None:
        counts["start"] += 1
        await real_start(self)

    async def counting_stop(self: Runtime) -> None:
        counts["stop"] += 1
        await real_stop(self)

    monkeypatch.setattr(Runtime, "start", counting_start)
    monkeypatch.setattr(Runtime, "stop", counting_stop)

    service = _service(tmp_path)
    service.start(blocking=False)
    assert counts == {"start": 1, "stop": 0}
    service.stop()
    assert counts == {"start": 1, "stop": 1}


def test_production_cannot_inject_a_runtime(tmp_path):
    """The application builds the one supported runtime from config.

    The prebuilt-runtime and builder constructors existed for the CLI and for
    fixture convenience; both are gone, and the fixtures drive lifespan.
    """
    import inspect

    parameters = inspect.signature(CaptureApplication).parameters
    assert list(parameters) == ["config"]


# -- addresses --------------------------------------------------------------
def test_overriding_the_port_moves_the_advertised_url(tmp_path):
    """`public_url` is what a trajectory's `base_url` is built from.

    Moving the bind port without moving it hands every caller a URL pointing at
    the old one, which looks perfectly ordinary and answers nothing.
    """
    moved = free_port()
    service = CaptureService(config=_config(free_port(), tmp_path), port=moved)
    assert service.config.proxy.port == moved
    assert service.config.proxy.public_url == f"http://127.0.0.1:{moved}"
    assert service.base_url == f"http://127.0.0.1:{moved}"


def test_an_explicit_public_url_survives_a_port_override(tmp_path):
    """Behind a load balancer it is deliberately not the bind address."""
    config = _config(free_port(), tmp_path)
    config = config.with_overrides(
        proxy=ProxyConfig(
            host=config.proxy.host,
            port=config.proxy.port,
            public_url="https://capture.internal",
        )
    )
    service = CaptureService(config=config, port=free_port())
    assert service.config.proxy.public_url == "https://capture.internal"


def test_binding_every_interface_still_advertises_a_usable_address(tmp_path):
    service = CaptureService(config=_config(free_port(), tmp_path), host="0.0.0.0")
    assert service.base_url.startswith("http://127.0.0.1:")


# -- the CLI ----------------------------------------------------------------
def test_the_cli_serves_through_the_same_call(monkeypatch, tmp_path):
    """`skyrl-capture serve` differs from an embedded caller in one argument."""
    import skyrl_capture.service as service_module
    from skyrl_capture.cli.main import app

    calls: list[dict[str, Any]] = []

    class RecordingService:
        def __init__(self, **kwargs: Any) -> None:
            from skyrl_capture.config import load_config

            self.config = load_config()

        def start(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(service_module, "CaptureService", RecordingService)
    monkeypatch.setenv("UPSTREAM_TYPE", "openai")
    monkeypatch.setenv("UPSTREAM_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("CAPTURE_RECORD_DIR", str(tmp_path / "traces"))

    from typer.testing import CliRunner

    result = CliRunner().invoke(app, ["serve"])
    assert result.exit_code == 0, result.output
    assert calls == [{"blocking": True}], "the CLI blocks; nothing else differs"
    assert threading.active_count() >= 1


def test_serving_without_a_record_directory_is_refused(monkeypatch):
    """Persistence is not a mode, so there is nothing to serve without a place
    to write. Refused at startup, where relaunching fixes it, rather than at
    the first request, by which time the run has been paid for."""
    from typer.testing import CliRunner

    from skyrl_capture.cli.main import app

    monkeypatch.delenv("CAPTURE_RECORD_DIR", raising=False)
    monkeypatch.setenv("UPSTREAM_URL", "http://127.0.0.1:1/v1")

    result = CliRunner().invoke(app, ["serve"])
    assert result.exit_code == 1
    assert "--record-dir" in result.output
