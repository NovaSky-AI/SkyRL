import io
import json
from types import SimpleNamespace

import httpx
import pytest

from skyrl.tinker import sdk_checks as checks


def test_profiler_processing_is_outside_publication_and_sample_timers(monkeypatch):
    clock = [0.0]
    calls = []

    def handle(request):
        calls.append(request.url.path)
        clock[0] += 10.0
        return httpx.Response(200)

    def sample(*args, **kwargs):
        calls.append("sample")
        clock[0] += 3.0
        result = SimpleNamespace(sequences=[SimpleNamespace(tokens=[1, 2])])
        return SimpleNamespace(result=lambda: result)

    sampler = SimpleNamespace(sample=sample)

    def publish():
        calls.append("publish")
        clock[0] += 2.0
        return sampler

    real_client = httpx.Client
    monkeypatch.setattr(checks.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        checks.httpx,
        "Client",
        lambda **kwargs: real_client(base_url="http://example.com/", transport=httpx.MockTransport(handle)),
    )
    report = io.StringIO()
    trainer = SimpleNamespace(save_weights_and_get_sampling_client=publish)
    returned = checks.publish_and_sample(trainer, None, report, "step_1", "http://example.com")
    assert returned is sampler
    assert calls == ["/start_profile", "publish", "/stop_profile", "/start_profile", "sample", "/stop_profile"]
    completed = {
        row["phase"]: row for row in map(json.loads, report.getvalue().splitlines()) if row["status"] == "completed"
    }
    assert completed["step_1/publication"]["seconds"] == 2.0
    assert completed["step_1/sample"]["seconds"] == 3.0
    assert completed["step_1/publication/profiler_stop"]["seconds"] == 10.0


def test_failed_stage_stops_capture_and_retains_the_original_failure(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request.url.path)
        return httpx.Response(200)

    real_client = httpx.Client
    monkeypatch.setattr(
        checks.httpx,
        "Client",
        lambda **kwargs: real_client(base_url="http://example.com/", transport=httpx.MockTransport(handle)),
    )
    with pytest.raises(ValueError, match="failed model request"):
        with checks.profile_inference("http://example.com", io.StringIO(), "sample"):
            raise ValueError("failed model request")
    assert calls == ["/start_profile", "/stop_profile"]


def test_disabled_inference_capture_does_not_contact_an_endpoint(monkeypatch):
    def unexpected_client(**kwargs):
        pytest.fail("disabled profiling must not construct a client")

    monkeypatch.setattr(checks.httpx, "Client", unexpected_client)
    report = io.StringIO()
    with checks.profile_inference(None, report, "sample"):
        pass
    assert report.getvalue() == ""
