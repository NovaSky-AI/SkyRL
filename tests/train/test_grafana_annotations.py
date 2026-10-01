"""Tests for optional Grafana run-name annotations."""

import json
from unittest.mock import Mock

import httpx
import pytest
import ray

from skyrl.train.config.config import GrafanaAnnotationsConfig
from skyrl.train.utils.grafana_annotations import GrafanaRunAnnotation
from skyrl.utils import grafana


def test_create_update_and_idempotent_finish():
    annotation = GrafanaRunAnnotation(GrafanaAnnotationsConfig(enabled=True), "run-one")
    annotation._request = Mock(return_value={"id": 12})
    annotation.start()
    annotation.finish()
    annotation.finish()
    calls = annotation._request.call_args_list
    assert len(calls) == 2
    assert calls[0].args[:2] == ("POST", "/api/annotations")
    assert calls[1].args[:2] == ("PUT", "/api/annotations/12")
    assert calls[1].args[2]["text"] == "run-one"
    assert calls[1].args[2]["timeEnd"] >= calls[1].args[2]["time"]
    assert "dashboardUID" not in calls[0].args[2]
    assert annotation.annotation_id == 12


def test_disabled_and_failed_api_calls_do_not_raise():
    disabled = GrafanaRunAnnotation(GrafanaAnnotationsConfig(), "disabled")
    disabled._request = Mock()
    disabled.start()
    disabled.finish()
    disabled._request.assert_not_called()
    annotation = GrafanaRunAnnotation(GrafanaAnnotationsConfig(enabled=True), "failed")
    annotation._request = Mock(side_effect=RuntimeError("HTTP failure"))
    annotation.start()
    annotation.finish()
    assert annotation.annotation_id is None


@pytest.fixture
def head_dispatch(monkeypatch):
    monkeypatch.setattr(ray, "is_initialized", Mock(return_value=True))
    monkeypatch.setattr(
        ray,
        "nodes",
        Mock(
            return_value=[
                {"Alive": True, "NodeID": "1" * 56, "Resources": {"CPU": 8}},
                {"Alive": False, "NodeID": "2" * 56, "Resources": {"node:__internal_head__": 1}},
                {"Alive": True, "NodeID": "3" * 56, "Resources": {"node:__internal_head__": 1}},
            ]
        ),
    )
    task = Mock()
    task.options.return_value = task
    task.remote.return_value = "task-ref"
    decorator = Mock(return_value=task)
    monkeypatch.setattr(ray, "remote", Mock(return_value=decorator))
    monkeypatch.setattr(ray, "get", Mock(return_value={"id": 42}))
    monkeypatch.setattr(ray, "cancel", Mock())
    return task, decorator


def test_requests_run_on_live_head_with_hard_affinity(head_dispatch):
    task, decorator = head_dispatch
    config = GrafanaAnnotationsConfig(enabled=True, timeout_seconds=2)
    annotation = GrafanaRunAnnotation(config, "worker-trainer")
    payload = annotation._payload()

    assert annotation._request("POST", "/api/annotations", payload) == {"id": 42}

    ray.remote.assert_called_once_with(num_cpus=0, max_retries=0)
    decorator.assert_called_once_with(grafana.request_from_head)
    strategy = task.options.call_args.kwargs["scheduling_strategy"]
    assert strategy.node_id == "3" * 56
    assert strategy.soft is False
    task.remote.assert_called_once_with("POST", "/api/annotations", payload, config.token_env_var, None, 2)
    ray.get.assert_called_once_with("task-ref", timeout=12)


@pytest.mark.parametrize("heads", [[], ["head-one", "head-two"]])
def test_missing_or_ambiguous_head_does_not_dispatch(head_dispatch, heads):
    ray.nodes.return_value = [
        {"Alive": True, "NodeID": head, "Resources": {"node:__internal_head__": 1}} for head in heads
    ]
    annotation = GrafanaRunAnnotation(GrafanaAnnotationsConfig(enabled=True), "no-head")
    annotation.start()
    annotation.finish()

    ray.remote.assert_not_called()
    assert annotation.annotation_id is None


def test_uninitialized_ray_does_not_dispatch(head_dispatch):
    ray.is_initialized.return_value = False
    annotation = GrafanaRunAnnotation(GrafanaAnnotationsConfig(enabled=True), "uninitialized")
    annotation.start()
    annotation.finish()

    ray.nodes.assert_not_called()
    ray.remote.assert_not_called()
    assert annotation.annotation_id is None


def test_head_task_timeout_is_cancelled_without_failing_run(head_dispatch):
    ray.get.side_effect = ray.exceptions.GetTimeoutError("timed out")
    annotation = GrafanaRunAnnotation(GrafanaAnnotationsConfig(enabled=True), "timeout")
    annotation.start()
    annotation.finish()

    ray.cancel.assert_called_once_with("task-ref", force=True)
    assert annotation.annotation_id is None


@pytest.mark.parametrize(
    "backend,organization,override,expected_url,expected_org",
    [
        (None, None, None, "http://localhost:3000/api/annotations", "1"),
        ("http://localhost:9481", "2", None, "http://localhost:9481/api/annotations", "2"),
        ("https://grafana.example/subpath/", "2", 3, "https://grafana.example/subpath/api/annotations", "3"),
    ],
)
def test_head_http_uses_backend_env_and_organization(
    monkeypatch, backend, organization, override, expected_url, expected_org
):
    monkeypatch.delenv("RAY_GRAFANA_HOST", raising=False)
    monkeypatch.delenv("RAY_GRAFANA_ORG_ID", raising=False)
    if backend:
        monkeypatch.setenv("RAY_GRAFANA_HOST", backend)
    if organization:
        monkeypatch.setenv("RAY_GRAFANA_ORG_ID", organization)
    monkeypatch.setenv("RAY_GRAFANA_IFRAME_HOST", "https://browser-gateway.example")
    monkeypatch.setenv("TEST_ANNOTATION_TOKEN", "test-token")
    config = GrafanaAnnotationsConfig(token_env_var="TEST_ANNOTATION_TOKEN", organization_id=override)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"id": 42})

    client_class = httpx.Client
    client_factory = Mock(side_effect=lambda **kwargs: client_class(transport=httpx.MockTransport(respond), **kwargs))
    monkeypatch.setattr(grafana.httpx, "Client", client_factory)

    assert grafana.request_from_head(
        "POST", "/api/annotations", {"text": "run"}, config.token_env_var, override, config.timeout_seconds
    ) == {"id": 42}
    client_factory.assert_called_once_with(timeout=config.timeout_seconds)
    assert len(requests) == 1
    assert str(requests[0].url) == expected_url
    assert requests[0].headers["X-Grafana-Org-Id"] == expected_org
    assert requests[0].headers["Authorization"] == "Bearer test-token"
    assert json.loads(requests[0].content) == {"text": "run"}


@pytest.mark.parametrize("backend", ["DISABLED", "", "localhost:3000", "file:///tmp/grafana", "http://user:pass@host"])
def test_invalid_backend_is_rejected_before_http(monkeypatch, backend):
    monkeypatch.setenv("RAY_GRAFANA_HOST", backend)
    client = Mock()
    monkeypatch.setattr(grafana.httpx, "Client", client)

    with pytest.raises(ValueError):
        grafana.request_from_head("POST", "/api/annotations", {}, "TEST_ANNOTATION_TOKEN", None, 5)
    client.assert_not_called()


def test_http_failure_is_not_retried(monkeypatch):
    monkeypatch.setenv("RAY_GRAFANA_HOST", "http://localhost:3000")
    monkeypatch.delenv("TEST_ANNOTATION_TOKEN", raising=False)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(403, json={"message": "Forbidden"})

    client_class = httpx.Client
    monkeypatch.setattr(
        grafana.httpx,
        "Client",
        lambda **kwargs: client_class(transport=httpx.MockTransport(respond), **kwargs),
    )
    with pytest.raises(httpx.HTTPStatusError):
        grafana.request_from_head("POST", "/api/annotations", {}, "TEST_ANNOTATION_TOKEN", None, 5)
    assert len(requests) == 1
    assert "Authorization" not in requests[0].headers
