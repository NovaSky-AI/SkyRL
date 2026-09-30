"""Tests for optional Grafana run-name annotations."""

import json
from unittest.mock import Mock

from skyrl.train.config.config import GrafanaAnnotationsConfig
from skyrl.train.utils.grafana_annotations import GrafanaRunAnnotation


def test_create_update_and_idempotent_finish(tmp_path):
    annotation = GrafanaRunAnnotation(GrafanaAnnotationsConfig(enabled=True), "run-one", str(tmp_path))
    annotation._request = Mock(return_value={"id": 12})
    annotation.start()
    annotation.finish("failed")
    annotation.finish("failed")
    calls = annotation._request.call_args_list
    assert len(calls) == 2
    assert calls[0].args[:2] == ("POST", "/api/annotations")
    assert calls[1].args[:2] == ("PUT", "/api/annotations/12")
    assert calls[1].args[2]["text"] == "run-one"
    assert calls[1].args[2]["timeEnd"] >= calls[1].args[2]["time"]
    assert "dashboardUID" not in calls[0].args[2]
    record = json.loads(annotation.path.read_text())
    assert record["run_status"] == "failed"
    assert record["annotation_id"] == 12


def test_disabled_and_failed_api_calls_do_not_raise(tmp_path):
    disabled = GrafanaRunAnnotation(GrafanaAnnotationsConfig(), "disabled", str(tmp_path))
    disabled._request = Mock()
    disabled.start()
    disabled.finish("success")
    disabled._request.assert_not_called()
    annotation = GrafanaRunAnnotation(GrafanaAnnotationsConfig(enabled=True), "failed", str(tmp_path))
    annotation._request = Mock(side_effect=RuntimeError("HTTP failure"))
    annotation.start()
    annotation.finish("failed")
    assert json.loads(annotation.path.read_text())["run_status"] == "failed"
