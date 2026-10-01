"""Cache reset must expose failure and forward the connector reset option."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("vllm")

from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (
    VLLMServerActor,
)

pytestmark = pytest.mark.vllm


@pytest.mark.parametrize(
    "body,query,expected_running,expected_connector",
    [
        ({}, "", False, False),
        ({"reset_running_requests": True, "reset_connector": True}, "", True, True),
        ({}, "?reset_external=true", False, True),
    ],
)
def test_prefix_cache_reset_forwards_options(body, query, expected_running, expected_connector):
    app = FastAPI()
    engine = SimpleNamespace(reset_prefix_cache=AsyncMock(return_value=True))
    VLLMServerActor._add_custom_endpoints(app, engine, SimpleNamespace(enable_lora=False))
    with TestClient(app) as client:
        response = client.post("/reset_prefix_cache" + query, json=body)
    assert response.status_code == 200
    assert response.json()["success"] is True
    engine.reset_prefix_cache.assert_awaited_once_with(
        reset_running_requests=expected_running, reset_connector=expected_connector
    )


def test_prefix_cache_reset_reports_blocks_still_held():
    app = FastAPI()
    engine = SimpleNamespace(reset_prefix_cache=AsyncMock(return_value=False))
    VLLMServerActor._add_custom_endpoints(app, engine, SimpleNamespace(enable_lora=False))
    with TestClient(app) as client:
        response = client.post("/reset_prefix_cache", json={"reset_connector": True})
    assert response.status_code == 200
    assert response.json()["success"] is False
