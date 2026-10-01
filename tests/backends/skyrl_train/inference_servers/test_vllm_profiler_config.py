"""Profiler dictionaries must work before the model engine is constructed."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("vllm")

from vllm.config import ProfilerConfig
from vllm.entrypoints.serve.profile.api_router import attach_router

from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (
    _normalize_profiler_config,
)

pytestmark = pytest.mark.vllm


def test_profiler_override_attaches_working_routes(tmp_path):
    args = Namespace(
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": str(tmp_path),
            "ignore_frontend": True,
            "max_iterations": 16,
        }
    )
    _normalize_profiler_config(args)
    assert isinstance(args.profiler_config, ProfilerConfig)
    assert args.profiler_config.max_iterations == 16
    app = FastAPI()
    app.state.args = args
    engine = SimpleNamespace(start_profile=AsyncMock(), stop_profile=AsyncMock())
    app.state.engine_client = engine
    attach_router(app)
    with TestClient(app) as client:
        assert client.post("/start_profile").status_code == 200
        assert client.post("/stop_profile").status_code == 200
    engine.start_profile.assert_awaited_once()
    engine.stop_profile.assert_awaited_once()


@pytest.mark.parametrize("value", [None, {}])
def test_disabled_profiler_keeps_routes_disabled(value):
    args = Namespace(profiler_config=value)
    _normalize_profiler_config(args)
    app = FastAPI()
    app.state.args = args
    attach_router(app)
    with TestClient(app) as client:
        assert client.post("/start_profile").status_code == 404


def test_typed_profiler_config_is_preserved(tmp_path):
    config = ProfilerConfig(profiler="torch", torch_profiler_dir=str(tmp_path))
    args = Namespace(profiler_config=config)
    _normalize_profiler_config(args)
    assert args.profiler_config is config


def test_invalid_profiler_override_fails_validation():
    with pytest.raises(ValueError):
        _normalize_profiler_config(Namespace(profiler_config={"max_iterations": -1}))
