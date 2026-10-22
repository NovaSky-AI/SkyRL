"""Torch-free tests for the ``--backend-config`` helpers and the engine's startup checks.

The checks against the built SkyRL-Train config are in
``tests/tinker/skyrl_train/test_backend_config_checks.py``.
"""

import re

import pytest

from skyrl.tinker.config import EngineConfig, flatten_backend_config
from skyrl.tinker.engine import TinkerEngine

BASE_MODEL = "Qwen/Qwen3-0.6B"
CHECKPOINT = {"base_model_checkpoint_path": "/models/custom"}


def test_flatten_backend_config():
    """Nested mappings become dot-keys; lists and empty mappings stay as values."""
    assert flatten_backend_config({"a": {"b": 1, "c": {"d": [2]}}, "e": {}}) == {"a.b": 1, "a.c.d": [2], "e": {}}


def _start_engine(monkeypatch, backend_config: dict, backend: str = "fsdp", **kwargs) -> None:
    """Construct a TinkerEngine whose backend is a stub."""
    monkeypatch.setattr("skyrl.tinker.engine.get_backend_classes", lambda *_a, **_k: (lambda *_: object(), dict))
    TinkerEngine(
        EngineConfig(
            base_model=BASE_MODEL,
            backend=backend,
            backend_config=backend_config,
            database_url="sqlite:///:memory:",
            **kwargs,
        )
    )


@pytest.mark.parametrize(
    ("backend_config", "kwargs", "expected"),
    [
        ({"strategy": "megatron"}, {}, "strategy='megatron': set by --backend ('fsdp')"),
        ({"runtime_role": "trainer"}, {}, "runtime_role='trainer': set by --runtime-role ('combined')"),
        ({"generator.inference_engine": {"served_model_name": "x"}}, CHECKPOINT, "served_model_name='x'"),
    ],
)
def test_engine_rejects_values_it_overwrites(monkeypatch, backend_config, kwargs, expected):
    """The engine sets these on the backend overrides itself, so a different value stops it at startup."""
    with pytest.raises(ValueError, match=re.escape(expected)):
        _start_engine(monkeypatch, backend_config, **kwargs)


@pytest.mark.parametrize(
    ("backend_config", "kwargs"),
    [
        ({"strategy": "fsdp", "runtime_role": "trainer"}, {"runtime_role": "trainer"}),
        ({"generator.inference_engine.served_model_name": BASE_MODEL}, CHECKPOINT),
        ({"strategy": "megatron", "use_ray": True}, {"backend": "jax"}),
    ],
)
def test_engine_accepts(monkeypatch, backend_config, kwargs):
    """Values equal to the engine's are redundant, and the jax backend is not checked."""
    _start_engine(monkeypatch, backend_config, **kwargs)
