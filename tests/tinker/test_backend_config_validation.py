"""Torch-free tests for the ``--backend-config`` checks on the SkyRL-Train backends.

The checks against the built SkyRL-Train config live in
``tests/tinker/skyrl_train/test_backend_config_checks.py``.
"""

import re
from dataclasses import dataclass
from enum import Enum

import pytest

from skyrl.tinker.config import (
    EngineConfig,
    backend_config_conflicts,
    flatten_backend_config,
    same_config_value,
)
from skyrl.tinker.engine import TinkerEngine

BASE_MODEL = "Qwen/Qwen3-0.6B"


class _Color(str, Enum):
    RED = "red"


@dataclass
class _Server:
    host: str
    port: int


def test_flatten_backend_config():
    """Nested mappings become dot-keys; lists and empty mappings stay as values."""
    assert flatten_backend_config(
        {
            "trainer.placement": {"colocate_all": False, "policy_num_gpus_per_node": 4},
            "trainer.policy.optimizer_config.adam_betas": [0.9, 0.95],
            "generator.inference_engine.engine_init_kwargs": {},
        }
    ) == {
        "trainer.placement.colocate_all": False,
        "trainer.placement.policy_num_gpus_per_node": 4,
        "trainer.policy.optimizer_config.adam_betas": [0.9, 0.95],
        "generator.inference_engine.engine_init_kwargs": {},
    }


@pytest.mark.parametrize(
    ("requested", "actual"),
    [
        (4, 4.0),
        ("4", 4),
        ("1e-5", 1e-5),
        ("true", True),
        ([0.9, 0.95], (0.9, 0.95)),
        ("red", _Color.RED),
        ([{"host": "a", "port": 1}], [_Server("a", 1)]),
        ({"max_model_len": 4096}, {"max_model_len": 4096, "enforce_eager": True}),
    ],
)
def test_same_config_value_allows_type_conversion(requested, actual):
    """OmegaConf converts values to the field's type, which must not count as a change."""
    assert same_config_value(requested, actual)


@pytest.mark.parametrize(
    ("requested", "actual"),
    [
        (4, 5),
        ("Qwen/Qwen3-0.6B", "qwen/qwen3-0.6b"),
        ("false", True),
        ([0.9, 0.95], [0.9, 0.999]),
        ({"max_model_len": 4096}, {"max_model_len": 2048}),
    ],
)
def test_same_config_value_detects_changes(requested, actual):
    """Different values stay different, and strings compare case-sensitively."""
    assert not same_config_value(requested, actual)


def test_backend_config_conflicts():
    """Differing server-set values and unused keys are reported; equal values are not."""
    problems = backend_config_conflicts(
        {
            "trainer.strategy": "megatron",
            "runtime_role": "combined",
            "trainer.policy.torch_profiler_config.enable": True,
        },
        {"trainer.strategy": ("fsdp", "set by --backend"), "runtime_role": ("combined", "set by --runtime-role")},
        unused_keys={"trainer.policy.torch_profiler_config": "use /start_profiling"},
    )

    assert problems == [
        "trainer.policy.torch_profiler_config.enable=True: use /start_profiling",
        "trainer.strategy='megatron': set by --backend ('fsdp')",
    ]


def _start_engine(monkeypatch, backend_config: dict, **kwargs) -> dict:
    """Construct a TinkerEngine with a stub backend and return the overrides it passed on."""
    captured = {}

    class BackendConfig:
        def __init__(self, **overrides):
            captured.update(overrides)

    class Backend:
        def __init__(self, _model, _config):
            pass

    monkeypatch.setattr("skyrl.tinker.engine.get_backend_classes", lambda *_args, **_kwargs: (Backend, BackendConfig))
    TinkerEngine(
        EngineConfig(
            base_model=BASE_MODEL,
            backend=kwargs.pop("backend", "fsdp"),
            backend_config=backend_config,
            database_url="sqlite:///:memory:",
            **kwargs,
        )
    )
    return captured


@pytest.mark.parametrize(
    ("backend_config", "kwargs", "expected"),
    [
        ({"strategy": "megatron"}, {}, "strategy='megatron': set by --backend ('fsdp')"),
        ({"runtime_role": "trainer"}, {}, "runtime_role='trainer': set by --runtime-role ('combined')"),
        ({"use_ray": False}, {}, "use_ray=False: only read by the jax backend"),
        (
            {"generator.inference_engine": {"served_model_name": "custom"}},
            {"base_model_checkpoint_path": "/models/custom"},
            "generator.inference_engine.served_model_name='custom'",
        ),
    ],
)
def test_engine_rejects_values_it_would_overwrite(monkeypatch, backend_config, kwargs, expected):
    """The engine sets these on the backend overrides itself, so a different value stops it at startup."""
    with pytest.raises(ValueError, match=re.escape(expected)):
        _start_engine(monkeypatch, backend_config, **kwargs)


def test_engine_accepts_values_equal_to_its_own(monkeypatch):
    """Redundant keys have the effect the user expects, so they stay allowed."""
    captured = _start_engine(
        monkeypatch,
        {"strategy": "fsdp", "runtime_role": "trainer", "generator.inference_engine.served_model_name": BASE_MODEL},
        runtime_role="trainer",
        base_model_checkpoint_path="/models/custom",
    )

    assert captured["runtime_role"] == "trainer"
    assert captured["generator.inference_engine.served_model_name"] == BASE_MODEL


def test_engine_does_not_check_the_jax_backend(monkeypatch):
    """use_ray selects the Ray JAX backend, so it is a valid jax key."""
    _start_engine(monkeypatch, {"use_ray": True}, backend="jax")
