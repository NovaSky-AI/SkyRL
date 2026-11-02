"""CPU tests for the backend_config checks (engine startup and config build) and the per-step AdamParams application."""

import re
from operator import attrgetter
from unittest.mock import Mock

import pytest

pytest.importorskip("skyrl.backends.skyrl_train_backend")

from skyrl.backends import skyrl_train_backend  # noqa: E402
from skyrl.backends.skyrl_train_backend import (  # noqa: E402
    FSDPBackendOverrides,
    SkyRLTrainBackend,
    _build_skyrl_train_config,
)
from skyrl.tinker import types  # noqa: E402
from skyrl.tinker.config import EngineConfig, flatten_backend_config  # noqa: E402
from skyrl.tinker.engine import TinkerEngine  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-0.6B"
CHECKPOINT = {"base_model_checkpoint_path": "/models/custom"}
LORA = {"lora_config": types.LoraConfig(rank=8, alpha=32, seed=0)}
FULL_FT = {"lora_config": types.LoraConfig(rank=0, alpha=32, seed=0)}


def _build(backend_config: dict, lora_config: types.LoraConfig | None = None, runtime_role: str = "combined"):
    overrides = FSDPBackendOverrides(runtime_role=runtime_role, **backend_config)
    return _build_skyrl_train_config(BASE_MODEL, overrides, lora_config)


# One case per rule: differing server values, always-replaced keys, outright client-carried
# keys, operator-only features, and the request-replaced loss type.
@pytest.mark.parametrize(
    ("backend_config", "kwargs", "expected"),
    [
        ({"trainer.strategy": "megatron"}, {}, "trainer.strategy='megatron': set by --backend ('fsdp')"),
        ({"trainer.algorithm.use_kl_loss": True}, {}, "use_kl_loss=True"),
        # Client-carried LoRA keys are rejected outright once a request carries them (#2464),
        # even when the value equals the request's.
        ({"trainer.policy.model.lora.rank": 32}, FULL_FT, "may not set ['trainer.policy.model.lora.rank']"),
        ({"trainer.critic": {"optimizer_config": {"lr": 1e-5}}}, {}, "replaced by AdamParams.learning_rate"),
        # set_optimizer_hparams rewrites the param groups before every step, so the same holds
        # for the kwargs spelling and the other AdamParams fields.
        (
            {"trainer.policy.megatron_config.optimizer_config_kwargs": {"lr": 1e-5}},
            {},
            "replaced by AdamParams.learning_rate",
        ),
        ({"trainer.critic.optimizer_config.weight_decay": 0.1}, {}, "replaced by AdamParams.weight_decay"),
        ({"trainer.policy": {"torch_profiler_config": {"enable": True}}}, {}, "--torch-profiler"),
        # The request always carries loss_fn and picks the loss; only "dual_clip" is honoured.
        (
            {"trainer.algorithm.policy_loss_type": "gspo"},
            {},
            "comes from each request's loss_fn, so this value would be ignored",
        ),
        ({"use_ray": False}, {}, "use_ray"),
    ],
)
def test_rejected(backend_config, kwargs, expected):
    """A backend_config value the server would not use as given is an error that says why."""
    with pytest.raises(ValueError, match=re.escape(expected)):
        _build(backend_config, **kwargs)


@pytest.mark.parametrize(
    ("backend_config", "kwargs", "key", "value"),
    [
        ({"trainer.strategy": "fsdp"}, {}, "trainer.strategy", "fsdp"),
        # "dual_clip" is the one policy_loss_type the server honours: ppo_policy_loss reads it
        # from the config to dual-clip loss_fn="ppo".
        ({"trainer.algorithm.policy_loss_type": "dual_clip"}, {}, "trainer.algorithm.policy_loss_type", "dual_clip"),
        # The request's train_attn/train_mlp reach the config (via _server_values(), #2464).
        (
            {},
            {"lora_config": types.LoraConfig(rank=8, alpha=32, seed=0, train_attn=False, train_mlp=True)},
            "trainer.policy.model.lora.train_attn",
            False,
        ),
        # An inference-only runtime receives no training requests, so backend_config is its only
        # way to enable vLLM LoRA serving.
        (
            {"trainer.policy.model.lora.rank": 32},
            {"runtime_role": "inference", **FULL_FT},
            "trainer.policy.model.lora.rank",
            32,
        ),
        ({"trainer.policy.model.lora": {"alpha": 128}}, LORA, "trainer.policy.model.lora.alpha", 128),
        # __post_init__ normalizes these itself, with a warning; that is not the server overriding them.
        ({"generator.inference_engine.enforce_eager": True}, LORA, "generator.inference_engine.enforce_eager", False),
    ],
)
def test_accepted(backend_config, kwargs, key, value):
    """Values equal to the server's, nested spellings and SkyRL-Train's own normalization build without an error."""
    assert attrgetter(key)(_build(backend_config, **kwargs)) == value


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
        ({"strategy": "fsdp"}, {}),
        ({"strategy": "megatron", "use_ray": True}, {"backend": "jax"}),
    ],
)
def test_engine_accepts(monkeypatch, backend_config, kwargs):
    """Values equal to the engine's are redundant, and the jax backend is not checked."""
    _start_engine(monkeypatch, backend_config, **kwargs)


def test_optim_step_applies_the_requests_adam_params_on_every_step(monkeypatch):
    """Every optim_step hands the full AdamParams to the workers, so no configured optimizer value rules."""
    monkeypatch.setattr(skyrl_train_backend, "logger", Mock())
    backend = object.__new__(SkyRLTrainBackend)
    backend.config = FSDPBackendOverrides()
    backend._model_ids_to_role = {"m": "policy"}
    backend._dispatch = Mock(**{"optim_step.return_value": 0.5})

    sdk_adam = types.AdamParams(
        learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0, grad_clip_norm=0.0
    )
    server_defaults = sdk_adam.model_copy(
        update={"beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01, "grad_clip_norm": 1.0}
    )
    for adam_params in (server_defaults, sdk_adam):
        backend.optim_step("m", types.OptimStepInput(adam_params=adam_params))

    assert [
        (call.args[0], call.args[1], call.kwargs["model_id"])
        for call in backend._dispatch.set_optimizer_hparams.call_args_list
    ] == [
        ("policy", server_defaults, "m"),
        ("policy", sdk_adam, "m"),
    ]
