"""CPU tests for the backend_config checks (engine startup and config build) and the AdamParams warning."""

import re
from operator import attrgetter
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("skyrl.backends.skyrl_train_backend")

from skyrl.backends import skyrl_train_backend  # noqa: E402
from skyrl.backends.skyrl_train_backend import (  # noqa: E402
    FSDPBackendOverrides,
    SkyRLTrainBackend,
    _build_skyrl_train_config,
    _unapplied_adam_params,
)
from skyrl.tinker import types  # noqa: E402
from skyrl.tinker.config import EngineConfig, flatten_backend_config  # noqa: E402
from skyrl.tinker.engine import TinkerEngine  # noqa: E402
from skyrl.train.config import OptimizerConfig  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-0.6B"
CHECKPOINT = {"base_model_checkpoint_path": "/models/custom"}
LORA = {"lora_config": types.LoraConfig(rank=8, alpha=32, seed=0)}
FULL_FT = {"lora_config": types.LoraConfig(rank=0, alpha=32, seed=0)}


def _build(backend_config: dict, lora_config: types.LoraConfig | None = None, runtime_role: str = "combined"):
    overrides = FSDPBackendOverrides(runtime_role=runtime_role, **backend_config)
    return _build_skyrl_train_config(BASE_MODEL, overrides, lora_config)


@pytest.mark.parametrize(
    ("backend_config", "kwargs", "expected"),
    [
        ({"trainer.strategy": "megatron"}, {}, "trainer.strategy='megatron': set by --backend ('fsdp')"),
        ({"trainer.critic.model.path": "Qwen/Qwen3-1.7B"}, {}, "trainer.critic.model.path='Qwen/Qwen3-1.7B'"),
        ({"trainer.policy.optimizer_config.scheduler": "cosine"}, {}, "optimizer_config.scheduler='cosine'"),
        ({"trainer.algorithm.use_kl_loss": True}, {}, "use_kl_loss=True"),
        ({"trainer.placement.colocate_all": True}, {"runtime_role": "trainer"}, "colocate_all=True"),
        # Client-carried LoRA keys are rejected outright once a request carries them (#2529),
        # even when the value equals the request's — so the _server_values() rank row that
        # would have rejected only a differing value never gets the chance.
        ({"trainer.policy.model.lora.rank": 32}, FULL_FT, "may not set ['trainer.policy.model.lora.rank']"),
        ({"trainer.policy.model.lora.rank": 8}, LORA, "may not set ['trainer.policy.model.lora.rank']"),
        ({"trainer.policy.model.lora.train_mlp": False}, LORA, "may not set ['trainer.policy.model.lora.train_mlp']"),
        ({"trainer.critic.model.lora.train_attn": False}, LORA, "may not set ['trainer.critic.model.lora.train_attn']"),
        ({"trainer.critic": {"optimizer_config": {"lr": 1e-5}}}, {}, "replaced by AdamParams.learning_rate"),
        ({"trainer.policy": {"torch_profiler_config": {"enable": True}}}, {}, "--torch-profiler"),
        ({"trainer.algorithm.policy_loss_type": "gspo"}, {}, "must be 'regular' or 'dual_clip'"),
        ({"use_ray": False}, {}, "use_ray"),
        # Compared as given: a string is not the server's bool, and SkyRL-Train keeps strings as strings.
        ({"trainer.algorithm.use_kl_loss": "false"}, {}, "use_kl_loss='false'"),
    ],
)
def test_rejected(backend_config, kwargs, expected):
    """A backend_config value the server would not use as given is an error that says why."""
    with pytest.raises(ValueError, match=re.escape(expected)):
        _build(backend_config, **kwargs)


@pytest.mark.parametrize(
    ("backend_config", "kwargs", "key", "value"),
    [
        ({"trainer.strategy": "fsdp", "trainer.algorithm.use_kl_loss": False}, {}, "trainer.strategy", "fsdp"),
        ({"trainer.algorithm.policy_loss_type": "dual_clip"}, {}, "trainer.algorithm.policy_loss_type", "dual_clip"),
        # The request's train_attn/train_mlp reach the config (via _server_values(), #2529).
        (
            {},
            {"lora_config": types.LoraConfig(rank=8, alpha=32, seed=0, train_attn=False, train_mlp=True)},
            "trainer.policy.model.lora.train_attn",
            False,
        ),
        # On an inference-only runtime, create_model's rank-0 alias keeps the rank that enables vLLM LoRA.
        (
            {"trainer.policy.model.lora.rank": 32},
            {"runtime_role": "inference", **FULL_FT},
            "trainer.policy.model.lora.rank",
            32,
        ),
        # The Tinker SDK cannot send alpha, so backend_config's alpha wins over the API placeholder.
        ({"trainer.policy.model.lora": {"alpha": 128}}, LORA, "trainer.policy.model.lora.alpha", 128),
        ({"trainer.placement": {"policy_num_gpus_per_node": 2}}, {}, "trainer.placement.policy_num_gpus_per_node", 2),
        # __post_init__ normalizes these itself, with a warning; that is not the server overriding them.
        ({"generator.inference_engine.enforce_eager": True}, LORA, "generator.inference_engine.enforce_eager", False),
        (
            {"trainer.print_example_interval": 2, "trainer.log_example_interval": 5},
            {},
            "trainer.print_example_interval",
            5,
        ),
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


# The Tinker SDK's AdamParams defaults, and the same with SkyRL-Train's optimizer defaults.
SDK_ADAM = types.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0, grad_clip_norm=0.0)
MATCHING_ADAM = SDK_ADAM.model_copy(update={"beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01, "grad_clip_norm": 1.0})


def test_unapplied_adam_params():
    """Each field that differs from the optimizer names the value in use and the key that sets it."""
    assert _unapplied_adam_params(MATCHING_ADAM, OptimizerConfig(), "critic") == []
    assert _unapplied_adam_params(MATCHING_ADAM, OptimizerConfig(adam_betas=[0.9, 0.999, 0.5]), "critic") == []
    assert _unapplied_adam_params(SDK_ADAM, OptimizerConfig(), "critic") == [
        "beta2=0.95 (the optimizer uses 0.999, set by trainer.critic.optimizer_config.adam_betas)",
        "eps=1e-12 (the optimizer uses 1e-08, not configurable)",
        "weight_decay=0.0 (the optimizer uses 0.01, set by trainer.critic.optimizer_config.weight_decay)",
        "grad_clip_norm=0.0 (the optimizer uses 1.0, set by trainer.critic.optimizer_config.max_grad_norm)",
    ]


def _stub_backend(monkeypatch, strategy: str = "fsdp", optimizer_config_kwargs: dict | None = None):
    """A backend with two policy models and a stub dispatch; returns it and its mocked logger."""
    logger = Mock()
    monkeypatch.setattr(skyrl_train_backend, "logger", logger)
    backend = object.__new__(SkyRLTrainBackend)
    backend.config = FSDPBackendOverrides()
    megatron_config = SimpleNamespace(optimizer_config_kwargs=optimizer_config_kwargs or {})
    policy = SimpleNamespace(optimizer_config=OptimizerConfig(), megatron_config=megatron_config)
    backend._cfg = SimpleNamespace(trainer=SimpleNamespace(strategy=strategy, policy=policy))
    backend._model_ids_to_role = {"a": "policy", "b": "policy"}
    backend._adam_params_checked = set()
    backend._dispatch = Mock(**{"optim_step.return_value": 0.5})
    return backend, logger


def test_optim_step_warns_once_per_model(monkeypatch):
    """Only a model's first optim_step with differing AdamParams warns; the learning rate is always applied."""
    backend, logger = _stub_backend(monkeypatch)
    for model_id, adam_params in [("a", MATCHING_ADAM), ("b", SDK_ADAM), ("b", SDK_ADAM)]:
        backend.optim_step(model_id, types.OptimStepInput(adam_params=adam_params))

    assert logger.warning.call_count == 1 and "b" in logger.warning.call_args.args
    backend._dispatch.set_lr.assert_called_with("policy", 1e-4, model_id="b")


def test_megatron_optimizer_kwargs_are_what_the_optimizer_uses(monkeypatch):
    """On Megatron, optimizer_config_kwargs override optimizer_config, so a request matching them is not warned about."""
    kwargs = {"adam_beta2": 0.95, "adam_eps": 1e-12, "weight_decay": 0.0, "clip_grad": 0.0}
    backend, logger = _stub_backend(monkeypatch, strategy="megatron", optimizer_config_kwargs=kwargs)
    backend.optim_step("a", types.OptimStepInput(adam_params=SDK_ADAM))
    logger.warning.assert_not_called()
