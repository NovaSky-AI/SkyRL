"""CPU tests for the SkyRL-Train backend's backend_config checks and its AdamParams warning."""

import importlib
import re
from operator import attrgetter
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("skyrl.backends.skyrl_train_backend")

from skyrl.backends import skyrl_train_backend  # noqa: E402
from skyrl.backends.skyrl_train_backend import (  # noqa: E402
    FSDPBackendOverrides,
    MegatronBackendOverrides,
    SkyRLTrainBackend,
    _build_skyrl_train_config,
    _unapplied_adam_params,
)
from skyrl.tinker import types  # noqa: E402
from skyrl.tinker.config import same_config_value  # noqa: E402
from skyrl.train.config import OptimizerConfig  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-0.6B"
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
        ({"trainer.policy.model.lora.rank": 32}, FULL_FT, "lora.rank=32"),
        ({"trainer.critic": {"optimizer_config": {"lr": 1e-5}}}, {}, "replaced by AdamParams.learning_rate"),
        ({"trainer.policy": {"torch_profiler_config": {"enable": True}}}, {}, "--torch-profiler"),
        ({"trainer.algorithm.policy_loss_type": "gspo"}, {}, "must be 'regular' or 'dual_clip'"),
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
        ({"trainer.strategy": "fsdp", "trainer.algorithm.use_kl_loss": False}, {}, "trainer.strategy", "fsdp"),
        ({"trainer.algorithm.policy_loss_type": "dual_clip"}, {}, "trainer.algorithm.policy_loss_type", "dual_clip"),
        ({"trainer.policy.model.lora.rank": 8}, LORA, "trainer.policy.model.lora.rank", 8),
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
        ({"trainer.micro_train_batch_size_per_gpu": "4"}, {}, "trainer.micro_train_batch_size_per_gpu", "4"),
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
    """Values equal to the server's, nested spellings, values in another JSON type and SkyRL-Train's own
    normalization build without an error."""
    assert same_config_value(value, attrgetter(key)(_build(backend_config, **kwargs)))


@pytest.mark.parametrize(
    "module",
    [
        "test_fsdp_tinker",
        "test_tinker_torch_profiler",
        "test_multi_lora_megatron",
        "test_multi_lora_megatron_colocated",
        "test_async_sample_routing",
    ],
)
def test_backend_configs_of_the_gpu_tests_pass(module):
    """The configs the Tinker GPU tests launch servers with build without a rejected value."""
    backend_config = importlib.import_module(f"tests.tinker.skyrl_train.{module}").BACKEND_CONFIG
    overrides = MegatronBackendOverrides if backend_config["strategy"] == "megatron" else FSDPBackendOverrides
    _build_skyrl_train_config(BASE_MODEL, overrides(**backend_config), LORA["lora_config"])


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
