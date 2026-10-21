"""CPU tests for the SkyRL-Train backend's backend_config checks and its AdamParams warning."""

import importlib
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("skyrl.backends.skyrl_train_backend")

from skyrl.backends import skyrl_train_backend  # noqa: E402
from skyrl.backends.skyrl_train_backend import (  # noqa: E402
    FSDPBackendOverrides,
    MegatronBackendOverrides,
    SkyRLTrainBackend,
    _apply_critic_lora,
    _build_skyrl_train_config,
    _config_value,
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
        # Not in _server_values: __post_init__ copies the deprecated log_example_interval over it.
        ({"trainer.print_example_interval": 2, "trainer.log_example_interval": 5}, {}, "=2: the server uses 5"),
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
    ],
)
def test_accepted(backend_config, kwargs, key, value):
    """Values equal to the server's, nested spellings and values in another JSON type are applied."""
    assert same_config_value(value, _config_value(_build(backend_config, **kwargs), key))


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


def test_critic_lora_comes_from_its_create_model_request():
    """The critic's rank must match its request; a backend_config alpha wins, as for the policy."""
    user_values = {"trainer.critic.model.lora.alpha": 64}
    cfg = _build(user_values)
    _apply_critic_lora(cfg, LORA["lora_config"], user_values)
    assert (cfg.trainer.critic.model.lora.rank, cfg.trainer.critic.model.lora.alpha) == (8, 64)
    with pytest.raises(ValueError, match=re.escape("trainer.critic.model.lora.rank=16")):
        _apply_critic_lora(_build({}), LORA["lora_config"], {"trainer.critic.model.lora.rank": 16})


# The Tinker SDK's AdamParams defaults, and the same with SkyRL-Train's optimizer defaults.
SDK_ADAM = types.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0, grad_clip_norm=0.0)
MATCHING_ADAM = SDK_ADAM.model_copy(update={"beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01, "grad_clip_norm": 1.0})


def test_unapplied_adam_params():
    """Each field that differs from the optimizer names the value in use and the key that sets it."""
    assert _unapplied_adam_params(MATCHING_ADAM, OptimizerConfig(), "critic") == []
    assert _unapplied_adam_params(SDK_ADAM, OptimizerConfig(), "critic") == [
        "beta2=0.95 (the optimizer uses 0.999, set by trainer.critic.optimizer_config.adam_betas)",
        "eps=1e-12 (the optimizer uses 1e-08, not configurable)",
        "weight_decay=0.0 (the optimizer uses 0.01, set by trainer.critic.optimizer_config.weight_decay)",
        "grad_clip_norm=0.0 (the optimizer uses 1.0, set by trainer.critic.optimizer_config.max_grad_norm)",
    ]


def test_optim_step_warns_once_per_model(monkeypatch):
    """Only a model's first optim_step with differing AdamParams warns; the learning rate is always applied."""
    logger = Mock()
    monkeypatch.setattr(skyrl_train_backend, "logger", logger)
    backend = object.__new__(SkyRLTrainBackend)
    backend.config = FSDPBackendOverrides()
    backend._cfg = SimpleNamespace(trainer=SimpleNamespace(policy=SimpleNamespace(optimizer_config=OptimizerConfig())))
    backend._model_ids_to_role = {"a": "policy", "b": "policy"}
    backend._adam_params_checked = set()
    backend._dispatch = Mock(**{"optim_step.return_value": 0.5})

    for model_id, adam_params in [("a", MATCHING_ADAM), ("b", SDK_ADAM), ("b", SDK_ADAM)]:
        backend.optim_step(model_id, types.OptimStepInput(adam_params=adam_params))

    assert logger.warning.call_count == 1 and "b" in logger.warning.call_args.args
    backend._dispatch.set_lr.assert_called_with("policy", 1e-4, model_id="b")
