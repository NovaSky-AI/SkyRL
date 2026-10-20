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
    _unapplied_adam_params,
)
from skyrl.tinker import types  # noqa: E402
from skyrl.tinker.config import flatten_backend_config, same_config_value  # noqa: E402
from skyrl.train.config import OptimizerConfig, SkyRLTrainConfig  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-0.6B"


def _lora(rank: int, alpha: float = 32) -> types.LoraConfig:
    return types.LoraConfig(rank=rank, alpha=alpha, seed=0)


def _build(backend_config: dict, lora_config: types.LoraConfig | None = None, **overrides) -> SkyRLTrainConfig:
    return _build_skyrl_train_config(BASE_MODEL, FSDPBackendOverrides(**overrides, **backend_config), lora_config)


@pytest.mark.parametrize(
    ("backend_config", "expected"),
    [
        ({"trainer.strategy": "megatron"}, "trainer.strategy='megatron': set by --backend ('fsdp')"),
        ({"trainer.critic.model.path": "Qwen/Qwen3-1.7B"}, "trainer.critic.model.path='Qwen/Qwen3-1.7B'"),
        ({"trainer.policy.optimizer_config.scheduler": "cosine"}, "trainer.policy.optimizer_config.scheduler"),
        ({"trainer.critic.optimizer_config.num_warmup_steps": 10}, "trainer.critic.optimizer_config.num_warmup_steps"),
        ({"trainer.algorithm.use_kl_loss": True}, "trainer.algorithm.use_kl_loss=True"),
    ],
)
def test_values_the_server_sets_differently_are_rejected(backend_config, expected):
    """Each entry of _server_values rejects a different backend_config value, naming what sets it."""
    with pytest.raises(ValueError, match=re.escape(expected)):
        _build(backend_config)


def test_values_equal_to_the_servers_are_accepted():
    """Redundant keys have the effect the user expects, so they stay allowed."""
    cfg = _build(
        {
            "trainer.strategy": "fsdp",
            "trainer.critic.model.path": BASE_MODEL,
            "trainer.policy.optimizer_config.scheduler": "constant_with_warmup",
            "trainer.algorithm.use_kl_loss": False,
            "trainer.algorithm.policy_loss_type": "dual_clip",
        }
    )

    assert cfg.trainer.algorithm.policy_loss_type == "dual_clip"


def test_colocate_all_is_server_set_on_single_role_runtimes():
    """Single-role runtimes never colocate, so colocate_all=True is rejected there."""
    with pytest.raises(ValueError, match=re.escape("trainer.placement.colocate_all=True")):
        _build_skyrl_train_config(
            BASE_MODEL, MegatronBackendOverrides(runtime_role="trainer", **{"trainer.placement.colocate_all": True})
        )
    _build({"trainer.placement.colocate_all": True})


def test_lora_rank_must_match_the_clients():
    """The create_model rank decides LoRA, including rank 0 (full fine-tuning)."""
    with pytest.raises(ValueError, match=re.escape("trainer.policy.model.lora.rank=32")):
        _build({"trainer.policy.model.lora.rank": 32}, _lora(rank=0))
    with pytest.raises(ValueError, match=re.escape("trainer.policy.model.lora.rank=32")):
        _build({"trainer.policy.model.lora.rank": 32}, _lora(rank=8))
    assert _build({"trainer.policy.model.lora.rank": 8}, _lora(rank=8)).trainer.policy.model.lora.rank == 8


def test_lora_rank_is_used_as_given_without_a_client_lora_config():
    """An inference-only runtime builds without a client LoRA config; the rank enables vLLM LoRA."""
    cfg = _build({"trainer.policy.model.lora.rank": 32}, runtime_role="inference")

    assert cfg.trainer.policy.model.lora.rank == 32


def test_backend_config_alpha_takes_precedence():
    """The Tinker SDK cannot send alpha, so backend_config's alpha wins over the API placeholder."""
    cfg = _build({"trainer.policy.model.lora": {"alpha": 128}}, _lora(rank=8, alpha=32))

    assert cfg.trainer.policy.model.lora.alpha == 128
    assert _build({}, _lora(rank=8, alpha=32)).trainer.policy.model.lora.alpha == 32


@pytest.mark.parametrize(
    ("backend_config", "expected"),
    [
        ({"trainer.policy.optimizer_config.lr": 1e-5}, "replaced by AdamParams.learning_rate"),
        ({"trainer.critic": {"optimizer_config": {"lr": 1e-5}}}, "trainer.critic.optimizer_config.lr=1e-05"),
        ({"trainer.policy": {"torch_profiler_config": {"enable": True}}}, "--torch-profiler"),
        ({"trainer.algorithm.policy_loss_type": "gspo"}, "must be 'regular' or 'dual_clip'"),
        ({"use_ray": False}, "use_ray"),
    ],
)
def test_keys_with_no_effect_are_rejected(backend_config, expected):
    """Keys a request replaces, the profiler config, loss types ppo_policy_loss rejects, and use_ray."""
    with pytest.raises(ValueError, match=re.escape(expected)):
        _build(backend_config)


def test_error_lists_every_key():
    """One error names every rejected key, so the launch command is fixed in one pass."""
    with pytest.raises(ValueError) as excinfo:
        _build({"trainer.strategy": "megatron", "trainer.policy.optimizer_config.lr": 1e-5})

    assert "trainer.strategy='megatron'" in str(excinfo.value)
    assert "trainer.policy.optimizer_config.lr=1e-05" in str(excinfo.value)


def test_values_replaced_outside_the_server_table_are_rejected():
    """A value that does not hold after the build is an error even if _server_values does not list it.

    ``log_example_interval`` is a deprecated name that ``__post_init__`` copies over
    ``print_example_interval``.
    """
    with pytest.raises(ValueError, match=re.escape("trainer.print_example_interval=2: the server uses 5")):
        _build({"trainer.print_example_interval": 2, "trainer.log_example_interval": 5})


def test_a_new_server_write_cannot_go_unnoticed(monkeypatch):
    """Simulates a future write to the built config that nobody added to _server_values."""
    build = SkyRLTrainConfig.from_cli_overrides

    def build_then_override(overrides):
        cfg = build(overrides)
        cfg.trainer.micro_train_batch_size_per_gpu = 99
        return cfg

    monkeypatch.setattr(SkyRLTrainConfig, "from_cli_overrides", staticmethod(build_then_override))

    with pytest.raises(ValueError, match=re.escape("trainer.micro_train_batch_size_per_gpu=4: the server uses 99")):
        _build({"trainer.micro_train_batch_size_per_gpu": 4})


def test_values_in_another_json_type_are_not_reported_as_changes():
    """A string for an int field or a list for a list field still counts as applied."""
    cfg = _build(
        {"trainer.micro_train_batch_size_per_gpu": "4", "trainer.policy.optimizer_config.adam_betas": [0.9, 0.95]}
    )

    assert same_config_value("4", cfg.trainer.micro_train_batch_size_per_gpu)
    assert list(cfg.trainer.policy.optimizer_config.adam_betas) == [0.9, 0.95]


def test_build_config_merges_nested_spelling_like_dot_keys():
    """``flatten_backend_config`` relies on a nested mapping meaning the same as its dotted keys."""
    cfg = _build({"trainer.placement": {"colocate_all": False, "policy_num_gpus_per_node": 2}})

    assert cfg.trainer.placement.colocate_all is False
    assert cfg.trainer.placement.policy_num_gpus_per_node == 2
    assert cfg.trainer.placement.policy_num_nodes == 1


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
    overrides = MegatronBackendOverrides if backend_config.get("strategy") == "megatron" else FSDPBackendOverrides

    _build_skyrl_train_config(BASE_MODEL, overrides(**backend_config), _lora(rank=8))


def test_server_side_keys_of_the_dapo_example_pass():
    """examples/tinker/dapo/run_tinker_server.sh sets keys the server also sets, to the server's values."""
    backend_config = {
        "trainer.strategy": "megatron",
        "trainer.algorithm.policy_loss_type": "dual_clip",
        "trainer.algorithm.clip_ratio_c": 10.0,
        "trainer.algorithm.loss_reduction": "token_mean_legacy",
        "trainer.algorithm.use_kl_loss": False,
        "trainer.policy.optimizer_config.weight_decay": 0.1,
        "trainer.policy.optimizer_config.max_grad_norm": 1.0,
        "trainer.policy.model.lora.alpha": 128,
    }

    cfg = _build_skyrl_train_config(BASE_MODEL, MegatronBackendOverrides(**backend_config), _lora(rank=128))

    assert cfg.trainer.policy.model.lora.alpha == 128


def test_critic_lora_comes_from_its_create_model_request():
    """The critic's rank must match its request; a backend_config alpha takes precedence, as for the policy."""
    cfg = _build({})
    _apply_critic_lora(cfg, _lora(rank=8, alpha=32), {})
    assert (cfg.trainer.critic.model.lora.rank, cfg.trainer.critic.model.lora.alpha) == (8, 32)

    user_values = flatten_backend_config({"trainer.critic.model.lora": {"alpha": 64}})
    cfg = _build(user_values)
    _apply_critic_lora(cfg, _lora(rank=8, alpha=32), user_values)
    assert cfg.trainer.critic.model.lora.alpha == 64

    with pytest.raises(ValueError, match=re.escape("trainer.critic.model.lora.rank=16")):
        _apply_critic_lora(_build({}), _lora(rank=8), {"trainer.critic.model.lora.rank": 16})


# AdamParams equal to SkyRL-Train's optimizer defaults.
MATCHING_ADAM_PARAMS = types.AdamParams(
    learning_rate=1e-5, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, grad_clip_norm=1.0
)
# The Tinker SDK's AdamParams defaults.
SDK_DEFAULT_ADAM_PARAMS = types.AdamParams(
    learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0, grad_clip_norm=0.0
)


def test_matching_adam_params_report_nothing():
    """Values equal to the optimizer config are applied in effect, so nothing is reported."""
    assert _unapplied_adam_params(MATCHING_ADAM_PARAMS, OptimizerConfig(), "policy") == []


def test_sdk_default_adam_params_report_each_difference():
    """Each differing field names the value in use and the key that sets it."""
    assert _unapplied_adam_params(SDK_DEFAULT_ADAM_PARAMS, OptimizerConfig(), "critic") == [
        "beta2=0.95 (the optimizer uses 0.999, set by trainer.critic.optimizer_config.adam_betas)",
        "eps=1e-12 (the optimizer uses 1e-08, not configurable)",
        "weight_decay=0.0 (the optimizer uses 0.01, set by trainer.critic.optimizer_config.weight_decay)",
        "grad_clip_norm=0.0 (the optimizer uses 1.0, set by trainer.critic.optimizer_config.max_grad_norm)",
    ]


def _backend_with_stub_dispatch() -> SkyRLTrainBackend:
    backend = object.__new__(SkyRLTrainBackend)
    backend.config = FSDPBackendOverrides()
    backend._cfg = SimpleNamespace(trainer=SimpleNamespace(policy=SimpleNamespace(optimizer_config=OptimizerConfig())))
    backend._model_ids_to_role = {"model-a": "policy", "model-b": "policy"}
    backend._adam_params_checked = set()
    backend._dispatch = Mock()
    backend._dispatch.optim_step.return_value = 0.5
    return backend


def test_optim_step_warns_once_per_model(monkeypatch):
    """The warning fires on each model's first optim_step only; the learning rate is still applied."""
    logger = Mock()
    monkeypatch.setattr(skyrl_train_backend, "logger", logger)
    backend = _backend_with_stub_dispatch()
    request = types.OptimStepInput(adam_params=SDK_DEFAULT_ADAM_PARAMS)

    backend.optim_step("model-a", request)
    backend.optim_step("model-a", request)
    assert logger.warning.call_count == 1
    assert "model-a" in logger.warning.call_args.args

    backend.optim_step("model-b", request)
    assert logger.warning.call_count == 2
    backend._dispatch.set_lr.assert_called_with("policy", 1e-4, model_id="model-b")


def test_optim_step_is_quiet_when_adam_params_match(monkeypatch):
    """A client that sends the optimizer's own values sees no warning."""
    logger = Mock()
    monkeypatch.setattr(skyrl_train_backend, "logger", logger)
    backend = _backend_with_stub_dispatch()

    backend.optim_step("model-a", types.OptimStepInput(adam_params=MATCHING_ADAM_PARAMS))

    logger.warning.assert_not_called()
