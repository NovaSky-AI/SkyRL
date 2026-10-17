"""Tests for the startup checks on ``--backend-config`` for the SkyRL-Train backends."""

import re

import pytest
from pydantic import ValidationError

from skyrl.tinker.config import EngineConfig, flatten_backend_config

BASE_MODEL = "Qwen/Qwen3-0.6B"


def _engine_config(backend_config: dict, backend: str = "fsdp", **kwargs) -> EngineConfig:
    return EngineConfig(base_model=BASE_MODEL, backend=backend, backend_config=backend_config, **kwargs)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("trainer.policy.optimizer_config.lr", 1e-5),
        ("trainer.critic.optimizer_config.lr", 1e-5),
        ("trainer.policy.model.lora.rank", 32),
        ("use_ray", False),
        ("trainer.policy.torch_profiler_config.enable", True),
    ],
)
def test_keys_the_server_always_replaces_are_rejected(key, value):
    """No value of these keys is ever used, so any value is rejected."""
    with pytest.raises(ValidationError, match=re.escape(key)):
        _engine_config({key: value})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("strategy", "megatron"),
        ("trainer.strategy", "megatron"),
        ("runtime_role", "trainer"),
        ("trainer.policy.model.path", "Qwen/Qwen3-1.7B"),
        ("trainer.critic.model.path", "Qwen/Qwen3-1.7B"),
        ("trainer.policy.optimizer_config.scheduler", "cosine"),
        ("trainer.critic.optimizer_config.num_warmup_steps", 10),
        ("trainer.algorithm.use_kl_loss", True),
        ("trainer.algorithm.policy_loss_type", "gspo"),
    ],
)
def test_values_that_differ_from_the_servers_are_rejected(key, value):
    """A value other than the one the server sets would be silently replaced."""
    with pytest.raises(ValidationError, match=re.escape(f"{key}={value!r}")):
        _engine_config({key: value})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("strategy", "fsdp"),
        ("trainer.strategy", "fsdp"),
        ("runtime_role", "combined"),
        ("trainer.critic.model.path", BASE_MODEL),
        ("trainer.policy.optimizer_config.scheduler", "constant_with_warmup"),
        ("trainer.policy.optimizer_config.num_warmup_steps", 0),
        ("trainer.algorithm.use_kl_loss", False),
        ("trainer.algorithm.policy_loss_type", "regular"),
        ("trainer.algorithm.policy_loss_type", "dual_clip"),
    ],
)
def test_values_equal_to_the_servers_are_accepted(key, value):
    """Redundant keys have the effect the user expects, so they stay allowed."""
    _engine_config({key: value})


def test_nested_spellings_are_checked_as_dot_keys():
    """``from_cli_overrides`` reads a nested mapping the same as the dotted key."""
    with pytest.raises(ValidationError, match=re.escape("trainer.policy.optimizer_config.lr=1e-05")):
        _engine_config({"trainer.policy": {"optimizer_config": {"lr": 1e-5}}})
    with pytest.raises(ValidationError, match=re.escape("trainer.policy.torch_profiler_config.enable")):
        _engine_config({"trainer.policy": {"torch_profiler_config": {"enable": True}}})


def test_error_lists_every_rejected_key():
    """One error names every rejected key, so the launch command is fixed in one pass."""
    with pytest.raises(ValidationError) as excinfo:
        _engine_config({"trainer.strategy": "megatron", "trainer.policy.optimizer_config.lr": 1e-5})
    message = str(excinfo.value)
    assert "trainer.strategy='megatron': set by --backend ('fsdp')" in message
    assert "trainer.policy.optimizer_config.lr=1e-05: replaced by AdamParams.learning_rate" in message


def test_lora_rank_is_accepted_on_an_inference_runtime():
    """An inference-only runtime builds its config without a client LoRA config, so the rank enables vLLM LoRA."""
    _engine_config({"trainer.policy.model.lora.rank": 32}, runtime_role="inference")


@pytest.mark.parametrize("runtime_role", ["trainer", "inference"])
def test_colocate_all_true_is_rejected_on_a_single_role_runtime(runtime_role):
    """Single-role runtimes force colocate_all=False."""
    with pytest.raises(ValidationError, match=re.escape("trainer.placement.colocate_all=True")):
        _engine_config({"trainer.placement.colocate_all": True}, runtime_role=runtime_role)
    _engine_config({"trainer.placement.colocate_all": False}, runtime_role=runtime_role)


def test_colocate_all_is_free_on_a_combined_runtime():
    """The combined runtime uses colocate_all as given."""
    _engine_config({"trainer.placement.colocate_all": True})
    _engine_config({"trainer.placement.colocate_all": False})


def test_checkpoint_path_sets_model_path_and_served_model_name():
    """With --base-model-checkpoint-path, model paths follow it and the served name follows --base-model."""
    checkpoint = "/models/custom"
    _engine_config(
        {
            "trainer.policy.model.path": checkpoint,
            "generator.inference_engine.served_model_name": BASE_MODEL,
        },
        base_model_checkpoint_path=checkpoint,
    )
    with pytest.raises(ValidationError, match=re.escape("generator.inference_engine.served_model_name='custom'")):
        _engine_config(
            {"generator.inference_engine.served_model_name": "custom"},
            base_model_checkpoint_path=checkpoint,
        )
    _engine_config({"generator.inference_engine.served_model_name": "custom"})


def test_jax_backend_is_not_checked():
    """The table describes the SkyRL-Train backends only."""
    _engine_config({"trainer.policy.optimizer_config.lr": 1e-5, "use_ray": True}, backend="jax")


def test_server_settings_of_the_example_scripts_are_accepted():
    """Keys from examples/tinker/{dapo,ppo}/run_tinker_server.sh that the server also sets."""
    _engine_config(
        {
            "trainer.strategy": "megatron",
            "trainer.algorithm.policy_loss_type": "dual_clip",
            "trainer.algorithm.use_kl_loss": False,
            "trainer.policy.optimizer_config.weight_decay": 0.1,
            "trainer.policy.optimizer_config.max_grad_norm": 1.0,
            "trainer.policy.model.lora.alpha": 128,
        },
        backend="megatron",
    )
    _engine_config({"trainer.critic.model.path": BASE_MODEL, "trainer.placement.colocate_all": True})


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
