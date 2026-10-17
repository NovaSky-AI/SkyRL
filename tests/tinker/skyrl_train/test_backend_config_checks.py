"""CPU tests for the SkyRL-Train backend's backend_config check and its AdamParams warning."""

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
from skyrl.train.config import OptimizerConfig  # noqa: E402

# AdamParams equal to SkyRL-Train's optimizer defaults.
MATCHING_ADAM_PARAMS = types.AdamParams(
    learning_rate=1e-5, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, grad_clip_norm=1.0
)
# The Tinker SDK's AdamParams defaults.
SDK_DEFAULT_ADAM_PARAMS = types.AdamParams(
    learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0, grad_clip_norm=0.0
)


def test_build_config_rejects_server_controlled_keys():
    """Callers that build the backend directly get the same check as the Tinker server."""
    overrides = FSDPBackendOverrides(**{"trainer.policy": {"optimizer_config": {"lr": 1e-5}}})

    with pytest.raises(ValueError, match="trainer.policy.optimizer_config.lr"):
        _build_skyrl_train_config("Qwen/Qwen3-0.6B", overrides)


def test_build_config_merges_nested_spelling_like_dot_keys():
    """``flatten_backend_config`` relies on a nested mapping meaning the same as its dotted keys."""
    overrides = FSDPBackendOverrides(**{"trainer.placement": {"colocate_all": False, "policy_num_gpus_per_node": 2}})

    cfg = _build_skyrl_train_config("Qwen/Qwen3-0.6B", overrides)

    assert cfg.trainer.placement.colocate_all is False
    assert cfg.trainer.placement.policy_num_gpus_per_node == 2
    assert cfg.trainer.placement.policy_num_nodes == 1


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
