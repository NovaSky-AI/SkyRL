"""A model whose workers fail to initialize releases them before the error reaches create_model's caller.

CPU-only: the worker groups are mocks, no Ray.
"""

from unittest.mock import Mock, patch

import pytest

skyrl_train_backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")

from skyrl.backends.skyrl_train.workers.worker_dispatch import (  # noqa: E402
    WorkerDispatch,
)
from skyrl.backends.skyrl_train_backend import (  # noqa: E402
    MegatronBackendOverrides,
    SkyRLTrainBackend,
    _build_skyrl_train_config,
)
from skyrl.tinker import types  # noqa: E402


def _backend() -> SkyRLTrainBackend:
    backend = object.__new__(SkyRLTrainBackend)
    backend._cfg = _build_skyrl_train_config("Qwen/Qwen3-0.6B", MegatronBackendOverrides())
    backend._colocate_pg = Mock()
    backend._tokenizer = Mock()
    backend._inference_engine_client = None
    backend._dispatch = None
    return backend


def _group_that_fails_to_initialize(dp_size: int = 1) -> Mock:
    group = Mock()
    group.async_init_model.side_effect = RuntimeError("model path not found")
    group.get_dp_size.return_value = dp_size
    return group


def test_policy_workers_are_shut_down_when_the_policy_fails_to_initialize():
    backend = _backend()
    policy = _group_that_fails_to_initialize()

    with patch("skyrl.backends.skyrl_train_backend.PPORayActorGroup", return_value=policy):
        with pytest.raises(RuntimeError, match="model path not found"):
            backend._build_policy(PolicyWorker=Mock(), model_id="model-a")

    policy.shutdown.assert_called_once_with()


def test_critic_workers_are_shut_down_and_unregistered_when_the_critic_fails_to_initialize():
    backend = _backend()
    policy = Mock()
    policy.get_dp_size.return_value = 2
    backend._dispatch = WorkerDispatch(backend._cfg, policy_actor_group=policy)
    critic = _group_that_fails_to_initialize(dp_size=3)

    with patch("skyrl.backends.skyrl_train_backend.PPORayActorGroup", return_value=critic):
        with pytest.raises(RuntimeError, match="model path not found"):
            backend._build_critic(CriticWorker=Mock(), lora_config=types.LoraConfig(rank=0, alpha=16, seed=0))

    critic.shutdown.assert_called_once_with()
    # lcm(2, 3) = 6 if the failed critic were still registered.
    assert backend._dispatch.get_lcm_dp_size() == 2
