"""A model whose workers fail to initialize releases them before the error reaches create_model's caller.

CPU-only: the worker groups are mocks, no Ray.
"""

import sys
from types import SimpleNamespace
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
from skyrl.train.utils.utils import ResolvedPlacementGroup  # noqa: E402


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


def test_a_failed_first_create_model_leaves_no_runtime_behind():
    backend = object.__new__(SkyRLTrainBackend)
    backend.base_model = "Qwen/Qwen3-0.6B"
    backend.config = MegatronBackendOverrides()
    backend._model_ids_to_role = {}
    backend._cfg = None
    backend._dispatch = None
    backend._colocate_pg = None
    backend._base_lora_signature = None
    backend._inference_engines_initialized = False
    colocate_pg = Mock()
    backend._create_colocate_pg = Mock(return_value=ResolvedPlacementGroup(colocate_pg))
    backend._build_policy = Mock(side_effect=RuntimeError("model path not found"))
    # Megatron is not installed in the CPU tests.
    megatron_worker = SimpleNamespace(PolicyWorker=Mock())

    with (
        patch("skyrl.backends.skyrl_train_backend.ray.is_initialized", return_value=True),
        patch("skyrl.backends.skyrl_train_backend.remove_placement_group") as remove_placement_group,
        patch.dict(sys.modules, {"skyrl.backends.skyrl_train.workers.megatron.megatron_worker": megatron_worker}),
    ):
        with pytest.raises(RuntimeError, match="model path not found"):
            backend.create_model("model-a", types.LoraConfig(rank=8, alpha=16, seed=0))

    remove_placement_group.assert_called_once_with(colocate_pg)
    # Sampling answers as it did before the failed call.
    with pytest.raises(RuntimeError, match="Create a model before sampling"):
        backend._ensure_inference_engines()
