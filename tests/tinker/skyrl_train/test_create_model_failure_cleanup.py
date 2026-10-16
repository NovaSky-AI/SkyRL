"""Unit tests for the cleanup after the first policy model fails to build.

With ``colocate_all`` the backend reserves every GPU through a placement group before the policy
workers are built. If anything fails before the model is registered, ``delete_model`` has nothing to
tear down, so ``create_model`` itself must release the reservation and reset its state. Otherwise
every later ``create_model`` waits for GPUs that the failed attempt still holds.

CPU-only: Ray and the worker build are mocked. Run:
  uv run --isolated --extra tinker --extra fsdp --extra dev \\
    pytest tests/tinker/skyrl_train/test_create_model_failure_cleanup.py
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

skyrl_train_backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")

from skyrl.backends.skyrl_train_backend import (  # noqa: E402
    MegatronBackendOverrides,
    SkyRLTrainBackend,
)
from skyrl.tinker import types  # noqa: E402

LORA_CONFIG = types.LoraConfig(rank=8, alpha=16, seed=0)
REMOVE_PG = "skyrl.backends.skyrl_train_backend.ray.util.remove_placement_group"
FSDP_WORKER = "skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker"


def _backend() -> SkyRLTrainBackend:
    """Build a backend before its first model without running __init__."""
    backend = object.__new__(SkyRLTrainBackend)
    backend.base_model = "Qwen/Qwen3-0.6B"
    backend.config = MegatronBackendOverrides()
    backend._model_ids_to_role = {}
    backend._model_metadata = {}
    backend._cfg = None
    backend._dispatch = None
    backend._colocate_pg = None
    backend._base_lora_signature = None
    return backend


def _create_model(backend: SkyRLTrainBackend, colocate_all: bool, pg: Mock, build_policy: Mock) -> None:
    cfg = Mock()
    cfg.trainer.strategy = "fsdp"
    cfg.trainer.placement.colocate_all = colocate_all
    # create_model imports the worker class lazily; the fake keeps the FSDP stack out of this test.
    fsdp_worker = ModuleType(FSDP_WORKER)
    fsdp_worker.PolicyWorker = Mock()
    with (
        patch.dict(sys.modules, {FSDP_WORKER: fsdp_worker}),
        patch("skyrl.backends.skyrl_train_backend._build_skyrl_train_config", return_value=cfg),
        patch("skyrl.backends.skyrl_train_backend.ray.is_initialized", return_value=True),
        patch.object(SkyRLTrainBackend, "_create_colocate_pg", return_value=pg),
        patch.object(SkyRLTrainBackend, "_build_policy", build_policy),
    ):
        backend.create_model("model-a", LORA_CONFIG)


def test_failed_policy_build_releases_colocate_pg():
    backend = _backend()
    pg = Mock()

    with patch(REMOVE_PG) as remove_pg:
        with pytest.raises(RuntimeError, match="out of memory"):
            _create_model(backend, True, pg, Mock(side_effect=RuntimeError("out of memory")))

    remove_pg.assert_called_once_with(pg.pg)
    assert backend._colocate_pg is None
    assert backend._cfg is None
    assert backend._dispatch is None
    assert backend._model_ids_to_role == {}
    assert backend._model_metadata == {}
    assert backend._base_lora_signature is None


def test_create_model_reserves_gpus_again_after_failed_build():
    backend = _backend()
    first_pg, second_pg = Mock(), Mock()

    with patch(REMOVE_PG) as remove_pg:
        with pytest.raises(RuntimeError, match="out of memory"):
            _create_model(backend, True, first_pg, Mock(side_effect=RuntimeError("out of memory")))
        _create_model(backend, True, second_pg, Mock())

    remove_pg.assert_called_once_with(first_pg.pg)
    assert backend._colocate_pg is second_pg
    assert backend._model_ids_to_role == {"model-a": "policy"}


def test_failed_policy_build_without_colocate_pg_resets_state():
    backend = _backend()

    with patch(REMOVE_PG) as remove_pg:
        with pytest.raises(RuntimeError, match="out of memory"):
            _create_model(backend, False, Mock(), Mock(side_effect=RuntimeError("out of memory")))

    remove_pg.assert_not_called()
    assert backend._colocate_pg is None
    assert backend._cfg is None
    assert backend._model_ids_to_role == {}


def test_failed_pg_removal_does_not_mask_the_build_error():
    backend = _backend()

    with patch(REMOVE_PG, side_effect=RuntimeError("gcs unavailable")):
        with pytest.raises(RuntimeError, match="out of memory"):
            _create_model(backend, True, Mock(), Mock(side_effect=RuntimeError("out of memory")))

    assert backend._colocate_pg is None
    assert backend._cfg is None


def test_pg_that_never_becomes_ready_is_removed():
    backend = _backend()
    backend._cfg = SimpleNamespace(
        generator=SimpleNamespace(
            inference_engine=SimpleNamespace(
                num_engines=2, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
            )
        )
    )
    pending_pg = Mock()

    with (
        patch("skyrl.backends.skyrl_train_backend.placement_group", return_value=pending_pg),
        patch(
            "skyrl.backends.skyrl_train_backend.get_ray_pg_ready_with_timeout",
            side_effect=RuntimeError("Failed to create placement group"),
        ),
        patch(REMOVE_PG) as remove_pg,
    ):
        with pytest.raises(RuntimeError, match="Failed to create placement group"):
            backend._create_colocate_pg()

    remove_pg.assert_called_once_with(pending_pg)


def test_failed_removal_of_unready_pg_does_not_mask_the_timeout():
    backend = _backend()
    backend._cfg = SimpleNamespace(
        generator=SimpleNamespace(
            inference_engine=SimpleNamespace(
                num_engines=2, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
            )
        )
    )

    with (
        patch("skyrl.backends.skyrl_train_backend.placement_group", return_value=Mock()),
        patch(
            "skyrl.backends.skyrl_train_backend.get_ray_pg_ready_with_timeout",
            side_effect=RuntimeError("Failed to create placement group"),
        ),
        patch(REMOVE_PG, side_effect=RuntimeError("gcs unavailable")),
        patch("skyrl.backends.skyrl_train_backend.logger") as logger,
    ):
        with pytest.raises(RuntimeError, match="Failed to create placement group"):
            backend._create_colocate_pg()

    logger.warning.assert_called_once()
    assert "gcs unavailable" in logger.warning.call_args.args[0]
