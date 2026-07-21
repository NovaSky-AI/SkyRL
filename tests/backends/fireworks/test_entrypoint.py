import pytest

from skyrl.backends.fireworks.runtime import FireworksRuntime
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.entrypoints.main_fireworks import (
    FullyAsyncFireworksExp,
    experiment_class,
)


def test_fireworks_provider_preflight_runs_before_tracker(
    monkeypatch, tmp_path
) -> None:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "fireworks"
    cfg.trainer.fireworks.base_model = "accounts/fireworks/models/test"
    cfg.trainer.fireworks.training_shape_id = "accounts/fireworks/trainingShapes/test"
    cfg.trainer.fireworks.trainer_job_id = "skyrl-smoke-test-trainer"
    cfg.trainer.fireworks.deployment_id = "skyrl-smoke-test-rollout"
    cfg.trainer.policy.model.lora.rank = 8
    cfg.trainer.export_path = str(tmp_path / "exports")
    cfg.trainer.ckpt_path = str(tmp_path / "checkpoints")

    exp = object.__new__(BasePPOExp)
    exp.cfg = cfg
    exp.tokenizer = object()
    exp._fireworks_runtime = None

    tracker_started = False

    def _start_tracker():
        nonlocal tracker_started
        tracker_started = True
        raise AssertionError("tracker must not start before provider preflight")

    def _reject_provider(**_kwargs):
        raise RuntimeError("dedicated trainer provisioning failed")

    monkeypatch.setattr(exp, "get_tracker", _start_tracker)
    monkeypatch.setattr(FireworksRuntime, "connect", staticmethod(_reject_provider))

    with pytest.raises(RuntimeError, match="dedicated trainer provisioning"):
        exp._setup_trainer()

    assert not tracker_started


def test_direct_entrypoint_selects_fully_async_scheduler() -> None:
    cfg = SkyRLTrainConfig()
    assert experiment_class(cfg) is BasePPOExp

    cfg.trainer.fully_async.enabled = True
    assert experiment_class(cfg) is FullyAsyncFireworksExp
