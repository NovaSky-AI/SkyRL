"""
Tests for the max_training_steps feature across all trainer paths.

Verifies that setting max_training_steps correctly caps training
regardless of epochs or dataset size, for RL (sync/async) and SFT trainers.

uv run --extra dev --extra skyrl-train pytest tests/train/test_max_training_steps.py -v
"""

import importlib.util
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl.train.config.config import SkyRLTrainConfig, TrainerConfig
from skyrl.train.config.sft_config import SFTConfig, validate_sft_cfg

# transformers is a core skyrl-train dependency on Linux (CI), but a uv
# override-dependency drops it on non-Linux platforms. The tests below that
# construct the full RL config / SFT trainer import it transitively, so skip
# them when it is genuinely absent (e.g. local macOS) rather than faking it.
requires_transformers = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers not installed (dropped by uv override on non-Linux)",
)


# ---------------------------------------------------------------------------
# Config-level tests: TrainerConfig (RL)
# ---------------------------------------------------------------------------


class TestTrainerConfigMaxSteps:
    """TrainerConfig.max_training_steps field behavior."""

    def test_default_is_none(self):
        cfg = TrainerConfig()
        assert cfg.max_training_steps is None

    def test_set_via_constructor(self):
        cfg = TrainerConfig(max_training_steps=10)
        assert cfg.max_training_steps == 10

    @requires_transformers
    def test_set_via_from_dict_config(self):
        cfg_dict = OmegaConf.create({"trainer": {"max_training_steps": 5}})
        cfg = SkyRLTrainConfig.from_dict_config(cfg_dict)
        assert cfg.trainer.max_training_steps == 5

    @requires_transformers
    def test_cli_override(self):
        cfg = SkyRLTrainConfig.from_cli_overrides(["trainer.max_training_steps=3"])
        assert cfg.trainer.max_training_steps == 3

    @requires_transformers
    def test_none_preserved_when_unset(self):
        cfg_dict = OmegaConf.create({"trainer": {"epochs": 5}})
        cfg = SkyRLTrainConfig.from_dict_config(cfg_dict)
        assert cfg.trainer.max_training_steps is None


# ---------------------------------------------------------------------------
# Config-level tests: SFTConfig
# ---------------------------------------------------------------------------


class TestSFTConfigMaxSteps:
    """SFTConfig.max_training_steps field behavior."""

    def test_default_is_none(self):
        cfg = SFTConfig()
        assert cfg.max_training_steps is None

    def test_set_via_constructor(self):
        cfg = SFTConfig(max_training_steps=7)
        assert cfg.max_training_steps == 7

    def test_set_via_cli_overrides(self):
        cfg = SFTConfig.from_cli_overrides(["max_training_steps=10"])
        assert cfg.max_training_steps == 10

    def test_set_via_dict_overrides(self):
        cfg = SFTConfig.from_cli_overrides({"max_training_steps": 12})
        assert cfg.max_training_steps == 12

    def test_validation_passes_with_max_training_steps(self):
        cfg = SFTConfig(max_training_steps=3)
        validate_sft_cfg(cfg)

    def test_validation_passes_without_max_training_steps(self):
        cfg = SFTConfig()
        validate_sft_cfg(cfg)

    def test_validation_rejects_zero(self):
        cfg = SFTConfig(max_training_steps=0)
        with pytest.raises(ValueError, match="max_training_steps must be > 0"):
            validate_sft_cfg(cfg)

    def test_validation_rejects_negative(self):
        cfg = SFTConfig(max_training_steps=-1)
        with pytest.raises(ValueError, match="max_training_steps must be > 0"):
            validate_sft_cfg(cfg)


class TestRLValidateCfgMaxSteps:
    """validate_cfg rejects max_training_steps <= 0 for RL configs."""

    def test_rejects_zero(self):
        from skyrl.train.utils.utils import validate_cfg

        cfg = SimpleNamespace(trainer=TrainerConfig(max_training_steps=0))
        with pytest.raises(ValueError, match="max_training_steps must be > 0"):
            validate_cfg(cfg)

    def test_rejects_negative(self):
        from skyrl.train.utils.utils import validate_cfg

        cfg = SimpleNamespace(trainer=TrainerConfig(max_training_steps=-5))
        with pytest.raises(ValueError, match="max_training_steps must be > 0"):
            validate_cfg(cfg)


# ---------------------------------------------------------------------------
# SFT Trainer: num_steps capping logic
# ---------------------------------------------------------------------------


class TestSFTTrainerMaxStepsCapping:
    """Verify the capping logic used in SFTTrainer.train().

    Exercises the real SFTTrainer._resolve_num_steps so the test fails if the
    trainer's resolution logic changes (rather than re-implementing it here).
    """

    @requires_transformers
    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            (dict(num_steps=None, num_epochs=10, steps_per_epoch=25, max_training_steps=3), 3),
            (dict(num_steps=50, num_epochs=1, steps_per_epoch=1, max_training_steps=5), 5),
            (dict(num_steps=50, num_epochs=1, steps_per_epoch=1, max_training_steps=None), 50),
            (dict(num_steps=10, num_epochs=1, steps_per_epoch=1, max_training_steps=1000), 10),
            (dict(num_steps=None, num_epochs=2, steps_per_epoch=25, max_training_steps=None), 50),
            (dict(num_steps=None, num_epochs=100, steps_per_epoch=1000, max_training_steps=1), 1),
        ],
        ids=[
            "caps_epoch_derived",
            "caps_explicit_num_steps",
            "no_cap_when_none",
            "no_effect_larger",
            "epoch_resolution_no_cap",
            "single_step",
        ],
    )
    def test_resolve_num_steps(self, kwargs, expected):
        from skyrl.train.sft_trainer import SFTTrainer

        assert SFTTrainer._resolve_num_steps(**kwargs) == expected

    @requires_transformers
    def test_worker_initialization_uses_capped_num_training_steps(self, monkeypatch):
        """SFTTrainer._init_workers must pass the capped step count to the LR
        scheduler. A lightweight fake actor group captures the value without
        spinning up Ray or real workers.
        """
        from skyrl.train import sft_trainer as sft_module

        captured = {}

        class FakeActorGroup:
            def __init__(self, *args, **kwargs):
                pass

            def async_init_model(self, model_path, num_training_steps=None):
                captured["model_path"] = model_path
                captured["num_training_steps"] = num_training_steps
                return None

            def async_run_ray_method(self, *args, **kwargs):
                return None

        fake_worker_module = SimpleNamespace(PolicyWorker=object)
        monkeypatch.setitem(
            sys.modules,
            "skyrl.backends.skyrl_train.workers.megatron.megatron_worker",
            fake_worker_module,
        )
        monkeypatch.setattr(sft_module, "placement_group", lambda *args, **kwargs: object())
        monkeypatch.setattr(sft_module, "get_ray_pg_ready_with_timeout", lambda *args, **kwargs: None)
        monkeypatch.setattr(sft_module, "ResolvedPlacementGroup", lambda pg: pg)
        monkeypatch.setattr(sft_module, "PPORayActorGroup", FakeActorGroup)
        monkeypatch.setattr(sft_module, "WorkerDispatch", lambda *args, **kwargs: object())
        monkeypatch.setattr(sft_module.ray, "get", lambda result: result)

        trainer = object.__new__(sft_module.SFTTrainer)
        trainer.sft_cfg = SFTConfig(num_steps=1000, max_training_steps=5)
        trainer.sft_cfg.placement.num_gpus_per_node = 1
        trainer.cfg = SimpleNamespace(
            trainer=SimpleNamespace(
                policy=SimpleNamespace(sequence_parallel_size=1, record_memory=False),
            )
        )
        trainer.tokenizer = SimpleNamespace(pad_token_id=0)

        trainer._init_workers()

        assert captured["num_training_steps"] == 5


# ---------------------------------------------------------------------------
# Integration: config round-trips
# ---------------------------------------------------------------------------


class TestConfigRoundTrips:
    """max_training_steps survives config construction paths."""

    @requires_transformers
    def test_rl_config_from_dict_config_roundtrip(self):
        cfg_dict = OmegaConf.create({"trainer": {"max_training_steps": 42, "epochs": 5, "train_batch_size": 8}})
        cfg = SkyRLTrainConfig.from_dict_config(cfg_dict)
        assert cfg.trainer.max_training_steps == 42
        assert cfg.trainer.epochs == 5

    def test_sft_config_from_dict_overrides(self):
        cfg = SFTConfig.from_cli_overrides({"max_training_steps": 7, "num_epochs": 3})
        assert cfg.max_training_steps == 7
        assert cfg.num_epochs == 3

    @requires_transformers
    def test_rl_config_max_steps_coexists_with_epochs(self):
        cfg = SkyRLTrainConfig.from_cli_overrides(["trainer.max_training_steps=10", "trainer.epochs=100"])
        assert cfg.trainer.max_training_steps == 10
        assert cfg.trainer.epochs == 100

    def test_sft_max_steps_coexists_with_num_steps(self):
        cfg = SFTConfig.from_cli_overrides(["max_training_steps=5", "num_steps=100"])
        assert cfg.max_training_steps == 5
        assert cfg.num_steps == 100
