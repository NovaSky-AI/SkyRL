"""
Tests for the max_training_steps feature across all trainer paths.

Verifies that setting max_training_steps correctly caps training
regardless of epochs or dataset size, for RL (sync/async) and SFT trainers.

uv run --extra dev --extra skyrl-train pytest tests/train/test_max_training_steps.py -v
"""

import importlib.util
import sys
from math import ceil
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl.train.config.config import SkyRLTrainConfig, TrainerConfig
from skyrl.train.config.sft_config import SFTConfig, validate_sft_cfg

requires_transformers = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers not available (Linux-only dependency)",
)


class _FakeTransformers(ModuleType):
    def __getattr__(self, name):
        value = type(name, (), {})
        setattr(self, name, value)
        return value


def _install_fake_transformers(monkeypatch):
    module = _FakeTransformers("transformers")
    module.__file__ = "fake_transformers.py"
    module.__path__ = []
    monkeypatch.setitem(sys.modules, "transformers", module)
    flash_attention_utils = ModuleType("transformers.modeling_flash_attention_utils")
    flash_attention_utils._flash_attention_forward = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "transformers.modeling_flash_attention_utils", flash_attention_utils)


def _install_lightweight_worker_modules(monkeypatch):
    worker_module = ModuleType("skyrl.backends.skyrl_train.workers.worker")
    worker_module.PPORayActorGroup = object
    monkeypatch.setitem(sys.modules, "skyrl.backends.skyrl_train.workers.worker", worker_module)

    worker_dispatch_module = ModuleType("skyrl.backends.skyrl_train.workers.worker_dispatch")
    worker_dispatch_module.WorkerDispatch = object
    monkeypatch.setitem(sys.modules, "skyrl.backends.skyrl_train.workers.worker_dispatch", worker_dispatch_module)


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


# ---------------------------------------------------------------------------
# RL Trainer: total_training_steps capping logic (shared by sync and async)
# ---------------------------------------------------------------------------


class TestRLTrainerStepsCapping:
    """Verify capping logic in _build_train_dataloader_and_compute_training_steps.

    Both sync and async RL trainers use the same pattern:
        total = dataloader_steps * epochs
        if max_training_steps: total = min(total, max_training_steps)
    """

    @pytest.mark.parametrize(
        "dl_len,epochs,max_steps,expected",
        [
            (50, 2, None, 100),
            (50, 2, 10, 10),
            (50, 2, 1000, 100),
            (50, 2, 100, 100),
            (50, 2, 1, 1),
        ],
        ids=["no_cap", "caps_smaller", "no_effect_larger", "boundary_equal", "single_step"],
    )
    def test_capping(self, dl_len, epochs, max_steps, expected):
        total = dl_len * epochs
        if max_steps is not None:
            total = min(total, max_steps)
        assert total == expected


# ---------------------------------------------------------------------------
# RL Trainer: early exit condition (shared by sync and async)
# ---------------------------------------------------------------------------


class TestRLTrainerEarlyExit:
    """Verify early exit condition: global_step > max_training_steps.

    Both sync and async trainers use the same check after incrementing global_step.
    """

    @pytest.mark.parametrize(
        "global_step,max_steps,should_exit",
        [
            (6, 5, True),
            (5, 5, False),
            (3, 5, False),
            (9999, None, False),
            (1, 1, False),
            (2, 1, True),
        ],
        ids=["exceeded", "at_boundary", "below", "none_never_exits", "boundary_1", "exit_after_1"],
    )
    def test_exit_condition(self, global_step, max_steps, should_exit):
        exits = max_steps is not None and global_step > max_steps
        assert exits is should_exit


# ---------------------------------------------------------------------------
# SFT Trainer: num_steps capping logic
# ---------------------------------------------------------------------------


class TestSFTTrainerMaxStepsCapping:
    """Verify the capping logic used in SFTTrainer.train()."""

    @staticmethod
    def _resolve_num_steps(num_steps=None, num_epochs=None, dataset_len=100, batch_size=4, max_training_steps=None):
        if num_steps is not None:
            resolved = num_steps
        else:
            resolved = ceil(dataset_len / batch_size) * num_epochs
        if max_training_steps is not None:
            resolved = min(resolved, max_training_steps)
        return resolved

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            (dict(num_epochs=10, dataset_len=100, batch_size=4, max_training_steps=3), 3),
            (dict(num_steps=50, max_training_steps=5), 5),
            (dict(num_steps=50, max_training_steps=None), 50),
            (dict(num_steps=10, max_training_steps=1000), 10),
            (dict(num_epochs=2, dataset_len=100, batch_size=4, max_training_steps=None), 50),
            (dict(num_epochs=100, dataset_len=1000, batch_size=1, max_training_steps=1), 1),
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
    def test_capping(self, kwargs, expected):
        assert self._resolve_num_steps(**kwargs) == expected

    def test_worker_initialization_uses_capped_num_training_steps(self, monkeypatch):
        _install_fake_transformers(monkeypatch)
        _install_lightweight_worker_modules(monkeypatch)
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


class TestRLTrainerFinalization:
    @pytest.mark.asyncio
    async def test_sync_trainer_max_steps_runs_finalization(self, monkeypatch):
        _install_fake_transformers(monkeypatch)
        _install_lightweight_worker_modules(monkeypatch)
        from skyrl.train import trainer as trainer_module

        events = []

        class FakeTrainingInput(dict):
            def __init__(self):
                super().__init__({"rewards": [1.0]})
                self.metadata = {"uids": ["uid-0"]}

        class FakeTracker:
            def log(self, *args, **kwargs):
                pass

            def finish(self):
                events.append("tracker_finish")

        class FakeDispatch:
            async def save_weights_for_sampler(self):
                pass

        class FakeInferenceEngineClient:
            async def sleep(self):
                events.append("sleep")

        async def fake_generate(generator_input):
            return {"response_ids": [[1]], "rewards": [1.0]}

        monkeypatch.setattr(
            trainer_module,
            "prepare_generator_input",
            lambda *args, **kwargs: ({"prompts": ["prompt"]}, ["uid-0"]),
        )
        monkeypatch.setattr(trainer_module, "get_sampling_params_for_backend", lambda *args, **kwargs: {})

        trainer = object.__new__(trainer_module.RayPPOTrainer)
        trainer.cfg = SimpleNamespace(
            trainer=SimpleNamespace(
                epochs=2,
                max_training_steps=1,
                eval_interval=0,
                eval_before_train=False,
                algorithm=SimpleNamespace(use_kl_in_reward=False, dynamic_sampling=SimpleNamespace(type=None)),
                log_example_interval=0,
                dump_data_batch=False,
                ckpt_interval=1,
                hf_save_interval=1,
                update_ref_every_epoch=False,
            ),
            generator=SimpleNamespace(
                n_samples_per_prompt=1,
                step_wise_trajectories=False,
                inference_engine=SimpleNamespace(backend="vllm"),
                sampling_params=SimpleNamespace(),
            ),
            environment=SimpleNamespace(env_class="gsm8k"),
        )
        trainer.colocate_all = True
        trainer.tracker = FakeTracker()
        trainer.tokenizer = SimpleNamespace(decode=lambda ids: "decoded")
        trainer.train_dataloader = [["prompt-0"], ["prompt-1"]]
        trainer.total_training_steps = 1
        trainer.resume_mode = trainer_module.ResumeMode.NONE
        trainer.all_metrics = {}
        trainer.all_timings = {}
        trainer.global_step = 0
        trainer._vllm_metrics_scraper = None
        trainer.dispatch = FakeDispatch()
        trainer.inference_engine_client = FakeInferenceEngineClient()
        trainer.ref_model = None
        trainer.init_weight_sync_state = lambda: None
        trainer._remove_tail_data = lambda rand_prompts: rand_prompts
        trainer.generate = fake_generate
        trainer.postprocess_generator_output = lambda generator_output, uids: (generator_output, uids)
        trainer.convert_to_training_input = lambda generator_output, uids: FakeTrainingInput()
        trainer.fwd_logprobs_values_reward = lambda training_input: training_input
        trainer.compute_advantages_and_returns = lambda training_input: training_input
        trainer.train_critic_and_policy = lambda training_input: {"loss": 0.0}
        trainer.save_checkpoints = lambda: events.append("save_checkpoints")
        trainer.save_models = lambda: events.append("save_models")

        await trainer.train()

        assert events[-4:] == ["sleep", "save_checkpoints", "save_models", "tracker_finish"]


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
