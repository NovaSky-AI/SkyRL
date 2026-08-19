from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.fireworks.sft import FireworksSFTDispatch
from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import SFTConfig, build_skyrl_config_for_sft
from skyrl.train.entrypoints.main_fireworks_sft import run
from skyrl.train.fireworks_sft_trainer import FireworksSFTTrainer
from skyrl.train.sft_trainer import SFTTrainer


class _Runtime:
    def __init__(self):
        self.training_client = SimpleNamespace()
        self.closed = 0

    async def close(self):
        self.closed += 1


class _Tracker:
    def __init__(self):
        self.finished = False

    def finish(self):
        self.finished = True


def _cfg() -> SFTConfig:
    cfg = SFTConfig(
        strategy="fireworks",
        max_length=512,
        remove_microbatch_padding=False,
        enable_ray_gpu_monitor=False,
    )
    cfg.model.path = "Qwen/Qwen3-4B"
    cfg.model.lora.rank = 8
    cfg.fireworks.base_model = "accounts/fireworks/models/qwen3-4b"
    cfg.fireworks.training_shape_id = "accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora"
    cfg.fireworks.trainer_job_id = "skyrl-smoke-sft-trainer"
    cfg.fireworks.max_seq_len = 32768
    return cfg


def _trainer() -> FireworksSFTTrainer:
    cfg = _cfg()
    trainer = FireworksSFTTrainer(cfg, skyrl_cfg=build_skyrl_config_for_sft(cfg))
    trainer.tokenizer = SimpleNamespace()
    return trainer


def test_fireworks_trainer_reuses_native_train_and_eval_loops() -> None:
    assert FireworksSFTTrainer.train is SFTTrainer.train
    assert FireworksSFTTrainer.train_step is SFTTrainer.train_step
    assert FireworksSFTTrainer.run_eval is SFTTrainer.run_eval


def test_native_train_step_uses_hosted_dispatch() -> None:
    trainer = _trainer()
    calls = []

    class Dispatch:
        def forward_backward(self, model, batch, loss_fn):
            calls.append(("forward_backward", model, loss_fn))
            return WorkerOutput("scalar", [], {"final_loss": 1.25})

        def optim_step(self, model):
            calls.append(("optim_step", model))
            return 0.75

    trainer.dispatch = Dispatch()
    result = trainer.train_step(TrainingInputBatch({"sequences": torch.tensor([[1, 2]])}), step=1)

    assert calls == [
        ("forward_backward", "policy", "cross_entropy"),
        ("optim_step", "policy"),
    ]
    assert result["loss"] == pytest.approx(1.25)
    assert result["grad_norm"] == pytest.approx(0.75)


def test_tracker_initialization_is_deferred() -> None:
    trainer = _trainer()

    trainer._init_tracker()

    assert trainer.tracker is None


def test_backend_setup_creates_no_local_workers(monkeypatch) -> None:
    trainer = _trainer()
    runtime = _Runtime()
    captured = {}

    def connect(**kwargs):
        captured.update(kwargs)
        return runtime

    def init_tracker(self):
        self.tracker = _Tracker()

    monkeypatch.setattr("skyrl.train.fireworks_sft_trainer.FireworksRuntime.connect", connect)
    monkeypatch.setattr(SFTTrainer, "_init_tracker", init_tracker)

    trainer._init_workers()

    assert captured["create_deployment"] is False
    assert isinstance(trainer.dispatch, FireworksSFTDispatch)
    assert isinstance(trainer.tracker, _Tracker)
    trainer.shutdown()
    assert runtime.closed == 1


def test_tracker_failure_closes_runtime(monkeypatch) -> None:
    trainer = _trainer()
    runtime = _Runtime()
    monkeypatch.setattr(
        "skyrl.train.fireworks_sft_trainer.FireworksRuntime.connect",
        lambda **kwargs: runtime,
    )

    def fail_tracker(self):
        raise RuntimeError("tracker failed")

    monkeypatch.setattr(SFTTrainer, "_init_tracker", fail_tracker)

    with pytest.raises(RuntimeError, match="tracker failed"):
        trainer._init_workers()

    assert runtime.closed == 1
    assert trainer._fireworks_runtime is None


def test_callback_checkpoint_request_fails_before_provider_save() -> None:
    trainer = _trainer()

    with pytest.raises(NotImplementedError, match="persistent checkpoints"):
        trainer.save_checkpoint()


def test_shutdown_is_idempotent() -> None:
    trainer = _trainer()
    runtime = _Runtime()
    tracker = _Tracker()
    trainer._fireworks_runtime = runtime
    trainer.tracker = tracker

    trainer.shutdown()
    trainer.shutdown()

    assert tracker.finished is True
    assert runtime.closed == 1


def test_entrypoint_run_uses_hosted_trainer_lifecycle(monkeypatch) -> None:
    events = []

    class Trainer:
        tracker = None
        global_step = 0

        def __init__(self, cfg, skyrl_cfg):
            events.append("init")

        def setup(self):
            events.append("setup")

        def train(self):
            events.append("train")

        def shutdown(self):
            events.append("shutdown")

    monkeypatch.setattr("skyrl.train.entrypoints.main_fireworks_sft.FireworksSFTTrainer", Trainer)

    run(_cfg())

    assert events == ["init", "setup", "train", "shutdown"]


def test_entrypoint_interrupts_without_error_logging(monkeypatch) -> None:
    events = []

    class Trainer:
        tracker = SimpleNamespace(log_exception=lambda *args, **kwargs: events.append("logged"))
        global_step = 0

        def __init__(self, cfg, skyrl_cfg):
            pass

        def setup(self):
            events.append("setup")

        def train(self):
            raise KeyboardInterrupt

        def shutdown(self):
            events.append("shutdown")

    monkeypatch.setattr("skyrl.train.entrypoints.main_fireworks_sft.FireworksSFTTrainer", Trainer)

    with pytest.raises(KeyboardInterrupt):
        run(_cfg())

    assert events == ["setup", "shutdown"]
