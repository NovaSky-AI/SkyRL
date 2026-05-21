"""GPU test: callbacks fire end-to-end during an SFT training run.

Runs a 1-step / 1-eval FSDP SFT job on Qwen2.5-0.5B-Instruct with a spy
callback registered, then asserts the full event sequence + per-event
CallbackInput payloads.

uv run --isolated --extra dev --extra fsdp pytest \
    tests/backends/skyrl_train/gpu/gpu_ci/test_sft_callbacks.py -v
"""

from skyrl.train.config.sft_config import (
    SFTConfig,
    SFTPlacementConfig,
    build_skyrl_config_for_sft,
)
from skyrl.train.sft_trainer import SFTTrainer
from skyrl.train.utils.callbacks import (
    CallbackInput,
    TrainingCallback,
    TrainingControl,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class RecorderCallback(TrainingCallback):
    """Spy: records every event with a snapshot of the relevant CallbackInput fields."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def _snap(self, name: str, ci: CallbackInput) -> None:
        self.events.append(
            (
                name,
                {
                    "global_step": ci.global_step,
                    "epoch": ci.epoch,
                    "total_steps": ci.total_steps,
                    "steps_per_epoch": ci.steps_per_epoch,
                    "has_batch": ci.batch is not None,
                    "has_metrics": ci.metrics is not None,
                    "metrics_keys": sorted((ci.metrics or {}).keys()),
                    "has_logs": ci.logs is not None,
                    "logs_keys": sorted((ci.logs or {}).keys()),
                    "ckpt_path": ci.ckpt_path,
                },
            )
        )

    def on_train_start(self, trainer, ci: CallbackInput, control: TrainingControl) -> None:
        self._snap("on_train_start", ci)

    def on_train_end(self, trainer, ci: CallbackInput, control: TrainingControl) -> None:
        self._snap("on_train_end", ci)

    def on_epoch_start(self, trainer, ci: CallbackInput, control: TrainingControl) -> None:
        self._snap("on_epoch_start", ci)

    def on_epoch_end(self, trainer, ci: CallbackInput, control: TrainingControl) -> None:
        self._snap("on_epoch_end", ci)

    def on_step_start(self, trainer, ci: CallbackInput, control: TrainingControl) -> None:
        self._snap("on_step_start", ci)

    def on_step_end(self, trainer, ci: CallbackInput, control: TrainingControl) -> None:
        self._snap("on_step_end", ci)

    def on_eval_start(self, trainer, ci: CallbackInput, control: TrainingControl) -> None:
        self._snap("on_eval_start", ci)

    def on_eval_end(self, trainer, ci: CallbackInput, control: TrainingControl) -> None:
        self._snap("on_eval_end", ci)

    def on_save(self, trainer, ci: CallbackInput, control: TrainingControl) -> None:
        self._snap("on_save", ci)

    def on_log(self, trainer, ci: CallbackInput, control: TrainingControl) -> None:
        self._snap("on_log", ci)


def _build_test_sft_config() -> SFTConfig:
    cfg = SFTConfig()
    cfg.strategy = "fsdp"
    cfg.model.path = MODEL_NAME
    cfg.placement = SFTPlacementConfig(num_nodes=1, num_gpus_per_node=1)
    # dataset_name / eval_dataset_name are unused — _load_and_tokenize is
    # monkeypatched. eval_dataset_name must be non-empty so load_eval_dataset
    # actually invokes _load_and_tokenize (rather than returning None).
    cfg.dataset_name = "unused-monkeypatched"
    cfg.dataset_split = "train"
    cfg.eval_dataset_name = "unused-monkeypatched"
    cfg.eval_dataset_split = "train"
    cfg.eval_interval = 1
    cfg.eval_before_train = False
    cfg.num_steps = 1
    cfg.num_epochs = None
    cfg.batch_size = 1
    cfg.micro_train_batch_size_per_gpu = 1
    cfg.max_length = 16
    cfg.use_sample_packing = False
    cfg.logger = "console"
    cfg.ckpt_path = ""
    cfg.hf_save_interval = 0
    return cfg


def _dummy_tokenized() -> list[dict]:
    """One synthetic example: 10 input tokens, 4 response tokens.

    Token ids are small ints within Qwen2.5's vocab range; the trainer
    treats them as opaque embeddings. The shape matches what
    `_tokenize_chat_last_assistant` produces.
    """
    return [
        {
            "input_ids": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "attention_mask": [1] * 10,
            "num_actions": 4,
            "loss_mask": [1, 1, 1, 1],
        }
    ]


def test_callbacks_fire_during_sft_training(ray_init_fixture, monkeypatch):
    """A 1-step SFT run dispatches every relevant callback with the right payload."""
    cfg = _build_test_sft_config()
    skyrl_cfg = build_skyrl_config_for_sft(cfg)

    recorder = RecorderCallback()
    trainer = SFTTrainer(cfg, skyrl_cfg=skyrl_cfg, callbacks=[recorder])

    # Bypass HF network fetch + tokenization: both load_dataset() and
    # load_eval_dataset() funnel through _load_and_tokenize.
    monkeypatch.setattr(trainer, "_load_and_tokenize", lambda *_args, **_kw: _dummy_tokenized())

    try:
        trainer.setup()
        trainer.train()
    finally:
        trainer.shutdown()

    event_names = [name for name, _ in recorder.events]

    expected = [
        "on_train_start",
        "on_epoch_start",
        "on_step_start",
        "on_step_end",
        "on_eval_start",
        "on_eval_end",
        "on_log",
        "on_epoch_end",
        "on_train_end",
    ]
    assert event_names == expected, f"unexpected event sequence: {event_names}"

    # Index snapshots for content checks. (Each event fires exactly once here.)
    snap_by_event = {name: snap for name, snap in recorder.events}

    # on_step_end carries batch + step metrics
    step_end = snap_by_event["on_step_end"]
    assert step_end["has_batch"], "on_step_end should see the training batch"
    assert step_end["has_metrics"], "on_step_end should see step metrics"
    assert "loss" in step_end["metrics_keys"], step_end["metrics_keys"]

    # on_eval_end carries eval metrics
    eval_end = snap_by_event["on_eval_end"]
    assert eval_end["has_metrics"], "on_eval_end should see eval metrics"
    assert "eval_loss" in eval_end["metrics_keys"], eval_end["metrics_keys"]

    # on_log carries the dict about to be committed to the tracker
    log_snap = snap_by_event["on_log"]
    assert log_snap["has_logs"], "on_log should see the log dict"
    log_keys = log_snap["logs_keys"]
    assert "train/loss" in log_keys, log_keys
    assert "eval/eval_loss" in log_keys, log_keys

    # No checkpoint configured -> on_save must not fire
    assert "on_save" not in event_names

    # Loop metadata is consistent across every event
    for name, snap in recorder.events:
        assert snap["total_steps"] == 1, f"{name}: total_steps={snap['total_steps']}"
        assert snap["steps_per_epoch"] == 1, f"{name}: steps_per_epoch={snap['steps_per_epoch']}"
