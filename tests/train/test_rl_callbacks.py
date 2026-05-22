"""CPU test: callbacks fire end-to-end during a RayPPOTrainer training run.

Mocks generation, the inference engine, and the GPU-heavy worker methods so
the test only exercises the orchestration in ``RayPPOTrainer.train()`` and
asserts the callback event sequence.

uv run --isolated --extra dev pytest tests/train/test_rl_callbacks.py -v
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.callbacks import (
    CallbackInput,
    TrainingCallback,
)
from tests.train.util import example_dummy_config

# ---------------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------------


class DummyDataset:
    """Single-batch dataset: one iteration -> one training step."""

    def __init__(self, size: int = 2):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return ([{"role": "user", "content": f"q{idx}"}], None)

    def collate_fn(self, batch):
        return batch


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

    def on_train_start(self, trainer, ci, control):
        self._snap("on_train_start", ci)

    def on_train_end(self, trainer, ci, control):
        self._snap("on_train_end", ci)

    def on_epoch_start(self, trainer, ci, control):
        self._snap("on_epoch_start", ci)

    def on_epoch_end(self, trainer, ci, control):
        self._snap("on_epoch_end", ci)

    def on_step_start(self, trainer, ci, control):
        self._snap("on_step_start", ci)

    def on_step_end(self, trainer, ci, control):
        self._snap("on_step_end", ci)

    def on_eval_start(self, trainer, ci, control):
        self._snap("on_eval_start", ci)

    def on_eval_end(self, trainer, ci, control):
        self._snap("on_eval_end", ci)

    def on_save(self, trainer, ci, control):
        self._snap("on_save", ci)

    def on_log(self, trainer, ci, control):
        self._snap("on_log", ci)


def _stub_training_input() -> TrainingInputBatch:
    """Minimal TrainingInputBatch that survives the keys the loop pops post-advantages."""
    batch = TrainingInputBatch(
        {
            "sequences": torch.zeros((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "loss_mask": torch.ones((1, 4), dtype=torch.long),
            "response_mask": torch.ones((1, 4), dtype=torch.long),
            "rewards": torch.zeros((1, 4)),
        }
    )
    # Loop body pops "uids" and (optionally) "is_last_step" after
    # compute_advantages_and_returns; "response_length" / "avg_response_length"
    # are read by downstream metrics in some paths.
    batch.metadata = {
        "uids": ["uid-0"],
        "response_length": 4,
        "avg_response_length": 4.0,
    }
    return batch


def _build_test_cfg():
    cfg = example_dummy_config()
    # 1 epoch over a 1-step dataloader; eval fires once at step 1
    cfg.trainer.epochs = 1
    cfg.trainer.eval_interval = 1
    cfg.trainer.eval_before_train = False
    cfg.trainer.ckpt_interval = 0
    cfg.trainer.hf_save_interval = 0
    cfg.trainer.ckpt_path = ""
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.dump_data_batch = False
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.update_ref_every_epoch = False
    cfg.trainer.algorithm.dynamic_sampling.type = None
    cfg.generator.step_wise_trajectories = False
    cfg.generator.inference_engine.enable_ray_prometheus_stats = False
    return cfg


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_callbacks_fire_during_rl_training(monkeypatch):
    """A 1-step PPO run dispatches every relevant callback with the right payload."""
    cfg = _build_test_cfg()

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2

    tracker = MagicMock()

    recorder = RecorderCallback()
    trainer = RayPPOTrainer(
        cfg=cfg,
        tracker=tracker,
        tokenizer=tokenizer,
        # train_batch_size=2 (from example_dummy_config) -> need >=2 examples to get 1 batch
        train_dataset=DummyDataset(size=2),
        eval_dataset=DummyDataset(size=2),
        inference_engine_client=None,
        generator=MagicMock(),
        callbacks=[recorder],
    )

    # Replace dispatch (normally built by build_models). The loop awaits
    # save_weights_for_sampler() before training and at the end of every step;
    # get_lcm_dp_size is called by _remove_tail_data via the real method.
    dispatch_mock = MagicMock()
    dispatch_mock.save_weights_for_sampler = AsyncMock(return_value=None)
    dispatch_mock.get_lcm_dp_size = MagicMock(return_value=1)
    trainer.dispatch = dispatch_mock

    # Stub out GPU-heavy / generator-touching methods. The train() loop is
    # mostly orchestration over these calls — replacing each one with a
    # passthrough (or async stub) lets us exercise the callback firings
    # without any workers / GPUs / network.
    monkeypatch.setattr(trainer, "init_weight_sync_state", lambda: None)
    monkeypatch.setattr(
        trainer,
        "generate",
        AsyncMock(return_value={"rollout_metrics": None, "response_ids": [[1]], "rewards": [0.0]}),
    )
    monkeypatch.setattr(trainer, "eval", AsyncMock(return_value={"eval/score": 0.5}))

    monkeypatch.setattr(trainer, "postprocess_generator_output", lambda gen_out, uids: (gen_out, uids))
    monkeypatch.setattr(trainer, "convert_to_training_input", lambda *_args, **_kw: _stub_training_input())
    monkeypatch.setattr(trainer, "fwd_logprobs_values_reward", lambda batch: batch)
    monkeypatch.setattr(trainer, "compute_advantages_and_returns", lambda batch: batch)
    monkeypatch.setattr(trainer, "train_critic_and_policy", lambda batch: {"policy_loss": 0.42})

    # prepare_generator_input has a heavy signature; bypass it via a
    # module-level patch so the trainer's call site gets back something
    # benign that the downstream (already-mocked) methods accept.
    monkeypatch.setattr(
        "skyrl.train.trainer.prepare_generator_input",
        lambda *_args, **_kw: ({"prompts": [[{"role": "user", "content": "q"}]]}, ["uid-0"]),
    )

    asyncio.run(trainer.train())

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

    snap_by_event = {name: snap for name, snap in recorder.events}

    # on_step_end carries the (mocked) batch + step metrics
    step_end = snap_by_event["on_step_end"]
    assert step_end["has_batch"], "on_step_end should see the training batch"
    assert step_end["has_metrics"], "on_step_end should see step metrics"
    assert "policy_loss" in step_end["metrics_keys"]

    # on_eval_end carries eval metrics
    eval_end = snap_by_event["on_eval_end"]
    assert eval_end["has_metrics"], "on_eval_end should see eval metrics"
    assert "eval/score" in eval_end["metrics_keys"]

    # on_log carries the log dict the trainer is about to commit
    log_snap = snap_by_event["on_log"]
    assert log_snap["has_logs"], "on_log should see the log dict"
    # the trainer always stamps trainer/epoch + trainer/global_step into log_payload
    assert "trainer/epoch" in log_snap["logs_keys"]
    assert "trainer/global_step" in log_snap["logs_keys"]

    # No checkpoint configured -> on_save must not fire
    assert "on_save" not in event_names

    # Loop metadata stays consistent across every event
    for name, snap in recorder.events:
        assert snap["total_steps"] == 1, f"{name}: total_steps={snap['total_steps']}"
        assert snap["steps_per_epoch"] == 1, f"{name}: steps_per_epoch={snap['steps_per_epoch']}"
