"""CPU test: callbacks fire end-to-end during a RayPPOTrainer training run.

Runs 2 training steps (1 epoch over a 4-example dummy dataset, train_batch_size=2)
with three callbacks registered:
  * RecorderCallback — snapshots every event for sequence + payload assertions.
  * ForceEvaluateAtStep — sets ``control.should_evaluate = True`` on the
    on_step_end of step 1, exercising the callback-driven eval path. With
    ``eval_interval=2`` step 1 would not normally eval, so any eval event
    seen at step 1 comes from this callback.
  * ForceSaveAtStep — sets ``control.should_save = True`` on the on_step_end
    of step 2, exercising the callback-driven save path.

Mocks generation, the inference engine, and the GPU-heavy worker methods so
the test only exercises the orchestration in ``RayPPOTrainer.train()``.

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


class ForceSaveAtStep(TrainingCallback):
    """Sets ``control.should_save = True`` on on_step_end when the global step matches."""

    def __init__(self, step: int):
        self.step = step

    def on_step_end(self, trainer, ci, control):
        if ci.global_step == self.step:
            control.should_save = True


class ForceEvaluateAtStep(TrainingCallback):
    """Sets ``control.should_evaluate = True`` on on_step_end when the global step matches."""

    def __init__(self, step: int):
        self.step = step

    def on_step_end(self, trainer, ci, control):
        if ci.global_step == self.step:
            control.should_evaluate = True


_FAKE_CKPT_PATH = "/fake/rl-callback-test/global_step_2"


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
    # 1 epoch over a 2-batch dataloader -> 2 steps total.
    cfg.trainer.epochs = 1
    # eval_interval=2 means step 1 has no interval-driven eval; only the force-evaluate
    # callback can trigger eval at step 1. Step 2 still gets an interval-driven eval.
    cfg.trainer.eval_interval = 2
    cfg.trainer.eval_before_train = False
    # ckpt_interval=0 means the callback-driven force-save is the only save path tested.
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


# TODO (sumanthrh): See if heavy mocking in this test can be reduced
# Currently there isn't a better option due to the number of worker-related training methods
def test_callbacks_fire_during_rl_training(monkeypatch):
    """A 2-step PPO run fires every relevant event, in order, with the right payloads."""
    cfg = _build_test_cfg()

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2

    tracker = MagicMock()

    recorder = RecorderCallback()
    force_eval = ForceEvaluateAtStep(step=1)
    force_save = ForceSaveAtStep(step=2)
    trainer = RayPPOTrainer(
        cfg=cfg,
        tracker=tracker,
        tokenizer=tokenizer,
        # train_batch_size=2 (from example_dummy_config) -> 4 examples gives 2 batches = 2 steps.
        train_dataset=DummyDataset(size=4),
        eval_dataset=DummyDataset(size=2),
        inference_engine_client=None,
        generator=MagicMock(),
        callbacks=[recorder, force_eval, force_save],
    )

    # Replace dispatch (normally built by build_models). The loop awaits
    # save_weights_for_sampler() before training and at the end of every step;
    # get_lcm_dp_size is called by _remove_tail_data via the real method.
    dispatch_mock = MagicMock()
    dispatch_mock.save_weights_for_sampler = AsyncMock(return_value=None)
    dispatch_mock.get_lcm_dp_size = MagicMock(return_value=1)
    trainer.dispatch = dispatch_mock

    # Stub out GPU-heavy / generator-touching methods. The train() loop is
    # mostly orchestration over these calls
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

    # Stub save_checkpoints so the callback-driven save doesn't touch disk.
    # on_save still receives the fake path the stub returns.
    monkeypatch.setattr(trainer, "save_checkpoints", lambda: _FAKE_CKPT_PATH)

    # prepare_generator_input has a heavy signature; bypass it via a
    # module-level patch so the trainer's call site gets back something
    # benign that the downstream (already-mocked) methods accept.
    monkeypatch.setattr(
        "skyrl.train.trainer.prepare_generator_input",
        lambda *_args, **_kw: ({"prompts": [[{"role": "user", "content": "q"}]]}, ["uid-0"]),
    )

    # Enable the resume branch and stamp a "load_checkpoints" marker into
    # recorder.events so the event-order assertion below proves on_train_start
    # fires AFTER the resume read. Returning (0, "") keeps the step count the
    # same as the no-resume case.
    from skyrl.train.utils.trainer_utils import ResumeMode

    trainer.resume_mode = ResumeMode.LATEST

    def _record_load_checkpoints():
        recorder.events.append(("load_checkpoints", {"global_step": trainer.global_step}))
        return (0, "")

    monkeypatch.setattr(trainer, "load_checkpoints", _record_load_checkpoints)

    asyncio.run(trainer.train())

    event_names = [name for name, _ in recorder.events]

    # eval_interval=2 -> step 1 has no interval-driven eval, so on_eval_start/end
    # at step 1 only appear because ForceEvaluateAtStep set should_evaluate.
    # on_save at step 2 only appears because ForceSaveAtStep set should_save.
    # Step 2's eval is interval-driven (last_step == total_training_steps).
    # The leading "load_checkpoints" marker proves on_train_start fires AFTER
    # the resume read (regression guard for callback timing).
    expected = [
        "load_checkpoints",
        "on_train_start",
        "on_epoch_start",
        # --- step 1 ---
        "on_step_start",
        "on_step_end",
        "on_eval_start",  # forced
        "on_eval_end",
        "on_log",
        # --- step 2 ---
        "on_step_start",
        "on_step_end",
        "on_save",  # forced
        "on_eval_start",  # interval
        "on_eval_end",
        "on_log",
        # --- epoch boundary + cleanup ---
        "on_epoch_end",
        "on_train_end",
    ]
    assert event_names == expected, f"unexpected event sequence: {event_names}"

    snaps_by_event: dict[str, list[dict]] = {}
    for name, snap in recorder.events:
        snaps_by_event.setdefault(name, []).append(snap)

    # Both step ends carry the (mocked) batch + step metrics
    for snap in snaps_by_event["on_step_end"]:
        assert snap["has_batch"], "on_step_end should see the training batch"
        assert snap["has_metrics"], "on_step_end should see step metrics"
        assert "policy_loss" in snap["metrics_keys"], snap["metrics_keys"]

    # Both eval ends carry eval metrics
    for snap in snaps_by_event["on_eval_end"]:
        assert snap["has_metrics"], "on_eval_end should see eval metrics"
        assert "eval/score" in snap["metrics_keys"], snap["metrics_keys"]

    # Both on_log calls carry the standard trainer keys
    for snap in snaps_by_event["on_log"]:
        log_keys = snap["logs_keys"]
        assert "trainer/epoch" in log_keys, log_keys
        assert "trainer/global_step" in log_keys, log_keys

    # on_save fired exactly once at step 2, with the fake ckpt path
    assert len(snaps_by_event["on_save"]) == 1, snaps_by_event["on_save"]
    save_snap = snaps_by_event["on_save"][0]
    assert save_snap["global_step"] == 2, save_snap
    assert save_snap["ckpt_path"] == _FAKE_CKPT_PATH, save_snap

    # The step_end that triggered the force-save must come immediately before on_save
    save_idx = event_names.index("on_save")
    assert event_names[save_idx - 1] == "on_step_end"

    # Two evals total: forced at step 1, interval at step 2.
    eval_steps = [snap["global_step"] for snap in snaps_by_event["on_eval_end"]]
    assert eval_steps == [1, 2], eval_steps

    # Loop metadata stays consistent across every callback event (skip the
    # synthetic "load_checkpoints" marker which doesn't carry CallbackInput).
    for name, snap in recorder.events:
        if name == "load_checkpoints":
            continue
        assert snap["total_steps"] == 2, f"{name}: total_steps={snap['total_steps']}"
        assert snap["steps_per_epoch"] == 2, f"{name}: steps_per_epoch={snap['steps_per_epoch']}"
