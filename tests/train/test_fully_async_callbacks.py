"""CPU test: callbacks fire end-to-end during a FullyAsyncRayPPOTrainer training run.

Runs the real fully async loop (generation workers, staleness manager, mini-batch collection and
``convert_generation_group_mini_batch_to_training_input``) over a small prompt dataset. Generation,
the inference engine and the GPU-heavy worker methods are mocked, so the tests exercise the
orchestration in ``FullyAsyncRayPPOTrainer.train()``, and the ``trajectory_ids`` that reach
``on_step_end`` travel the same path as in a real run.

uv run --isolated --extra skyrl-train --extra dev pytest tests/train/test_fully_async_callbacks.py -v
"""

import asyncio
from collections import Counter
from unittest.mock import AsyncMock, MagicMock

import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.utils.callbacks import CallbackInput, TrainingCallback
from skyrl.train.utils.trainer_utils import ResumeMode
from tests.train.util import example_dummy_config

# ---------------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------------

_N_SAMPLES_PER_PROMPT = 2
_FAKE_CKPT_PATH = "/fake/fully-async-callback-test/global_step_2"


class DummyPromptDataset:
    """Prompt rows in the shape ``prepare_generator_input`` reads; the async dataloader draws one per batch."""

    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "prompt": [{"role": "user", "content": f"q{idx}"}],
            "env_class": None,
            "env_extras": {},
            "uid": str(idx),
        }

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
                    "batch_rows": ci.batch.batch_size if ci.batch is not None else None,
                    "trajectory_ids": ci.trajectory_ids,
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


class StampLogs(TrainingCallback):
    """Adds ``custom/stamp`` to every on_log payload and keeps each payload it received."""

    def __init__(self):
        self.payloads: list[dict] = []

    def on_log(self, trainer, ci, control):
        ci.logs["custom/stamp"] = ci.global_step
        self.payloads.append(ci.logs)


def _stub_training_input(uids: list[str]) -> TrainingInputBatch:
    """Minimal TrainingInputBatch with one row per uid that survives the keys ``_run_training`` pops."""
    num_rows = len(uids)
    batch = TrainingInputBatch(
        {
            "sequences": torch.zeros((num_rows, 4), dtype=torch.long),
            "attention_mask": torch.ones((num_rows, 4), dtype=torch.long),
            "loss_mask": torch.ones((num_rows, 4), dtype=torch.long),
            "response_mask": torch.ones((num_rows, 4), dtype=torch.long),
            "rewards": torch.zeros((num_rows, 4)),
        }
    )
    batch.metadata = {
        "uids": list(uids),
        "response_length": 4,
        "avg_response_length": 4.0,
    }
    return batch


def _varied_rewards(uid: str, n: int) -> list[float]:
    """Rewards that differ within the group, so the group has reward variance."""
    return [float(i % 2) for i in range(n)]


def _build_test_cfg():
    cfg = example_dummy_config()
    cfg.trainer.epochs = 1
    cfg.trainer.fully_async.enabled = True
    # 2 prompts per step: 4 prompts give 2 steps per epoch.
    cfg.trainer.train_batch_size = 2
    cfg.trainer.policy_mini_batch_size = 2
    cfg.trainer.fully_async.max_staleness_steps = 0
    cfg.trainer.fully_async.num_parallel_generation_workers = 2
    cfg.trainer.algorithm.policy_loss_type = "rollout_is"
    cfg.trainer.algorithm.dynamic_sampling.type = None
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.update_ref_every_epoch = False
    cfg.trainer.eval_interval = 0
    cfg.trainer.eval_before_train = False
    cfg.trainer.ckpt_interval = 0
    cfg.trainer.hf_save_interval = 0
    cfg.trainer.ckpt_path = ""
    cfg.trainer.dump_data_batch = False
    # Keeps the Ray GPU monitor's scrape thread out of the step logs.
    cfg.trainer.enable_ray_gpu_monitor = False
    cfg.generator.batched = False
    cfg.generator.n_samples_per_prompt = _N_SAMPLES_PER_PROMPT
    cfg.generator.step_wise_trajectories = False
    cfg.generator.inference_engine.enable_ray_prometheus_stats = False
    return cfg


def _make_trainer(monkeypatch, cfg, num_prompts: int, callbacks=None, rewards_for_uid=_varied_rewards):
    """Builds a FullyAsyncRayPPOTrainer with generation, the worker methods and checkpointing mocked.

    Returns the trainer and a list that receives, for each converted mini-batch, the uids passed to
    ``convert_to_training_input`` (one per batch row, in row order).
    """

    def _generate(generator_input):
        trajectory_ids = generator_input["trajectory_ids"]
        n = len(trajectory_ids)
        return {
            "prompt_token_ids": [[1, 2] for _ in range(n)],
            "response_ids": [[3, 4] for _ in range(n)],
            "rewards": rewards_for_uid(trajectory_ids[0].instance_id, n),
            "loss_masks": [[1, 1] for _ in range(n)],
            "stop_reasons": ["stop"] * n,
            "rollout_metrics": {},
            "rollout_logprobs": None,
            "trajectory_ids": list(trajectory_ids),
        }

    generator = MagicMock()
    generator.generate = AsyncMock(side_effect=_generate)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.decode = MagicMock(return_value="")

    trainer = FullyAsyncRayPPOTrainer(
        cfg=cfg,
        tracker=MagicMock(),
        tokenizer=tokenizer,
        train_dataset=DummyPromptDataset(num_prompts),
        eval_dataset=DummyPromptDataset(2),
        inference_engine_client=None,
        generator=generator,
        callbacks=callbacks,
    )

    # Replace dispatch (normally built by build_models).
    dispatch_mock = MagicMock()
    dispatch_mock.save_weights_for_sampler = AsyncMock(return_value=None)
    dispatch_mock.get_timing_metrics = MagicMock(return_value={})
    dispatch_mock.finalize_pending_saves = MagicMock()
    dispatch_mock.get_lcm_dp_size = MagicMock(return_value=1)
    trainer.dispatch = dispatch_mock

    converted_uids: list[list[str]] = []

    def _convert_to_training_input(generator_output, uids):
        converted_uids.append(list(uids))
        return _stub_training_input(uids)

    monkeypatch.setattr(trainer, "convert_to_training_input", _convert_to_training_input)
    monkeypatch.setattr(trainer, "init_weight_sync_state", lambda: None)
    monkeypatch.setattr(trainer, "fwd_logprobs_values_reward", lambda batch: batch)
    monkeypatch.setattr(trainer, "compute_advantages_and_returns", lambda batch: batch)
    monkeypatch.setattr(trainer, "train_critic_and_policy", lambda batch: {"policy_loss": 0.42})
    # A fresh dict per call, so keys a callback adds to one eval's metrics don't carry into the next.
    monkeypatch.setattr(trainer, "eval", AsyncMock(side_effect=lambda *_args, **_kw: {"eval/score": 0.5}))
    # Saves don't touch disk; on_save still receives the fake path.
    monkeypatch.setattr(trainer, "save_checkpoints", lambda: _FAKE_CKPT_PATH)
    monkeypatch.setattr(trainer, "save_models", lambda: None)
    return trainer, converted_uids


def _snaps(recorder: RecorderCallback, event: str) -> list[dict]:
    return [snap for name, snap in recorder.events if name == event]


def _id_pairs(trajectory_ids) -> list[tuple[str, int]]:
    """``TrajectoryID`` is unhashable, so compare IDs as ``(instance_id, repetition_id)`` tuples."""
    return [(tid.instance_id, tid.repetition_id) for tid in trajectory_ids]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_callbacks_fire_during_fully_async_training(monkeypatch):
    """A 2-step fully async run fires every event, in order, with the right payloads."""
    cfg = _build_test_cfg()
    # eval_interval=2 means step 1 has no interval-driven eval; only the force-evaluate
    # callback can trigger eval at step 1. Step 2 still gets an interval-driven eval.
    cfg.trainer.eval_interval = 2

    recorder = RecorderCallback()
    trainer, converted_uids = _make_trainer(
        monkeypatch,
        cfg,
        num_prompts=4,
        callbacks=[recorder, ForceEvaluateAtStep(step=1), ForceSaveAtStep(step=2)],
    )

    # Enable the resume branch and stamp a "load_checkpoints" marker into recorder.events, so the
    # event-order assertion below shows on_train_start fires after the resume read.
    trainer.resume_mode = ResumeMode.LATEST

    def _record_load_checkpoints():
        recorder.events.append(("load_checkpoints", {"global_step": trainer.global_step}))
        return (0, "", set(), set(), None)

    monkeypatch.setattr(trainer, "load_checkpoints", _record_load_checkpoints)

    asyncio.run(trainer.train())

    event_names = [name for name, _ in recorder.events]
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
        "on_eval_start",  # interval
        "on_eval_end",
        "on_save",  # forced
        "on_log",
        # --- epoch boundary + cleanup ---
        "on_epoch_end",
        "on_train_end",
    ]
    assert event_names == expected, f"unexpected event sequence: {event_names}"

    step_ends = _snaps(recorder, "on_step_end")
    assert [snap["global_step"] for snap in step_ends] == [1, 2]
    assert len(converted_uids) == len(step_ends)
    for snap, uids in zip(step_ends, converted_uids):
        assert snap["has_batch"], "on_step_end should see the training batch"
        assert "policy_loss" in snap["metrics_keys"], snap["metrics_keys"]
        trajectory_ids = snap["trajectory_ids"]
        # One ID per batch row, in the row order convert_to_training_input received.
        assert len(trajectory_ids) == snap["batch_rows"]
        assert [tid.instance_id for tid in trajectory_ids] == uids
        # Every sample of each of the step's groups, once.
        expected_pairs = [(uid, rep) for uid in set(uids) for rep in range(_N_SAMPLES_PER_PROMPT)]
        assert Counter(_id_pairs(trajectory_ids)) == Counter(expected_pairs)
    # Across both steps, each of the 4 prompts is trained exactly once.
    all_pairs = [pair for snap in step_ends for pair in _id_pairs(snap["trajectory_ids"])]
    assert sorted(all_pairs) == sorted((str(i), rep) for i in range(4) for rep in range(_N_SAMPLES_PER_PROMPT))

    # Two evals total: forced at step 1, interval at step 2.
    eval_ends = _snaps(recorder, "on_eval_end")
    assert [snap["global_step"] for snap in eval_ends] == [1, 2]
    for snap in eval_ends:
        assert "eval/score" in snap["metrics_keys"], snap["metrics_keys"]

    # on_save fired exactly once at step 2, with the fake ckpt path.
    saves = _snaps(recorder, "on_save")
    assert len(saves) == 1, saves
    assert saves[0]["global_step"] == 2, saves[0]
    assert saves[0]["ckpt_path"] == _FAKE_CKPT_PATH, saves[0]

    for snap in _snaps(recorder, "on_log"):
        log_keys = snap["logs_keys"]
        assert "trainer/global_step" in log_keys, log_keys
        assert any(key.startswith("timing/") for key in log_keys), log_keys

    # Loop counters stay consistent across every callback event (skip the synthetic
    # "load_checkpoints" marker). on_train_end comes after the epoch loop, so its epoch is not checked.
    for name, snap in recorder.events:
        if name == "load_checkpoints":
            continue
        assert snap["total_steps"] == 2, f"{name}: total_steps={snap['total_steps']}"
        assert snap["steps_per_epoch"] == 2, f"{name}: steps_per_epoch={snap['steps_per_epoch']}"
        if name != "on_train_end":
            assert snap["epoch"] == 0, f"{name}: epoch={snap['epoch']}"


def test_each_step_logs_one_merged_payload(monkeypatch):
    """Each tracker.log call is committed and receives the payload on_log saw, including callback-added keys.

    Eval before training logs at step 0; each training step logs once, with its metrics and timings.
    """
    cfg = _build_test_cfg()
    cfg.trainer.eval_interval = 2
    cfg.trainer.eval_before_train = True

    stamper = StampLogs()
    trainer, _ = _make_trainer(monkeypatch, cfg, num_prompts=4, callbacks=[stamper])

    asyncio.run(trainer.train())

    log_calls = trainer.tracker.log.call_args_list
    assert [log_call.kwargs["step"] for log_call in log_calls] == [0, 1, 2]
    assert all(log_call.kwargs["commit"] is True for log_call in log_calls)
    assert len(stamper.payloads) == len(log_calls)
    for log_call, payload in zip(log_calls, stamper.payloads):
        assert log_call.args[0] is payload
    assert [payload["custom/stamp"] for payload in stamper.payloads] == [0, 1, 2]

    eval_before_train, step_1, step_2 = stamper.payloads
    assert "eval/score" in eval_before_train
    step_keys = ("trainer/global_step", "trainer/epoch", "async/staleness_mean", "timing/step", "timing/run_training")
    for payload in (step_1, step_2):
        for key in step_keys:
            assert key in payload, sorted(payload)
    # eval_interval=2: only step 2 evals.
    assert "eval/score" not in step_1
    assert "eval/score" in step_2


def test_dropped_groups_are_excluded_from_trajectory_ids(monkeypatch):
    """Groups dropped by sample_full_batch never reach on_step_end.

    One of 6 prompts has zero reward variance and is dropped. The other 5 fill 2 steps; the third step
    collects the last group, runs out of prompts, discards it and is abandoned.
    """
    cfg = _build_test_cfg()
    cfg.trainer.fully_async.sample_full_batch = True
    cfg.trainer.algorithm.zero_variance_filter = True

    dropped_uid = "0"

    def _rewards(uid, n):
        return [1.0] * n if uid == dropped_uid else _varied_rewards(uid, n)

    recorder = RecorderCallback()
    trainer, converted_uids = _make_trainer(
        monkeypatch, cfg, num_prompts=6, callbacks=[recorder], rewards_for_uid=_rewards
    )

    asyncio.run(trainer.train())

    event_names = [name for name, _ in recorder.events]
    # The abandoned step fires on_step_start with no matching on_step_end, then the epoch ends.
    assert event_names.count("on_step_start") == 3
    assert event_names.count("on_step_end") == 2
    assert event_names[-4:] == ["on_log", "on_step_start", "on_epoch_end", "on_train_end"], event_names

    step_ends = _snaps(recorder, "on_step_end")
    assert len(converted_uids) == len(step_ends)
    trained_uids = set()
    for snap, uids in zip(step_ends, converted_uids):
        instance_ids = [tid.instance_id for tid in snap["trajectory_ids"]]
        assert instance_ids == uids
        trained_uids.update(instance_ids)
    assert dropped_uid not in trained_uids
    # 4 groups trained; the discarded fifth one is absent too.
    assert len(trained_uids) == 4
    assert trainer.generator.generate.await_count == 6


def test_early_stop_still_ends_the_epoch(monkeypatch):
    """Stopping at max_training_steps mid-run fires on_epoch_end once, before on_train_end."""
    cfg = _build_test_cfg()
    cfg.trainer.epochs = 2
    cfg.trainer.max_training_steps = 1

    recorder = RecorderCallback()
    trainer, _ = _make_trainer(monkeypatch, cfg, num_prompts=4, callbacks=[recorder])

    asyncio.run(trainer.train())

    event_names = [name for name, _ in recorder.events]
    assert event_names == [
        "on_train_start",
        "on_epoch_start",
        "on_step_start",
        "on_step_end",
        "on_log",
        "on_epoch_end",
        "on_train_end",
    ], event_names


def test_callbacks_register_via_constructor_and_add_callback(monkeypatch):
    """Callbacks passed to the constructor and registered with add_callback both receive every event."""
    cfg = _build_test_cfg()
    from_constructor = RecorderCallback()
    added = RecorderCallback()
    trainer, _ = _make_trainer(monkeypatch, cfg, num_prompts=4, callbacks=[from_constructor])
    trainer.add_callback(added)

    asyncio.run(trainer.train())

    constructor_events = [name for name, _ in from_constructor.events]
    assert constructor_events, "the constructor callback received no events"
    assert [name for name, _ in added.events] == constructor_events


def test_training_without_callbacks(monkeypatch):
    """A run with no callbacks completes and makes one committed tracker.log call per step."""
    cfg = _build_test_cfg()
    trainer, _ = _make_trainer(monkeypatch, cfg, num_prompts=4)

    asyncio.run(trainer.train())

    log_calls = trainer.tracker.log.call_args_list
    assert [log_call.kwargs["step"] for log_call in log_calls] == [1, 2]
    assert all(log_call.kwargs["commit"] is True for log_call in log_calls)
    trainer.tracker.finish.assert_called_once()


def test_resume_reports_the_resumed_step_and_epoch(monkeypatch):
    """A run resumed mid-epoch reports the checkpoint's step and epoch, and trains the epoch's remaining prompts."""
    cfg = _build_test_cfg()
    cfg.trainer.epochs = 2

    recorder = RecorderCallback()
    trainer, _ = _make_trainer(monkeypatch, cfg, num_prompts=4, callbacks=[recorder])
    trainer.resume_mode = ResumeMode.LATEST
    # Checkpoint at step 3: the first step of epoch 1 trained prompts "0" and "1".
    monkeypatch.setattr(trainer, "load_checkpoints", lambda: (3, "/fake/global_step_3", {"0", "1"}, set(), 1))

    asyncio.run(trainer.train())

    train_start = _snaps(recorder, "on_train_start")[0]
    assert (train_start["global_step"], train_start["epoch"]) == (3, 1), train_start
    assert [snap["epoch"] for snap in _snaps(recorder, "on_epoch_start")] == [1]

    step_ends = _snaps(recorder, "on_step_end")
    assert [snap["global_step"] for snap in step_ends] == [4]
    assert {tid.instance_id for tid in step_ends[0]["trajectory_ids"]} == {"2", "3"}

    for name, snap in recorder.events:
        assert snap["total_steps"] == 4, f"{name}: total_steps={snap['total_steps']}"
        assert snap["steps_per_epoch"] == 2, f"{name}: steps_per_epoch={snap['steps_per_epoch']}"
