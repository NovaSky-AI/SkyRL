"""
CPU tests for `trainer.fully_async.save_during_eval`: the saves of an eval step run while the eval runs.

The real `FullyAsyncRayPPOTrainer.train()` loop runs one training step with generation, training and weight sync
stubbed. The eval and the saves record the time interval in which they ran.

uv run --isolated --extra skyrl-train --extra dev pytest tests/train/test_fully_async_save_during_eval.py
"""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.utils.metrics import ScalarGauges, TrainingPhaseGauge
from skyrl.train.utils.trainer_utils import ResumeMode

MINI_BATCH_SIZE = 2
# Upper bound for the eval to wait for the HF model save to start. When the save runs during the eval, the eval
# does not wait this long.
SAVE_START_TIMEOUT_S = 5.0


class _FakeAsyncDataloader:
    """Counts the trained UIDs of the epoch, which `train()` checks after each step."""

    def __init__(self):
        self.trained = 0

    def num_trained(self):
        return self.trained

    async def mark_consumed_uids(self, uids):
        self.trained += len(uids)

    async def reset_at_epoch_end(self):
        self.trained = 0


def _make_trainer(
    *,
    save_during_eval,
    ckpt_interval=0,
    eval_error=None,
    save_start_timeout_s=SAVE_START_TIMEOUT_S,
    save_duration_s=0.0,
):
    """Build a trainer for one step with an eval and an HF model save due at that step.

    The eval waits until the HF model save of the same step has started, for at most ``save_start_timeout_s``.
    Then it raises ``eval_error`` if one is given. ``trainer.intervals`` maps ``(name, global_step)`` to the
    ``(start, end)`` monotonic times of each eval and save.
    """
    trainer = FullyAsyncRayPPOTrainer.__new__(FullyAsyncRayPPOTrainer)
    trainer.cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            weight_sync_timeout_s=None,
            step_timeout_s=None,
            eval_interval=1,
            eval_before_train=False,
            epochs=1,
            ckpt_interval=ckpt_interval,
            hf_save_interval=1,
            max_training_steps=None,
            update_ref_every_epoch=False,
            critic=SimpleNamespace(model=SimpleNamespace(path=None)),
            fully_async=SimpleNamespace(save_during_eval=save_during_eval),
        )
    )
    trainer.resume_mode = ResumeMode.NONE
    trainer.mini_batch_size = MINI_BATCH_SIZE
    trainer.num_steps_per_epoch = 1
    trainer.total_training_steps = 1
    trainer.num_parallel_generation_workers = MINI_BATCH_SIZE
    trainer._gen_buffer_maxsize = MINI_BATCH_SIZE
    trainer.sample_full_batch = False
    trainer.async_train_dataloader = _FakeAsyncDataloader()
    trainer._staleness_manager = SimpleNamespace(
        notify_capacity_change=AsyncMock(), validate_state_at_epoch_end=AsyncMock()
    )
    trainer.dispatch = SimpleNamespace(
        save_weights_for_sampler=AsyncMock(),
        get_timing_metrics=lambda: {},
        finalize_pending_saves=lambda model: None,
    )
    trainer.ref_model = None
    trainer.init_weight_sync_state = lambda: None
    trainer._ray_gpu_monitor = None
    trainer._vllm_metrics_scraper = None
    trainer._profiler_start = trainer._profiler_step = trainer._profiler_stop = lambda: None
    trainer._phase_gauge = TrainingPhaseGauge()
    trainer._loop_gauges = ScalarGauges()
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer.logged = []
    trainer.tracker = SimpleNamespace(
        log=lambda data, step, commit=False: trainer.logged.append((step, dict(data))), finish=lambda: None
    )
    trainer.finalize_metrics = AsyncMock()
    trainer.convert_generation_group_mini_batch_to_training_input = lambda groups, dropped: None
    trainer._run_training = AsyncMock(return_value={})

    async def idle_generator(buffer, failure):
        await asyncio.Event().wait()

    trainer._run_generate_for_a_group_loop = idle_generator

    async def collect(buffer, all_generators_done, failure):
        uids = [f"{trainer.global_step}-{i}" for i in range(MINI_BATCH_SIZE)]
        return [SimpleNamespace(uid=uid) for uid in uids], [], False

    trainer._collect_generation_mini_batch = collect

    trainer.intervals = {}
    save_started = threading.Event()

    async def fake_eval():
        step, start = trainer.global_step, time.monotonic()
        await asyncio.to_thread(save_started.wait, save_start_timeout_s)
        trainer.intervals[("eval", step)] = (start, time.monotonic())
        if eval_error is not None:
            raise eval_error
        return {"eval/score": 1.0}

    def fake_save_checkpoints():
        step, start = trainer.global_step, time.monotonic()
        trainer.intervals[("save_checkpoints", step)] = (start, time.monotonic())
        return f"global_step_{step}"

    def fake_save_models():
        step, start = trainer.global_step, time.monotonic()
        save_started.set()
        time.sleep(save_duration_s)
        trainer.intervals[("save_hf_model", step)] = (start, time.monotonic())

    trainer.eval = fake_eval
    trainer.save_checkpoints = fake_save_checkpoints
    trainer.save_models = fake_save_models
    return trainer


@pytest.mark.asyncio
async def test_checkpoint_and_hf_model_save_run_during_eval():
    trainer = _make_trainer(save_during_eval=True, ckpt_interval=1)

    await trainer.train()

    eval_start, eval_end = trainer.intervals[("eval", 1)]
    ckpt_start, ckpt_end = trainer.intervals[("save_checkpoints", 1)]
    save_start, _ = trainer.intervals[("save_hf_model", 1)]
    # Both saves start during the eval, and the HF model save starts after the checkpoint ends.
    assert eval_start < ckpt_start <= ckpt_end <= save_start < eval_end
    # Both ended before the step logged its timings.
    step_timings = next(data for step, data in trainer.logged if step == 1 and "timing/eval" in data)
    assert {"timing/eval", "timing/save_hf_model"} <= set(step_timings)
    assert any(step == 1 and data.get("eval/score") == 1.0 for step, data in trainer.logged)


@pytest.mark.asyncio
async def test_hf_model_save_runs_after_eval_by_default():
    # The save cannot start during the eval here, so the eval does not wait for it.
    trainer = _make_trainer(save_during_eval=False, save_start_timeout_s=0.0)

    await trainer.train()

    _, eval_end = trainer.intervals[("eval", 1)]
    save_start, _ = trainer.intervals[("save_hf_model", 1)]
    assert eval_end <= save_start


@pytest.mark.asyncio
async def test_failed_eval_waits_for_the_hf_model_save():
    trainer = _make_trainer(save_during_eval=True, eval_error=RuntimeError("eval failed"), save_duration_s=0.3)

    with pytest.raises(RuntimeError, match="eval failed"):
        await trainer.train()

    _, eval_end = trainer.intervals[("eval", 1)]
    _, save_end = trainer.intervals[("save_hf_model", 1)]
    assert eval_end < save_end
