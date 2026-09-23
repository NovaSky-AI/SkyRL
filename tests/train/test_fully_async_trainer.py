"""
CPU unit tests for fully-async trainer building blocks that back `sample_full_batch`:
the staleness manager's filtered-rollout accounting, the dataloader's trained-vs-filtered
UID tracking, and the consumer's exhaustion-aware buffer drain.
"""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from torchdata.stateful_dataloader import StatefulDataLoader

from skyrl.train.config import SamplingParams
from skyrl.train.config.config import FullyAsyncConfig
from skyrl.train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GeneratedOutputGroup,
    _AsyncDataloader,
    _AsyncStalenessManager,
)
from skyrl.train.utils.trainer_utils import ResumeMode


def _make_async_dataloader(num_prompts: int, mini_batch_size: int) -> _AsyncDataloader:
    """Build an _AsyncDataloader over a trivial dataset of `num_prompts` single-prompt batches."""
    dataset = [[{"uid": str(i)}] for i in range(num_prompts)]
    # batch_size=1 (one prompt per draw) and identity collate so each batch is a list with one dict.
    loader = StatefulDataLoader(dataset, batch_size=1, collate_fn=lambda batch: batch[0])
    return _AsyncDataloader(loader, mini_batch_size)


# --------------------------------------------------------------------------------------
# _AsyncStalenessManager
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_staleness_manager_filter_restores_capacity():
    """Dropping an accepted group via on_rollout_filtered must give producer capacity back.

    This is the deadlock regression: without reclassifying accepted -> filtered, dropped groups
    keep `accepted` climbing against a fixed staleness ceiling and starve the producers.
    """
    mgr = _AsyncStalenessManager(max_concurrent_generation_groups=4, mini_batch_size=2, max_staleness_steps=1)
    # consumer_capacity = (max_staleness_steps + current_global_step) * mini_batch = (1 + 1) * 2 = 4.
    for _ in range(4):
        await mgr.acquire_submission_slot()
    for _ in range(4):
        await mgr.on_rollout_accepted()

    # At the staleness ceiling: accepted == 4 == ceiling, so no producer capacity remains.
    assert mgr._compute_capacity_unlocked() == 0

    await mgr.on_rollout_filtered()

    # Capacity restored by exactly one slot, and accounting is consistent.
    assert mgr._compute_capacity_unlocked() == 1
    assert mgr._stat.accepted == 3
    assert mgr._stat.filtered == 1
    assert mgr._stat.running == 0
    assert mgr._stat.submitted == 4


@pytest.mark.asyncio
async def test_staleness_manager_validate_epoch_end_with_filtered():
    """At epoch end, submitted == accepted + filtered and accepted == trained steps * mini_batch."""
    mgr = _AsyncStalenessManager(max_concurrent_generation_groups=4, mini_batch_size=2, max_staleness_steps=1)
    # Submit and finish 4 groups; drop 2 of them, train on the remaining 2 (one step).
    for _ in range(4):
        await mgr.acquire_submission_slot()
    for _ in range(4):
        await mgr.on_rollout_accepted()
    for _ in range(2):
        await mgr.on_rollout_filtered()

    # One training step completed -> we are now working on global_step 2.
    await mgr.notify_capacity_change(2)
    await mgr.validate_state_at_epoch_end(global_step=2)  # must not raise


# --------------------------------------------------------------------------------------
# _AsyncDataloader
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_dataloader_filtered_uids_tracking():
    adl = _make_async_dataloader(num_prompts=6, mini_batch_size=2)

    await adl.mark_consumed_uids(["0", "1"])
    await adl.mark_filtered_uids(["2"])

    assert adl.num_trained() == 2
    assert set(adl.get_filtered_uids_list()) == {"2"}
    assert set(adl.get_consumed_uids_list()) == {"0", "1", "2"}


@pytest.mark.asyncio
async def test_async_dataloader_skips_filtered_uids():
    adl = _make_async_dataloader(num_prompts=6, mini_batch_size=2)
    await adl.mark_consumed_uids(["0", "1"])
    await adl.mark_filtered_uids(["2"])

    seen = []
    while True:
        prompts = await adl.get_next_non_consumed_data()
        if prompts is None:
            break
        seen.append(prompts[0]["uid"])

    # Trained (0, 1) and filtered (2) are all skipped; only the rest are drawn.
    assert seen == ["3", "4", "5"]


@pytest.mark.asyncio
async def test_async_dataloader_load_state_restores_filtered():
    adl = _make_async_dataloader(num_prompts=6, mini_batch_size=2)
    adl.load_state_from_checkpoint({"0", "1", "2"}, {"2"})

    assert adl.num_trained() == 2
    assert set(adl.get_filtered_uids_list()) == {"2"}

    seen = []
    while True:
        prompts = await adl.get_next_non_consumed_data()
        if prompts is None:
            break
        seen.append(prompts[0]["uid"])
    assert seen == ["3", "4", "5"]


@pytest.mark.asyncio
async def test_async_dataloader_load_state_without_filtered_is_backward_compatible():
    adl = _make_async_dataloader(num_prompts=6, mini_batch_size=2)
    # Old checkpoints have no filtered set; default treats everything consumed as trained.
    adl.load_state_from_checkpoint({"0", "1"})
    assert adl.num_trained() == 2
    assert adl.get_filtered_uids_list() == []


# --------------------------------------------------------------------------------------
# _drain_next_group
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_next_group_returns_buffered_items_then_exhaustion():
    buffer: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    buffer.put_nowait("a")
    buffer.put_nowait("b")

    # _drain_next_group uses no instance state, so a bare object stands in for `self`.
    drain = FullyAsyncRayPPOTrainer._drain_next_group
    dummy = object()

    assert await drain(dummy, buffer, done) == "a"
    assert await drain(dummy, buffer, done) == "b"

    # Buffer empty and generators done -> exhausted.
    done.set()
    assert await drain(dummy, buffer, done) is None


@pytest.mark.asyncio
async def test_drain_next_group_drains_remaining_before_exhaustion():
    """If generators finish while items remain, those items are returned before None."""
    buffer: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    buffer.put_nowait("a")
    done.set()  # generators done, but a real item is still buffered

    drain = FullyAsyncRayPPOTrainer._drain_next_group
    dummy = object()
    assert await drain(dummy, buffer, done) == "a"
    assert await drain(dummy, buffer, done) is None


@pytest.mark.asyncio
async def test_drain_next_group_blocks_until_item_arrives():
    buffer: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    drain = FullyAsyncRayPPOTrainer._drain_next_group
    dummy = object()

    async def delayed_put():
        await asyncio.sleep(0.05)
        buffer.put_nowait("x")

    producer = asyncio.create_task(delayed_put())
    assert await drain(dummy, buffer, done) == "x"
    await producer


# --------------------------------------------------------------------------------------
# _should_keep_group
# --------------------------------------------------------------------------------------


def _trainer_with_tol(tol: float):
    """A stand-in for `self` exposing just the cfg field _should_keep_group reads."""
    return SimpleNamespace(
        cfg=SimpleNamespace(trainer=SimpleNamespace(algorithm=SimpleNamespace(zero_variance_filter_tol=tol)))
    )


def _group(rewards, loss_masks, uid="u"):
    return GeneratedOutputGroup(
        generator_output={"rewards": rewards, "loss_masks": loss_masks},
        uid=uid,
        global_step_when_scheduled=0,
    )


def test_should_keep_group():
    keep = FullyAsyncRayPPOTrainer._should_keep_group

    # Zero-variance group -> drop.
    assert keep(_trainer_with_tol(0.0), _group([1.0, 1.0], [[1], [1]])) is False
    # Group with reward spread -> keep.
    assert keep(_trainer_with_tol(0.0), _group([1.0, 0.0], [[1], [1]])) is True
    # Singleton -> keep.
    assert keep(_trainer_with_tol(0.0), _group([1.0], [[1]])) is True
    # Masked trajectories are ignored: two equal live rewards + one masked -> still zero-variance.
    assert keep(_trainer_with_tol(0.0), _group([1.0, 1.0, 0.0], [[1], [1], [0]])) is False
    # Near-equal float rewards within tol -> drop.
    assert keep(_trainer_with_tol(1e-6), _group([0.6667, 0.66670001], [[1], [1]])) is False


def test_reprefix_metrics():
    """generate/X -> generate_<suffix>/X, preserving the leading namespace for tracker grouping."""
    reprefix = FullyAsyncRayPPOTrainer._reprefix_metrics
    out = reprefix(
        {"generate/avg_num_tokens": 10.0, "environment/score": 0.5, "bare": 1},
        "dropped",
    )
    assert out == {
        "generate_dropped/avg_num_tokens": 10.0,
        "environment_dropped/score": 0.5,
        "dropped/bare": 1,
    }


def test_should_keep_group_token_level_rewards():
    """Token-level rewards are collapsed to per-trajectory sequence rewards for the variance check."""
    keep = FullyAsyncRayPPOTrainer._should_keep_group

    # Two trajectories, both summing to 1.0 -> zero variance -> drop.
    assert (
        keep(
            _trainer_with_tol(0.0),
            _group([[0.0, 1.0], [1.0, 0.0]], [[1, 1], [1, 1]]),
        )
        is False
    )
    # One trajectory sums to 1.0, the other to 0.0 -> variance -> keep.
    assert (
        keep(
            _trainer_with_tol(0.0),
            _group([[0.0, 1.0], [0.0, 0.0]], [[1, 1], [1, 1]]),
        )
        is True
    )


class _ControlledGenerator:
    def __init__(self, fail=False):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.active = 0
        self.fail = fail

    async def generate(self, inputs):
        self.active += 1
        if self.active == 2:
            self.started.set()
        try:
            if self.fail and inputs["trajectory_ids"][0].instance_id == "1":
                await self.started.wait()
                if self.fail == "cancelled_worker":
                    raise asyncio.CancelledError
                raise ValueError("controlled generator failure")
            await self.release.wait()
            return {"rewards": [0.0, 1.0], "loss_masks": [[1], [1]]}
        finally:
            self.active -= 1


def _lifecycle_trainer(sample_full_batch=False, timeout=None, fail=False, count=2):
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    async_config = FullyAsyncConfig()
    async_config.generation_timeout_seconds = timeout
    trainer.cfg = SimpleNamespace(
        generator=SimpleNamespace(
            n_samples_per_prompt=2,
            inference_engine=SimpleNamespace(backend="vllm"),
            sampling_params=SamplingParams(),
        ),
        environment=SimpleNamespace(env_class="gsm8k"),
        trainer=SimpleNamespace(
            fully_async=async_config,
            algorithm=SimpleNamespace(zero_variance_filter_tol=0.0),
            epochs=1,
            eval_interval=0,
        ),
    )
    trainer.global_step = 1
    trainer.mini_batch_size = 2
    trainer.sample_full_batch = sample_full_batch
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer._phase_gauge = SimpleNamespace(timed_phase=lambda *args: nullcontext())
    trainer._loop_gauges = Mock()
    dataset = [
        [{"uid": str(i), "prompt": [{"role": "user", "content": "hi"}], "env_class": None, "env_extras": {}}]
        for i in range(count)
    ]
    loader = StatefulDataLoader(dataset, batch_size=1, collate_fn=lambda rows: rows[0])
    trainer.async_train_dataloader = _AsyncDataloader(loader, mini_batch_size=2)
    trainer._staleness_manager = _AsyncStalenessManager(count, 2, count // 2 - 1)
    trainer.generator = _ControlledGenerator(fail=fail)
    trainer.resume_mode = ResumeMode.NONE
    trainer.init_weight_sync_state = Mock()
    trainer.dispatch = SimpleNamespace(save_weights_for_sampler=AsyncMock())
    trainer._ray_gpu_monitor = None
    trainer._profiler_start = Mock()
    trainer._profiler_stop = Mock()
    trainer.total_training_steps = trainer.num_steps_per_epoch = 1
    trainer.num_parallel_generation_workers = trainer._gen_buffer_maxsize = count
    return trainer


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_full_batch", [False, True])
@pytest.mark.parametrize("failure", ["deadline", "exception", "cancellation", "cancelled_worker"])
async def test_train_generation_failure_cleans_up_workers(sample_full_batch, failure):
    trainer = _lifecycle_trainer(
        sample_full_batch,
        timeout=0.05 if failure == "deadline" else None,
        fail=failure if failure in ("exception", "cancelled_worker") else False,
    )
    task = asyncio.create_task(trainer.train())
    try:
        if failure == "cancellation":
            await asyncio.wait_for(trainer.generator.started.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            match = {
                "deadline": "Generation for prompt .* exceeded 0.05 seconds",
                "exception": "controlled generator failure",
                "cancelled_worker": "Generation worker was cancelled unexpectedly",
            }[failure]
            with pytest.raises(RuntimeError, match=match):
                await asyncio.wait_for(task, 2)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert trainer.generator.active == 0
    assert trainer._staleness_manager._stat.running == 0
    assert trainer.async_train_dataloader.num_trained() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 2.0])
async def test_successful_generation_preserves_groups(timeout):
    trainer = _lifecycle_trainer(timeout=timeout)
    buffer = asyncio.Queue()
    tasks = [asyncio.create_task(trainer._run_generate_for_a_group_loop(buffer)) for _ in range(2)]
    try:
        await asyncio.wait_for(trainer.generator.started.wait(), 2)
        trainer.generator.release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert {buffer.get_nowait().uid for _ in range(buffer.qsize())} == {"0", "1"}
        assert trainer._staleness_manager._stat.accepted == 2
        assert trainer._staleness_manager._stat.running == 0
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_drain_does_not_consume_later_groups():
    buffer = asyncio.Queue()
    done = asyncio.Event()
    task = asyncio.create_task(FullyAsyncRayPPOTrainer._drain_next_group(None, buffer, done))
    for _ in range(3):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    buffer.put_nowait("next-group")
    for _ in range(3):
        await asyncio.sleep(0)
    assert buffer.get_nowait() == "next-group"


@pytest.mark.asyncio
async def test_generation_deadline_excludes_submission_wait():
    trainer = _lifecycle_trainer(timeout=0.05)
    manager = trainer._staleness_manager
    for _ in range(2):
        await manager.acquire_submission_slot()
    task = asyncio.create_task(trainer._run_generate_for_a_group_loop(asyncio.Queue()))
    try:
        await asyncio.sleep(0.1)
        assert not task.done()
        assert trainer.generator.active == 0
        trainer.generator.release.set()
        for _ in range(2):
            await manager.on_rollout_rejected()
        await asyncio.wait_for(task, 2)
        assert manager._stat.accepted == 2
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _complete_epoch_trainer(sample_full_batch=False, timeout=2, count=2):
    trainer = _lifecycle_trainer(sample_full_batch, timeout=timeout, count=count)
    trainer.cfg.trainer.ckpt_interval = 0
    trainer.cfg.trainer.hf_save_interval = 0
    trainer.cfg.trainer.max_training_steps = None
    trainer.cfg.trainer.update_ref_every_epoch = False
    trainer.cfg.trainer.critic = SimpleNamespace(model=SimpleNamespace(path=None))
    trainer._vllm_metrics_scraper = None
    trainer.tracker = Mock()
    trainer._profiler_step = Mock()
    trainer.dispatch.get_timing_metrics = Mock(return_value={})
    trainer.dispatch.finalize_pending_saves = Mock()
    trainer.convert_generation_group_mini_batch_to_training_input = Mock(return_value={})
    trainer._run_training = AsyncMock(return_value={})
    return trainer


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_full_batch", [False, True])
async def test_successful_training_epoch_with_generation_deadline(sample_full_batch):
    trainer = _complete_epoch_trainer(sample_full_batch)
    trainer.generator.release.set()
    await asyncio.wait_for(trainer.train(), 2)
    trainer._run_training.assert_awaited_once()
    assert trainer.global_step == 2
    assert trainer._staleness_manager._stat.accepted == 2
    assert trainer._staleness_manager._stat.running == 0
    trainer.tracker.finish.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_full_batch", [False, True])
@pytest.mark.parametrize("outcome", ["timeout", "error", "stop"])
async def test_generation_failure_during_final_metrics(sample_full_batch, outcome):
    trainer = _complete_epoch_trainer(sample_full_batch, timeout=0.05 if outcome == "timeout" else None, count=4)
    trainer.cfg.trainer.max_training_steps = 1
    release = asyncio.Event()
    failed = asyncio.Event()

    async def generate(inputs):
        if int(inputs["trajectory_ids"][0].instance_id) >= 2:
            try:
                await release.wait()
                raise ValueError("failure during metrics")
            finally:
                failed.set()
        return {"rewards": [0.0, 1.0], "loss_masks": [[1], [1]]}

    async def metrics():
        if outcome == "error":
            release.set()
        if outcome != "stop":
            await failed.wait()
        return {}

    trainer.generator.generate = generate
    trainer._vllm_metrics_scraper = SimpleNamespace(sample=metrics, aclose=AsyncMock())
    if outcome == "stop":
        await asyncio.wait_for(trainer.train(), 2)
        trainer.tracker.finish.assert_called_once()
    else:
        match = "exceeded 0.05 seconds" if outcome == "timeout" else "failure during metrics"
        with pytest.raises(RuntimeError, match=match):
            await asyncio.wait_for(trainer.train(), 2)
        trainer.tracker.finish.assert_not_called()
    assert trainer._staleness_manager._stat.running == 0


@pytest.mark.asyncio
async def test_shutdown_preserves_failure_not_yet_seen_by_watcher():
    error = RuntimeError("worker finished before watcher ran")

    async def fail():
        raise error

    worker = asyncio.create_task(fail())
    await asyncio.gather(worker, return_exceptions=True)
    failure = asyncio.get_running_loop().create_future()
    watcher = asyncio.create_task(FullyAsyncRayPPOTrainer._watch_generation_workers([worker], asyncio.Event(), failure))
    await FullyAsyncRayPPOTrainer._stop_generation_workers([worker], watcher, failure)
    assert failure.result() is error
