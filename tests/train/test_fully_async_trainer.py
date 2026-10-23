"""
CPU unit tests for fully-async trainer building blocks that back `sample_full_batch`:
the staleness manager's filtered-rollout accounting, the dataloader's trained-vs-filtered
UID tracking, and the consumer's exhaustion-aware buffer drain.
"""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from skyrl.train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GeneratedOutputGroup,
    _AsyncDataloader,
    _AsyncStalenessManager,
)
from skyrl.train.utils.async_utils import BackgroundFailure
from skyrl.train.utils.metrics import ScalarGauges, TrainingPhaseGauge
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
    failure = BackgroundFailure()

    assert await drain(dummy, buffer, done, failure) == "a"
    assert await drain(dummy, buffer, done, failure) == "b"

    # Buffer empty and generators done -> exhausted.
    await FullyAsyncRayPPOTrainer._watch_generators_done([asyncio.create_task(asyncio.sleep(0))], done, failure)
    assert await drain(dummy, buffer, done, failure) is None


@pytest.mark.asyncio
async def test_drain_next_group_drains_remaining_before_exhaustion():
    """If generators finish while items remain, those items are returned before None."""
    buffer: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    buffer.put_nowait("a")
    done.set()  # generators done, but a real item is still buffered

    drain = FullyAsyncRayPPOTrainer._drain_next_group
    dummy = object()
    assert await drain(dummy, buffer, done, BackgroundFailure()) == "a"
    assert await drain(dummy, buffer, done, BackgroundFailure()) is None


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
    assert await drain(dummy, buffer, done, BackgroundFailure()) == "x"
    await producer


@pytest.mark.asyncio
async def test_drain_next_group_raises_when_worker_fails_mid_drain():
    """A worker failure while other workers are still alive must raise, not block or read as exhaustion."""
    buffer: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    failure = BackgroundFailure()
    drain = FullyAsyncRayPPOTrainer._drain_next_group
    err = RuntimeError("generator crashed")
    trainer = FullyAsyncRayPPOTrainer.__new__(FullyAsyncRayPPOTrainer)
    trainer.async_train_dataloader = SimpleNamespace(get_next_non_consumed_data=AsyncMock(side_effect=err))

    async def live_worker():
        await asyncio.sleep(3600)

    tasks = [
        asyncio.create_task(trainer._run_generate_for_a_group_loop(buffer, failure)),
        asyncio.create_task(live_worker()),
    ]
    watcher = asyncio.create_task(FullyAsyncRayPPOTrainer._watch_generators_done(tasks, done, failure))
    with pytest.raises(RuntimeError) as exc_info:
        await asyncio.wait_for(drain(trainer, buffer, done, failure), timeout=5)
    assert exc_info.value is err
    assert err.__notes__ == ["raised in background generation worker"]
    assert not done.is_set()
    buffer.put_nowait("a")
    with pytest.raises(RuntimeError) as buffered_exc:
        await drain(trainer, buffer, done, failure)
    assert buffered_exc.value is err
    assert buffer.get_nowait() == "a"
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.wait_for(watcher, timeout=5)


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


# --------------------------------------------------------------------------------------
# train(): checkpoint step names
# --------------------------------------------------------------------------------------


def _make_train_loop_trainer(
    *,
    num_prompts=4,
    mini_batch_size=2,
    epochs=1,
    ckpt_interval=1,
    hf_save_interval=0,
    max_training_steps=None,
    exhaust_after_steps=None,
    partial_at_exhaustion=0,
    resume=None,
    ckpt_root="ckpt",
):
    """Build a FullyAsyncRayPPOTrainer whose train() runs on CPU with generation and training stubbed.

    Mini-batches are drawn from a real ``_AsyncDataloader``. ``exhaust_after_steps`` ends each
    epoch early (as ``sample_full_batch`` does) after that many steps, returning
    ``partial_at_exhaustion`` groups as the partial mini-batch to discard. ``resume`` is the
    ``load_checkpoints`` return value. Saves record ``(global_step, epoch)``; checkpoint saves
    return ``{ckpt_root}/global_step_{N}``.
    """
    trainer = FullyAsyncRayPPOTrainer.__new__(FullyAsyncRayPPOTrainer)
    num_steps_per_epoch = num_prompts // mini_batch_size
    trainer.cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            weight_sync_timeout_s=None,
            step_timeout_s=None,
            eval_interval=0,
            eval_before_train=False,
            epochs=epochs,
            ckpt_interval=ckpt_interval,
            hf_save_interval=hf_save_interval,
            max_training_steps=max_training_steps,
            update_ref_every_epoch=False,
            critic=SimpleNamespace(model=SimpleNamespace(path=None)),
        )
    )
    trainer.resume_mode = ResumeMode.NONE if resume is None else ResumeMode.LATEST
    trainer.load_checkpoints = lambda: resume
    trainer.mini_batch_size = mini_batch_size
    trainer.num_steps_per_epoch = num_steps_per_epoch
    trainer.total_training_steps = num_steps_per_epoch * epochs
    if max_training_steps is not None:
        trainer.total_training_steps = min(trainer.total_training_steps, max_training_steps)
    trainer.num_parallel_generation_workers = mini_batch_size
    trainer._gen_buffer_maxsize = mini_batch_size
    trainer.sample_full_batch = False
    trainer.async_train_dataloader = _make_async_dataloader(num_prompts, mini_batch_size)
    trainer._staleness_manager = SimpleNamespace(
        load_state_from_checkpoint=lambda step: None,
        notify_capacity_change=AsyncMock(),
        validate_state_at_epoch_end=AsyncMock(),
        on_rollout_filtered=AsyncMock(),
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
    trainer.tracker = SimpleNamespace(log=lambda *args, **kwargs: None, finish=lambda: None)
    trainer.finalize_metrics = AsyncMock()
    trainer.convert_generation_group_mini_batch_to_training_input = lambda groups, dropped: None
    trainer._run_training = AsyncMock(return_value={})

    async def idle_generator(buffer, failure):
        await asyncio.Event().wait()

    trainer._run_generate_for_a_group_loop = idle_generator

    steps_in_epoch = {"count": 0}

    async def collect(buffer, all_generators_done, failure):
        async def draw(num_groups):
            groups = []
            for _ in range(num_groups):
                prompts = await trainer.async_train_dataloader.get_next_non_consumed_data()
                groups.append(
                    GeneratedOutputGroup(generator_output={}, uid=prompts[0]["uid"], global_step_when_scheduled=0)
                )
            return groups

        if exhaust_after_steps is not None and steps_in_epoch["count"] == exhaust_after_steps:
            steps_in_epoch["count"] = 0
            return await draw(partial_at_exhaustion), [], True
        steps_in_epoch["count"] += 1
        return await draw(mini_batch_size), [], False

    trainer._collect_generation_mini_batch = collect
    trainer.saved_checkpoints = []
    trainer.saved_models = []

    def save_checkpoints():
        trainer.saved_checkpoints.append((trainer.global_step, trainer.epoch))
        return os.path.join(ckpt_root, f"global_step_{trainer.global_step}")

    trainer.save_checkpoints = save_checkpoints
    trainer.save_models = lambda: trainer.saved_models.append(trainer.global_step)
    return trainer


@pytest.mark.asyncio
async def test_final_saves_are_named_after_the_last_trained_step():
    """Two steps run; the epoch-end save already covers step 2, so nothing is saved after the loop."""
    trainer = _make_train_loop_trainer(ckpt_interval=1, hf_save_interval=1)

    await trainer.train()

    assert trainer.global_step == 2
    assert trainer.saved_checkpoints == [(1, 0), (2, 0)]
    assert trainer.saved_models == [1, 2]


@pytest.mark.asyncio
async def test_max_training_steps_saves_the_last_trained_step():
    trainer = _make_train_loop_trainer(num_prompts=8, ckpt_interval=2, hf_save_interval=2, max_training_steps=3)

    await trainer.train()

    assert trainer.global_step == 3
    assert trainer.saved_checkpoints == [(2, 0), (3, 0)]
    assert trainer.saved_models == [2, 3]


@pytest.mark.asyncio
async def test_early_epoch_end_saves_the_last_trained_step():
    """An epoch that runs out of groups after one step saves step 1, and the next epoch trains step 2."""
    trainer = _make_train_loop_trainer(epochs=2, ckpt_interval=5, hf_save_interval=5, exhaust_after_steps=1)

    await trainer.train()

    assert trainer.global_step == 2
    assert trainer.saved_checkpoints == [(1, 0), (2, 1)]
    assert trainer.saved_models == [1, 2]


@pytest.mark.asyncio
async def test_epoch_ending_before_the_first_step_saves_nothing():
    """No checkpoint or HF model is saved for step 0, which no step has trained."""
    trainer = _make_train_loop_trainer(
        ckpt_interval=1, hf_save_interval=1, exhaust_after_steps=0, partial_at_exhaustion=1
    )

    await trainer.train()

    assert trainer.global_step == 0
    assert trainer.saved_checkpoints == []
    assert trainer.saved_models == []


@pytest.mark.asyncio
@pytest.mark.parametrize("partial_at_exhaustion", [0, 1])
async def test_early_epoch_end_records_prompts_filtered_after_the_last_checkpoint(tmp_path, partial_at_exhaustion):
    """Prompts filtered after the last trained step's checkpoint are added to its fully-async state,
    without saving the model again."""
    trainer = _make_train_loop_trainer(
        num_prompts=6,
        ckpt_interval=1,
        exhaust_after_steps=1,
        partial_at_exhaustion=partial_at_exhaustion,
        ckpt_root=str(tmp_path),
    )
    checkpoint_dir = tmp_path / "global_step_1"
    checkpoint_dir.mkdir()

    await trainer.train()

    assert trainer.saved_checkpoints == [(1, 0)]
    state_path = checkpoint_dir / "fully_async_state.pt"
    if not partial_at_exhaustion:
        assert not state_path.exists()
        return
    state = torch.load(state_path, weights_only=False)
    assert state["epoch"] == 0
    assert len(state["consumed_uids"]) == 3
    assert len(state["filtered_uids"]) == 1
    assert set(state["filtered_uids"]) <= set(state["consumed_uids"])


@pytest.mark.asyncio
async def test_early_epoch_end_leaves_the_resumed_from_checkpoint_untouched(tmp_path):
    """Prompts filtered right after resuming are not written into the checkpoint the run resumed from."""
    checkpoint_dir = tmp_path / "global_step_1"
    checkpoint_dir.mkdir()
    state_path = checkpoint_dir / "fully_async_state.pt"
    torch.save({"consumed_uids": ["0", "1"], "filtered_uids": [], "epoch": 0}, state_path)
    original = state_path.read_bytes()
    trainer = _make_train_loop_trainer(
        num_prompts=6,
        ckpt_interval=1,
        exhaust_after_steps=0,
        partial_at_exhaustion=1,
        resume=(1, str(checkpoint_dir), {"0", "1"}, set(), 0),
        ckpt_root=str(tmp_path),
    )

    await trainer.train()

    assert trainer.saved_checkpoints == []
    assert state_path.read_bytes() == original


@pytest.mark.asyncio
async def test_failed_early_epoch_end_save_keeps_the_next_step():
    """If the early epoch-end save fails, ``global_step`` still names the next step, as for any other failure."""
    trainer = _make_train_loop_trainer(epochs=2, ckpt_interval=5, exhaust_after_steps=1)

    def failing_save():
        raise OSError("disk full")

    trainer.save_checkpoints = failing_save

    with pytest.raises(OSError, match="disk full"):
        await trainer.train()
    assert trainer.global_step == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resume",
    [
        pytest.param((2, "ckpt/global_step_2", set(), set(), 1), id="all_epochs_done"),
        # The epoch ended early after step 1 with every prompt trained or filtered.
        pytest.param((1, "ckpt/global_step_1", {"0", "1", "2", "3"}, {"2", "3"}, 0), id="last_epoch_exhausted"),
    ],
)
async def test_resume_with_no_steps_left_saves_nothing(resume):
    """A resumed run with nothing left to train neither syncs weights nor saves a checkpoint or HF model."""
    trainer = _make_train_loop_trainer(ckpt_interval=1, hf_save_interval=1, exhaust_after_steps=0, resume=resume)

    await trainer.train()

    assert trainer.global_step == resume[0]
    assert trainer.saved_checkpoints == []
    assert trainer.saved_models == []
    trainer.dispatch.save_weights_for_sampler.assert_not_awaited()
