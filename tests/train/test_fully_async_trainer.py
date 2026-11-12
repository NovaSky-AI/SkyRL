"""
CPU unit tests for fully-async trainer building blocks that back `sample_full_batch`:
the staleness manager's filtered-rollout accounting, the dataloader's trained-vs-filtered
UID tracking, and the consumer's exhaustion-aware buffer drain.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from torchdata.stateful_dataloader import StatefulDataLoader

from skyrl.train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GeneratedOutputGroup,
    _AsyncDataloader,
    _AsyncStalenessManager,
)
from skyrl.train.utils.async_utils import BackgroundFailure


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


# ---------------------------------------------------------------------------
# Subclass hooks
# ---------------------------------------------------------------------------


def test_subclass_hooks_run_in_order_around_the_step(monkeypatch):
    """A subclass sees its hooks in this order per step: before training, training, checkpoint (when
    ``checkpoint_before_weight_sync`` is set), weight sync. ``_after_training_loop`` runs once, before the
    final checkpoint."""
    from unittest.mock import MagicMock

    from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
    from tests.train.test_rl_callbacks import (
        _FAKE_CKPT_PATH,
        DummyDataset,
        _build_test_cfg,
        _stub_training_input,
    )

    order = []

    class _Hooked(FullyAsyncRayPPOTrainer):
        checkpoint_before_weight_sync = True

        def _validate_placement_and_capacity(self) -> None:
            order.append("validate")

        async def _before_training_step(self) -> None:
            order.append("before")

        async def _sync_weights_after_training_step(self) -> None:
            order.append("sync")

        async def _after_training_loop(self) -> None:
            order.append("after_loop")

    cfg = _build_test_cfg()
    cfg.trainer.fully_async.enabled = True
    cfg.trainer.fully_async.num_parallel_generation_workers = cfg.trainer.train_batch_size
    cfg.trainer.policy_mini_batch_size = cfg.trainer.train_batch_size
    cfg.trainer.algorithm.policy_loss_type = "rollout_is"
    cfg.generator.batched = False
    cfg.trainer.ckpt_interval = 1
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    trainer = _Hooked(
        cfg=cfg,
        tracker=MagicMock(),
        tokenizer=tokenizer,
        train_dataset=DummyDataset(size=4),
        eval_dataset=DummyDataset(size=2),
        inference_engine_client=None,
        generator=MagicMock(),
    )
    assert order == ["validate"]

    dispatch_mock = MagicMock()
    dispatch_mock.save_weights_for_sampler = AsyncMock(side_effect=lambda *a, **k: order.append("initial_sync"))
    dispatch_mock.get_timing_metrics = MagicMock(return_value={})
    trainer.dispatch = dispatch_mock
    monkeypatch.setattr(trainer, "init_weight_sync_state", lambda: None)
    monkeypatch.setattr(trainer, "_run_generate_for_a_group_loop", AsyncMock(return_value=None))
    step_groups = iter([["uid-0", "uid-1"], ["uid-2", "uid-3"]])

    async def _collect(*_args, **_kwargs):
        return [SimpleNamespace(uid=uid) for uid in next(step_groups)], [], False

    async def _train(*_args, **_kwargs):
        order.append("train")
        return {"policy_loss": 0.1}

    def _save():
        order.append("ckpt")
        return _FAKE_CKPT_PATH

    monkeypatch.setattr(trainer, "_collect_generation_mini_batch", _collect)
    monkeypatch.setattr(
        trainer, "convert_generation_group_mini_batch_to_training_input", lambda *_args: _stub_training_input()
    )
    monkeypatch.setattr(trainer, "_run_training", _train)
    monkeypatch.setattr(trainer, "eval", AsyncMock(return_value={"eval/score": 0.5}))
    monkeypatch.setattr(trainer, "save_checkpoints", _save)
    monkeypatch.setattr(trainer._staleness_manager, "validate_state_at_epoch_end", AsyncMock(return_value=None))

    asyncio.run(trainer.train())

    assert order == [
        "validate",
        "initial_sync",
        "before",
        "train",
        "ckpt",
        "sync",
        "before",
        "train",
        "ckpt",
        "sync",
        "after_loop",
        "ckpt",
    ], order


def test_checkpoint_before_weight_sync_rejects_sample_full_batch():
    """An exhausted epoch saves its checkpoint with no training step before it, so a subclass that
    needs the engines asleep for the checkpoint cannot use ``sample_full_batch``."""
    from unittest.mock import MagicMock

    from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
    from tests.train.test_rl_callbacks import DummyDataset, _build_test_cfg

    class _CheckpointsWhileEnginesSleep(FullyAsyncRayPPOTrainer):
        checkpoint_before_weight_sync = True

    cfg = _build_test_cfg()
    cfg.trainer.fully_async.enabled = True
    cfg.trainer.fully_async.num_parallel_generation_workers = cfg.trainer.train_batch_size
    cfg.trainer.policy_mini_batch_size = cfg.trainer.train_batch_size
    cfg.trainer.algorithm.policy_loss_type = "rollout_is"
    cfg.generator.batched = False
    cfg.trainer.fully_async.sample_full_batch = True
    cfg.trainer.algorithm.zero_variance_filter = True
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    with pytest.raises(AssertionError, match="checkpoint_before_weight_sync"):
        _CheckpointsWhileEnginesSleep(
            cfg=cfg,
            tracker=MagicMock(),
            tokenizer=tokenizer,
            train_dataset=DummyDataset(size=4),
            eval_dataset=DummyDataset(size=2),
            inference_engine_client=None,
            generator=MagicMock(),
        )
