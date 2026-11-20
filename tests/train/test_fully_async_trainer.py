"""
CPU unit tests for fully-async trainer building blocks that back `sample_full_batch`:
the staleness manager's filtered-rollout accounting, the dataloader's trained-vs-filtered
UID tracking, and the consumer's exhaustion-aware buffer drain.
"""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
from tests.train.util import example_dummy_config


def _make_async_dataloader(num_prompts: int, mini_batch_size: int) -> _AsyncDataloader:
    """Build an _AsyncDataloader over a trivial dataset of `num_prompts` single-prompt batches."""
    dataset = [[{"uid": str(i)}] for i in range(num_prompts)]
    # batch_size=1 (one prompt per draw) and identity collate so each batch is a list with one dict.
    loader = StatefulDataLoader(dataset, batch_size=1, collate_fn=lambda batch: batch[0])
    return _AsyncDataloader(loader, mini_batch_size)


async def _train_from_checkpoint(
    source, destination, *, load_cursor=True, sample_full_batch=True, max_steps=None, drop_all=False, dispatch=None
):
    class Prompts(list):
        collate_fn = staticmethod(list)

    prompts = Prompts({"uid": str(idx), "prompt": [], "env_class": None, "env_extras": {}} for idx in range(8))
    cfg = example_dummy_config()
    cfg.trainer.resume_mode = "from_path" if source else "none"
    cfg.trainer.resume_path = str(source) if source else None
    cfg.trainer.resume_load_dataloader_state = load_cursor
    cfg.trainer.ckpt_path = str(destination)
    cfg.trainer.ckpt_interval = 7
    cfg.trainer.epochs = 3
    cfg.trainer.max_training_steps = max_steps
    cfg.trainer.eval_interval = -1
    cfg.trainer.enable_ray_gpu_monitor = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.policy_mini_batch_size = 2
    cfg.trainer.fully_async.enabled = True
    cfg.trainer.fully_async.num_parallel_generation_workers = 4
    cfg.trainer.fully_async.sample_full_batch = sample_full_batch
    cfg.trainer.algorithm.zero_variance_filter = True
    cfg.trainer.update_ref_every_epoch = True
    cfg.generator.n_samples_per_prompt = 2
    cfg.generator.inference_engine.enable_ray_prometheus_stats = False
    trained = []
    generated = []

    async def generate(inputs):
        uid = inputs["trajectory_ids"][0].instance_id
        generated.append((trainer.epoch, uid))
        return {"rewards": [0.0, float(not drop_all and uid != "0")], "loss_masks": [[1], [1]]}

    async def train_batch(uids):
        trained.append((trainer.global_step, trainer.epoch, uids))
        return {}

    trainer = FullyAsyncRayPPOTrainer(cfg, MagicMock(), None, prompts, None, SimpleNamespace(generate=generate))
    trainer.dispatch = dispatch if dispatch is not None else MagicMock()
    trainer.dispatch.save_weights_for_sampler = AsyncMock()
    trainer.init_weight_sync_state = MagicMock()
    trainer.ref_model = MagicMock()
    trainer.update_ref_with_policy = MagicMock()
    trainer._cleanup_old_checkpoints = MagicMock()
    trainer.convert_generation_group_mini_batch_to_training_input = lambda kept, dropped: [group.uid for group in kept]
    trainer._run_training = train_batch
    await asyncio.wait_for(trainer.train(), timeout=10)
    return trained, generated


@pytest.mark.asyncio
@pytest.mark.parametrize("resume_step", [7, 8], ids=["partway", "boundary"])
@pytest.mark.parametrize("sample_full_batch", [False, True])
async def test_full_resume_of_shortened_epoch(tmp_path, resume_step, sample_full_batch):
    initial = tmp_path / "global_step_6"
    initial.mkdir()
    torch.save({"global_step": 6}, initial / "trainer_state.pt")

    first_run = tmp_path / "uninterrupted"
    reference, generated = await _train_from_checkpoint(
        initial, first_run, load_cursor=False, sample_full_batch=sample_full_batch
    )
    assert [epoch for _, epoch, _ in reference] == [1, 1] + [2] * (3 if sample_full_batch else 4)
    assert int((first_run / "latest_ckpt_global_step.txt").read_text()) == reference[-1][0]
    source = first_run
    if not sample_full_batch:
        source = tmp_path / "stopped"
        await _train_from_checkpoint(initial, source, load_cursor=False, sample_full_batch=False, max_steps=resume_step)
        assert int((source / "latest_ckpt_global_step.txt").read_text()) == resume_step
    checkpoint = source / f"global_step_{resume_step}"
    assert torch.load(checkpoint / "trainer_state.pt", weights_only=False)["global_step"] == resume_step
    if not sample_full_batch:
        saved_cursor = torch.load(checkpoint / "fully_async_state.pt")
        stopped = await _train_from_checkpoint(
            checkpoint, source, load_cursor=False, sample_full_batch=False, max_steps=resume_step
        )
        assert stopped == ([], [])
        assert torch.load(checkpoint / "fully_async_state.pt") == saved_cursor
    resumed = tmp_path / "resumed"
    actual, regenerated = await _train_from_checkpoint(checkpoint, resumed, sample_full_batch=sample_full_batch)
    assert actual == [batch for batch in reference if batch[0] > resume_step]
    consumed = torch.load(checkpoint / "fully_async_state.pt")["consumed_uids"]
    assert [uid for epoch, uid in regenerated if epoch == 1] == [
        uid for epoch, uid in generated if epoch == 1 and uid not in consumed
    ]
    if resume_step == 7:
        assert (resumed / "global_step_8" / "fully_async_state.pt").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "progress, expected_steps, expected_uids",
    [({}, [8, 9], ["3", "4", "5", "6", "7"]), ({"skipped_steps_in_epoch": 3}, [], [])],
    ids=["legacy", "shortened-boundary"],
)
async def test_full_resume_uses_epoch_state_not_global_step(tmp_path, progress, expected_steps, expected_uids):
    checkpoint = tmp_path / "global_step_7"
    checkpoint.mkdir()
    torch.save({"global_step": 7}, checkpoint / "trainer_state.pt")
    state = {"epoch": 1, "consumed_uids": ["0", "1", "2"], "filtered_uids": ["0"], **progress}
    torch.save(state, checkpoint / "fully_async_state.pt")
    trained, generated = await _train_from_checkpoint(checkpoint, tmp_path / "resumed")
    assert [step for step, epoch, _ in trained if epoch == 1] == expected_steps
    assert sorted(uid for epoch, uid in generated if epoch == 1) == expected_uids


@pytest.mark.asyncio
async def test_filtered_epochs_checkpoint_only_completed_steps(tmp_path):
    original = tmp_path / "original"
    dispatch = MagicMock()
    trained, generated = await _train_from_checkpoint(None, original, drop_all=True, dispatch=dispatch)
    assert not trained and len(generated) == 24
    dispatch.save_checkpoint.assert_called_once()
    assert {path.name for path in original.glob("global_step_*")} == {"global_step_0"}
    assert await _train_from_checkpoint(original / "global_step_0", tmp_path / "resumed") == ([], [])


@pytest.mark.asyncio
async def test_partial_resume_generation_budget_bounds_and_replaces_filtered_groups(monkeypatch):
    class Generator:
        async def generate(self, generator_input):
            uid = generator_input["uid"]
            await asyncio.sleep(int(uid) * 0.01)
            rewards = [1.0, 1.0] if uid == "0" else [0.0, 1.0]
            return {"rewards": rewards, "loss_masks": [[1], [1]]}

    def prepare_generator_input(rand_prompts, *_args, **_kwargs):
        uid = rand_prompts[0]["uid"]
        return {"uid": uid, "prompts": []}, [uid]

    monkeypatch.setattr(
        "skyrl.train.fully_async_trainer.prepare_generator_input",
        prepare_generator_input,
    )
    monkeypatch.setattr(
        "skyrl.train.fully_async_trainer.get_sampling_params_for_backend",
        lambda *_args, **_kwargs: {},
    )

    trainer = FullyAsyncRayPPOTrainer.__new__(FullyAsyncRayPPOTrainer)
    trainer.async_train_dataloader = _make_async_dataloader(num_prompts=8, mini_batch_size=2)
    trainer._staleness_manager = _AsyncStalenessManager(
        max_concurrent_generation_groups=4,
        mini_batch_size=2,
        max_staleness_steps=1,
    )
    trainer.generator = Generator()
    trainer.global_step = 8
    trainer.mini_batch_size = 2
    trainer.num_parallel_generation_workers = 4
    trainer.sample_full_batch = True
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer._phase_gauge = SimpleNamespace(timed_phase=lambda *_args: nullcontext())
    trainer.cfg = SimpleNamespace(
        generator=SimpleNamespace(
            n_samples_per_prompt=2,
            inference_engine=SimpleNamespace(backend="test"),
            sampling_params={},
        ),
        environment=SimpleNamespace(env_class="test"),
        trainer=SimpleNamespace(algorithm=SimpleNamespace(zero_variance_filter_tol=0.0)),
    )

    buffer = asyncio.Queue(maxsize=4)
    generation_budget = asyncio.Semaphore(2)
    tasks = [
        asyncio.create_task(trainer._run_generate_for_a_group_loop(buffer, BackgroundFailure(), generation_budget))
        for _ in range(4)
    ]
    try:
        kept, dropped, exhausted = await asyncio.wait_for(
            trainer._collect_generation_mini_batch(buffer, asyncio.Event(), BackgroundFailure(), generation_budget),
            timeout=1,
        )
        while trainer._staleness_manager._stat.running:
            await asyncio.sleep(0)

        assert [group.uid for group in dropped] == ["0"]
        assert [group.uid for group in kept] == ["1", "2"]
        assert exhausted is False
        assert buffer.empty()
        assert trainer._staleness_manager._stat.accepted == 2
        assert trainer._staleness_manager._stat.filtered == 1
        assert all(not task.done() for task in tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # Exhaustion must also wake peers blocked on permits so sample_full_batch can terminate.
    trainer.async_train_dataloader = _make_async_dataloader(num_prompts=2, mini_batch_size=2)
    trainer._staleness_manager = _AsyncStalenessManager(
        max_concurrent_generation_groups=4,
        mini_batch_size=2,
        max_staleness_steps=1,
    )
    buffer = asyncio.Queue(maxsize=4)
    generation_budget = asyncio.Semaphore(4)
    tasks = [
        asyncio.create_task(trainer._run_generate_for_a_group_loop(buffer, BackgroundFailure(), generation_budget))
        for _ in range(4)
    ]
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
        assert buffer.qsize() == 2
        assert trainer._staleness_manager._stat.running == 0
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


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
