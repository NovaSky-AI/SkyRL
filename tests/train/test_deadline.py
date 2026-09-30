"""CPU unit tests for skyrl.train.utils.deadline."""

import asyncio
import pickle
import time
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import ray

from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.train.utils import Timer, deadline
from skyrl.train.utils.async_utils import cleanup_preserving_primary
from skyrl.train.utils.deadline import StepTimeoutError, WeightSyncTimeoutError
from skyrl.train.utils.trainer_utils import ResumeMode


@pytest.mark.asyncio
async def test_ray_get_uses_binding_deadline_and_timer_stage(monkeypatch):
    calls = []

    def fake_get(refs, **kwargs):
        calls.append(kwargs)
        raise ray.exceptions.GetTimeoutError()

    monkeypatch.setattr(deadline.ray, "get", fake_get)

    async with deadline.step_deadline(3, 10.0):
        with Timer("train_critic_and_policy"):
            async with deadline.step_deadline(3, 1000.0, WeightSyncTimeoutError):
                with pytest.raises(StepTimeoutError) as exc_info:
                    deadline.ray_get(["ref"], "forward_backward")
                assert type(exc_info.value) is StepTimeoutError
                assert (exc_info.value.stage, exc_info.value.operation, exc_info.value.budget_s) == (
                    "train_critic_and_policy",
                    "forward_backward",
                    10.0,
                )

            async with deadline.step_deadline(3, 1.0, WeightSyncTimeoutError):
                with pytest.raises(WeightSyncTimeoutError) as exc_info:
                    deadline.ray_get(["ref"], "broadcast_to_inference_engines")
                assert exc_info.value.budget_s == 1.0
        assert deadline.remaining() > 1.0
    assert 0 < calls[0]["timeout"] <= 10.0
    assert 0 < calls[1]["timeout"] <= 1.0


@pytest.mark.asyncio
async def test_actor_group_backload_times_out():
    @ray.remote
    class SlowBackload:
        def backload_to_gpu(self, backload_optimizer=True, backload_model=True):
            time.sleep(30)

    actor = SlowBackload.remote()
    group = PPORayActorGroup.__new__(PPORayActorGroup)
    group._actor_handlers = [actor]
    try:
        async with deadline.step_deadline(1, 0.2):
            with Timer("sync_weights"):
                with pytest.raises(StepTimeoutError) as exc_info:
                    group.backload_to_gpu()
    finally:
        ray.kill(actor)
    assert (exc_info.value.stage, exc_info.value.operation) == ("sync_weights", "backload_to_gpu")
    assert 0.15 <= exc_info.value.elapsed_s < 10


@pytest.mark.parametrize("error_cls", [StepTimeoutError, WeightSyncTimeoutError])
def test_timeout_errors_pickle(error_cls):
    err = error_cls(4, "sync_weights", "broadcast_to_inference_engines", 60.0, 61.5)
    restored = pickle.loads(pickle.dumps(err))
    assert type(restored) is error_cls
    assert (restored.global_step, restored.stage, restored.operation, restored.budget_s, restored.elapsed_s) == (
        4,
        "sync_weights",
        "broadcast_to_inference_engines",
        60.0,
        61.5,
    )


@pytest.mark.asyncio
async def test_example_async_trainer_bounds_buffer_wait_and_cancels_generator():
    trainer_cls = import_module("examples.train.async.async_trainer").AsyncRayPPOTrainer
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.colocate_all = False
    trainer.resume_mode = ResumeMode.NONE
    trainer.cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            step_timeout_s=0.05,
            weight_sync_timeout_s=None,
            eval_interval=0,
            epochs=1,
            ckpt_interval=0,
            hf_save_interval=0,
            update_ref_every_epoch=False,
        )
    )
    trainer.dispatch = SimpleNamespace(save_weights_for_sampler=AsyncMock())
    trainer.init_weight_sync_state = lambda: None
    trainer.total_training_steps = 1
    trainer.train_dataloader = [None]
    trainer.all_timings = {}
    trainer.all_metrics = {}
    trainer._profiler_start = lambda: None
    trainer._profiler_stop = lambda: None
    generator_cancelled = asyncio.Event()

    async def blocked_generator(buffer, failure):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            generator_cancelled.set()
            raise

    trainer._run_generate_loop = blocked_generator
    with pytest.raises(StepTimeoutError) as exc_info:
        await asyncio.wait_for(trainer.train(), timeout=5)
    assert (exc_info.value.stage, exc_info.value.operation) == ("step", "wait_for_generation_buffer")
    assert generator_cancelled.is_set()


# --------------------------------------------------------------------------------------
# Awaited sections
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expiry_cancels_the_await_and_names_stage_and_operation():
    with pytest.raises(StepTimeoutError) as exc_info:
        async with deadline.step_deadline(2, 0.05), Timer("step"):
            with Timer("generate"), deadline.operation("generate"):
                await asyncio.sleep(30)
    err = exc_info.value
    assert (err.global_step, err.stage, err.operation, err.budget_s) == (2, "generate", "generate", 0.05)
    assert err.elapsed_s < 10


@pytest.mark.asyncio
async def test_expiry_outside_any_operation_reports_await():
    with pytest.raises(StepTimeoutError) as exc_info:
        async with deadline.step_deadline(2, 0.05), Timer("wait_for_generation_buffer"):
            await asyncio.Event().wait()
    assert (exc_info.value.stage, exc_info.value.operation) == ("wait_for_generation_buffer", "await")


@pytest.mark.asyncio
async def test_nested_weight_sync_deadline_raises_its_own_error():
    with pytest.raises(WeightSyncTimeoutError) as exc_info:
        async with deadline.step_deadline(5, 60.0), Timer("step"):
            async with deadline.step_deadline(5, 0.05, WeightSyncTimeoutError), Timer("sync_weights"):
                with deadline.operation("/pause"):
                    await asyncio.sleep(30)
    err = exc_info.value
    assert (err.stage, err.operation, err.budget_s) == ("sync_weights", "/pause", 0.05)


@pytest.mark.asyncio
async def test_unrelated_timeout_error_passes_through():
    err = TimeoutError("from the body")
    with pytest.raises(TimeoutError) as exc_info:
        async with deadline.step_deadline(1, 60.0):
            raise err
    assert exc_info.value is err

    with pytest.raises(TimeoutError) as exc_info:
        async with deadline.step_deadline(1, 60.0):
            async with asyncio.timeout(0.01):
                await asyncio.sleep(30)
    assert not isinstance(exc_info.value, StepTimeoutError)


@pytest.mark.asyncio
async def test_cleanup_is_skipped_once_deadline_expired():
    calls = []

    async def resume_generation():
        calls.append("resume")

    with pytest.raises(StepTimeoutError) as exc_info:
        async with deadline.step_deadline(9, 0.05), Timer("sync_weights"):
            async with cleanup_preserving_primary(resume_generation, "resume_generation"):
                await asyncio.sleep(30)
    assert calls == []
    assert exc_info.value.stage == "sync_weights"


@pytest.mark.asyncio
async def test_hung_cleanup_is_bounded_and_primary_survives():
    async def hung_resume():
        await asyncio.Event().wait()

    primary = RuntimeError("broadcast failed")
    with pytest.raises(RuntimeError) as exc_info:
        async with deadline.step_deadline(9, 0.2), Timer("sync_weights"):
            async with cleanup_preserving_primary(hung_resume, "resume_generation"):
                raise primary
    assert exc_info.value is primary
    assert primary.__notes__ == [
        "resume_generation did not finish during cleanup before the step deadline expired"
    ]
