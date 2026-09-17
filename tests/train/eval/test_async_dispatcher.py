"""
uv run --isolated --extra dev pytest tests/train/eval/test_async_dispatcher.py
"""

import asyncio
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from skyrl.train.config.config import EvalDispatchConfig, TrainerConfig
from skyrl.train.eval import EvalBackend, EvalLease, EvalSkip, SingleAsyncEvalDispatcher


class FakeBackend(EvalBackend):
    """``sync`` blocks on a per-step gate the test opens; records entries, outstanding leases, releases."""

    def __init__(self):
        self.entered = []  # (global_step, export_dir), in sync-entry order
        self.released = []  # global_steps, in release order
        self.skip = {}  # global_step -> reason: sync raises EvalSkip after its gate opens
        self.release_error = None
        self.close_error = None
        self.closed = False
        self.leases_out = 0
        self.max_leases_out = 0
        self._gates = {}

    def gate(self, step) -> asyncio.Event:
        return self._gates.setdefault(step, asyncio.Event())

    def open(self, *steps):
        for step in steps:
            self.gate(step).set()

    async def sync(self, req):
        self.entered.append((req.global_step, req.export_dir))
        await self.gate(req.global_step).wait()
        if req.global_step in self.skip:
            raise EvalSkip(self.skip[req.global_step])
        self.leases_out += 1
        self.max_leases_out = max(self.max_leases_out, self.leases_out)
        return EvalLease(generator=MagicMock(name=f"generator-{req.global_step}"), handle=req.global_step)

    async def release(self, lease):
        self.leases_out -= 1
        self.released.append(lease.handle)
        if self.release_error is not None:
            raise self.release_error

    async def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def _dispatcher(*, max_queue_size=2, overflow_policy="backpressure", run_eval=None):
    trainer_cfg = TrainerConfig(
        export_path="/exports",
        eval_dispatch=EvalDispatchConfig(
            mode="reserved", max_queue_size=max_queue_size, overflow_policy=overflow_policy
        ),
    )
    trainer_cfg.policy.model.path = "org/launch-model"
    backend = FakeBackend()
    events = []  # (event_name, fields, the task the callback ran on)
    step = {"value": 0}
    if run_eval is None:
        run_eval = AsyncMock(side_effect=lambda **kw: {"eval/x": float(kw["global_step"])})
    dispatcher = SingleAsyncEvalDispatcher(
        trainer_cfg,
        backend=backend,
        run_eval=run_eval,
        on_event=lambda name, **fields: events.append((name, fields, asyncio.current_task())),
        current_step=lambda: step["value"],
    )
    return dispatcher, backend, run_eval, events, step


async def _let_run(rounds: int = 20):
    """Give the eval tasks a few turns of the loop; the fakes never block on real I/O."""
    for _ in range(rounds):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_submit_returns_before_the_eval_runs():
    dispatcher, backend, run_eval, events, _ = _dispatcher()

    await dispatcher.submit(5)
    await _let_run()

    assert backend.entered == [(5, "/exports/global_step_5/policy")]
    assert backend.released == []
    run_eval.assert_not_awaited()
    assert dispatcher.get_completed() == []
    assert dispatcher.pending_steps() == {5}
    assert [name for name, _, _ in events] == ["on_eval_start"]

    backend.open(5)
    await dispatcher.drain()


@pytest.mark.asyncio
async def test_step_zero_loads_the_launch_weights_and_other_steps_the_run_export():
    dispatcher, backend, _, _, _ = _dispatcher()

    await dispatcher.submit(0)  # eval_before_train on a fresh run: no export exists, none is needed
    await dispatcher.submit(7)
    backend.open(0, 7)
    await dispatcher.drain()

    assert backend.entered == [(0, "org/launch-model"), (7, "/exports/global_step_7/policy")]


@pytest.mark.asyncio
async def test_results_are_collected_at_the_evaluated_step_with_lag():
    dispatcher, backend, run_eval, events, step = _dispatcher()
    await dispatcher.submit(5)
    step["value"] = 8  # the loop moved on while the eval ran
    backend.open(5)
    await _let_run()

    results = dispatcher.get_completed()

    assert [(r.global_step, r.skipped_reason) for r in results] == [(5, None)]
    assert results[0].metrics["eval/x"] == 5.0
    assert results[0].metrics["eval/lag_steps"] == 3
    assert results[0].metrics["eval/duration_seconds"] >= 0.0
    assert results[0].duration_seconds == results[0].metrics["eval/duration_seconds"]
    assert dispatcher.pending_steps() == set()
    assert backend.released == [5]
    run_eval.assert_awaited_once_with(generator=ANY, global_step=5)
    assert [(name, fields["global_step"]) for name, fields, _ in events] == [("on_eval_start", 5), ("on_eval_end", 5)]
    assert events[1][1]["metrics"] is results[0].metrics


@pytest.mark.asyncio
async def test_backpressure_settles_exactly_the_oldest():
    dispatcher, backend, _, events, _ = _dispatcher(max_queue_size=2)
    await dispatcher.submit(5)
    await dispatcher.submit(10)
    await _let_run()
    assert dispatcher.pending_steps() == {5, 10}

    submit_15 = asyncio.create_task(dispatcher.submit(15))
    await _let_run()
    assert not submit_15.done()  # blocked on the oldest, 5
    assert backend.entered == [(5, "/exports/global_step_5/policy")]  # 10 is still waiting on the lock

    backend.open(5)
    await submit_15

    assert [r.global_step for r in dispatcher.get_completed()] == [5]  # settled inside submit(15)
    assert dispatcher.pending_steps() == {10, 15}  # nobody waited on 10
    assert [name for name, _, _ in events] == ["on_eval_start", "on_eval_start", "on_eval_end", "on_eval_start"]

    backend.open(10, 15)
    await dispatcher.drain()


@pytest.mark.asyncio
async def test_skip_drops_the_point_without_stalling():
    dispatcher, backend, _, events, _ = _dispatcher(max_queue_size=2, overflow_policy="skip")
    await dispatcher.submit(5)
    await dispatcher.submit(10)

    await dispatcher.submit(15)  # returns at once

    results = dispatcher.get_completed()
    assert [(r.global_step, r.skipped_reason, r.metrics) for r in results] == [(15, "busy", {})]
    assert dispatcher.pending_steps() == {5, 10}
    assert [fields["global_step"] for name, fields, _ in events if name == "on_eval_start"] == [5, 10]

    backend.open(5, 10)
    await dispatcher.drain()


@pytest.mark.asyncio
async def test_force_bypasses_skip():
    dispatcher, backend, _, _, _ = _dispatcher(max_queue_size=2, overflow_policy="skip")
    await dispatcher.submit(5)
    await dispatcher.submit(10)

    submit_15 = asyncio.create_task(dispatcher.submit(15, force=True))
    await _let_run()
    assert not submit_15.done()  # backpressure instead of a skip

    backend.open(5)
    await submit_15

    assert dispatcher.pending_steps() == {10, 15}
    assert [r.global_step for r in dispatcher.get_completed()] == [5]

    backend.open(10, 15)
    await dispatcher.drain()


@pytest.mark.asyncio
async def test_one_synced_version_at_a_time_in_fifo_order():
    dispatcher, backend, _, _, _ = _dispatcher(max_queue_size=3)
    await dispatcher.submit(5)
    await dispatcher.submit(10)
    await _let_run()
    assert [s for s, _ in backend.entered] == [5]  # 10 waits for 5's lease to be released

    backend.open(5)
    await _let_run()
    assert [s for s, _ in backend.entered] == [5, 10]
    assert backend.released == [5]

    backend.open(10)
    results = await dispatcher.drain()

    assert [r.global_step for r in results] == [5, 10]
    assert backend.max_leases_out == 1


@pytest.mark.asyncio
async def test_callbacks_fire_on_the_caller_side_only():
    dispatcher, backend, _, events, _ = _dispatcher()
    caller = asyncio.current_task()
    await dispatcher.submit(5)
    backend.open(5)
    await _let_run()
    assert [name for name, _, _ in events] == ["on_eval_start"]  # the finished eval has not been collected

    dispatcher.get_completed()

    assert [name for name, _, _ in events] == ["on_eval_start", "on_eval_end"]
    assert all(task is caller for _, _, task in events)


@pytest.mark.asyncio
async def test_eval_skip_and_crash_settle_as_skipped_points_and_release_the_lease():
    crash = RuntimeError("eval body exploded")

    async def run_eval(**kw):
        if kw["global_step"] == 15:
            raise crash
        return {"eval/x": float(kw["global_step"])}

    dispatcher, backend, _, _, _ = _dispatcher(max_queue_size=4, run_eval=run_eval)
    backend.skip[5] = "sync_mismatch"
    backend.skip[10] = "ckpt_missing"
    for step in (5, 10, 15, 20):
        await dispatcher.submit(step)
    backend.open(5, 10, 15, 20)

    results = await dispatcher.drain()

    assert [(r.global_step, r.skipped_reason) for r in results] == [
        (5, "sync_mismatch"),
        (10, "ckpt_missing"),
        (15, "crashed"),
        (20, None),
    ]
    assert results[2].metrics == {}
    assert backend.released == [15, 20]  # a skip raised inside sync never held a lease


@pytest.mark.asyncio
async def test_a_raising_release_is_logged_and_the_point_still_settles():
    dispatcher, backend, _, _, _ = _dispatcher()
    backend.release_error = RuntimeError("release failed")
    await dispatcher.submit(5)
    backend.open(5)

    results = await dispatcher.drain()

    assert [(r.global_step, r.skipped_reason) for r in results] == [(5, None)]
    assert results[0].metrics["eval/x"] == 5.0
    assert backend.released == [5]


@pytest.mark.asyncio
async def test_drain_returns_everything_and_get_completed_never_blocks():
    dispatcher, backend, _, _, _ = _dispatcher()
    await dispatcher.submit(5)
    await dispatcher.submit(10)

    assert dispatcher.get_completed() == []  # nothing settled, nothing waited for

    backend.open(5, 10)
    results = await dispatcher.drain()

    assert [r.global_step for r in results] == [5, 10]
    assert dispatcher.get_completed() == []
    assert dispatcher.pending_steps() == set()


@pytest.mark.asyncio
async def test_close_cancels_in_flight_evals_and_releases_the_backend():
    body_started = asyncio.Event()

    async def run_eval(**kw):
        body_started.set()
        await asyncio.Event().wait()  # parked forever: only cancellation ends it

    dispatcher, backend, _, _, _ = _dispatcher(run_eval=run_eval)
    await dispatcher.submit(5)
    await dispatcher.submit(10)
    backend.open(5)
    await body_started.wait()  # 5 holds its lease inside the eval body; 10 waits on the lock

    await dispatcher.close()

    assert backend.released == [5]  # the lease held at cancellation is given back; 10 never had one
    assert backend.closed
    assert dispatcher.pending_steps() == set()
    assert dispatcher.get_completed() == []  # cancelled evals are not reported


@pytest.mark.asyncio
async def test_close_lets_a_backend_error_propagate():
    dispatcher, backend, _, _, _ = _dispatcher()
    backend.close_error = RuntimeError("teardown failed")

    with pytest.raises(RuntimeError, match="teardown failed"):
        await dispatcher.close()


@pytest.mark.asyncio
async def test_caller_kwargs_are_not_forwarded():
    dispatcher, backend, run_eval, _, _ = _dispatcher()

    await dispatcher.submit(5, vllm_metrics_scraper=object())
    backend.open(5)
    await dispatcher.drain()

    run_eval.assert_awaited_once_with(generator=ANY, global_step=5)
