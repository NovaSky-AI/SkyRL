"""
uv run --isolated --extra dev pytest tests/train/eval/test_dispatcher.py
"""

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from skyrl.train.eval import BlockingEvalDispatcher


def _dispatcher(run_eval=None):
    """A blocking dispatcher over mocks."""
    run_eval = AsyncMock(return_value={"eval/x": 1.0}) if run_eval is None else run_eval
    on_event = MagicMock()
    return BlockingEvalDispatcher(run_eval, on_event=on_event), run_eval, on_event


@pytest.mark.asyncio
async def test_blocking_submit_brackets_the_eval_with_the_callbacks():
    order = []
    run_eval = AsyncMock(side_effect=lambda **_: order.append("run_eval") or {"eval/x": 1.0})
    dispatcher, _, on_event = _dispatcher(run_eval)
    on_event.side_effect = lambda name, **_: order.append(name)
    sentinel = object()

    await dispatcher.submit(7, vllm_metrics_scraper=sentinel)

    assert order == ["on_eval_start", "run_eval", "on_eval_end"]
    run_eval.assert_awaited_once_with(vllm_metrics_scraper=sentinel)
    assert on_event.call_args_list == [
        call("on_eval_start", global_step=7),
        call("on_eval_end", global_step=7, metrics={"eval/x": 1.0}),
    ]


@pytest.mark.asyncio
async def test_blocking_submit_baseline_shape_passes_no_kwargs():
    dispatcher, run_eval, _ = _dispatcher()

    await dispatcher.submit(0)

    run_eval.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_get_completed_returns_each_result_once_at_the_evaluated_step():
    dispatcher, _, on_event = _dispatcher()
    await dispatcher.submit(7)

    results = dispatcher.get_completed()

    assert [(r.global_step, r.metrics, r.skipped_reason) for r in results] == [(7, {"eval/x": 1.0}, None)]
    assert results[0].duration_seconds >= 0.0
    # Exactly once: nothing is handed back twice, and collecting fires no further callbacks.
    assert dispatcher.get_completed() == []
    assert await dispatcher.drain() == []
    assert on_event.call_count == 2


@pytest.mark.asyncio
async def test_drain_returns_the_settled_results_in_submission_order():
    dispatcher, _, _ = _dispatcher()
    await dispatcher.submit(2)
    await dispatcher.submit(3)

    assert [r.global_step for r in await dispatcher.drain()] == [2, 3]
    assert dispatcher.get_completed() == []


@pytest.mark.asyncio
async def test_blocking_submit_propagates_exceptions():
    dispatcher, _, on_event = _dispatcher(AsyncMock(side_effect=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        await dispatcher.submit(1)

    assert dispatcher.get_completed() == []
    assert on_event.call_args_list == [call("on_eval_start", global_step=1)]


@pytest.mark.asyncio
async def test_callback_exception_propagates_from_submit():
    dispatcher, _, on_event = _dispatcher()

    def _raise_on_end(name, **_):
        if name == "on_eval_end":
            raise ValueError("callback failed")

    on_event.side_effect = _raise_on_end

    with pytest.raises(ValueError, match="callback failed"):
        await dispatcher.submit(3)

    assert dispatcher.get_completed() == []
