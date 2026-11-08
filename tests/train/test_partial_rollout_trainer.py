"""CPU tests for colocated partial rollout.

The tests cover the staleness manager without a staleness term, the hold-and-resume of aborted requests in the client, and
the order of engine sleep, training, checkpoint and weight sync in one step."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    InferenceServerHTTPError,
    RemoteGenerateResult,
    RemoteInferenceClient,
)
from skyrl.train.fully_async_trainer import _AsyncStalenessManager
from skyrl.train.partial_rollout_trainer import PartialRolloutRayPPOTrainer
from tests.train.test_rl_callbacks import (
    _FAKE_CKPT_PATH,
    DummyDataset,
    RecorderCallback,
    _build_test_cfg,
    _stub_training_input,
)

# ---------------------------------------------------------------------------
# _AsyncStalenessManager without a staleness term
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admission_without_staleness_holds_slot_until_trained():
    """A group holds its slot from submission until the step that trains on it.

    A finished group that waits in the buffer still holds its slot. So running + buffered never exceeds
    the in-flight bound, which is also the buffer size, and the cut fraction is
    1 - mini_batch / in_flight. There is no staleness term."""
    ctrl = _AsyncStalenessManager(max_concurrent_generation_groups=3, mini_batch_size=2, max_staleness_steps=None)
    for _ in range(3):
        await ctrl.acquire_submission_slot()
    fourth = asyncio.create_task(ctrl.acquire_submission_slot())
    await asyncio.sleep(0)
    assert not fourth.done()
    # Finishing generation does not free the slot: the group is buffered, not trained.
    await ctrl.on_rollout_accepted()
    await ctrl.on_rollout_accepted()
    await asyncio.sleep(0)
    assert not fourth.done()
    # Training step 1 consumed 2 groups: two slots free up.
    await ctrl.notify_capacity_change(2)
    await asyncio.wait_for(fourth, timeout=1)
    await ctrl.acquire_submission_slot()
    sixth = asyncio.create_task(ctrl.acquire_submission_slot())
    await asyncio.sleep(0)
    assert not sixth.done()
    # A filtered (dropped) group leaves the buffer and frees its slot.
    await ctrl.on_rollout_accepted()
    await ctrl.on_rollout_filtered()
    await asyncio.wait_for(sixth, timeout=1)
    for _ in range(3):
        await ctrl.on_rollout_accepted()
    await ctrl.on_rollout_filtered()
    # 6 submitted = 4 trained (steps 1, 2) + 2 filtered.
    await ctrl.notify_capacity_change(3)
    await ctrl.validate_state_at_epoch_end(global_step=3)


# ---------------------------------------------------------------------------
# RemoteInferenceClient.resume_aborted_requests
# ---------------------------------------------------------------------------


def _result(ids, logprobs, stop_reason):
    return RemoteGenerateResult(
        raw_response={},
        response_ids=ids,
        response_logprobs=logprobs,
        stop_reason=stop_reason,
        routed_experts=None,
        sample_support=None,
    )


def _client(resume: bool) -> RemoteInferenceClient:
    client = RemoteInferenceClient(
        proxy_url="http://router", server_urls=["http://s0"], data_parallel_size=1, resume_aborted_requests=resume
    )
    client._generate_client = MagicMock()
    return client


@pytest.mark.asyncio
async def test_generate_resumes_aborted_request_after_gate_reopens():
    """The client resubmits an aborted request after the gate reopens.

    The resubmission appends the generated tokens to the prompt and reduces the token budget. The
    caller receives one spliced completion."""
    client = _client(resume=True)
    calls = []

    async def fake_generate(**kwargs):
        calls.append((list(kwargs["prompt_token_ids"]), dict(kwargs["sampling_params"])))
        if len(calls) == 1:
            # The trainer's pause() closes the gate before the abort result from the engine arrives.
            client._close_generation_gate()
            return _result([10, 11], [-0.1, -0.2], "abort")
        return _result([12], [-0.3], "stop")

    client._generate_client.generate = fake_generate
    client._close_generation_gate()

    task = asyncio.create_task(
        client._generate_single([1, 2], {"max_tokens": 5, "logprobs": 0}, session_id=None, model="m")
    )
    await asyncio.sleep(0.05)
    assert len(calls) == 0, "no request while the gate is closed"
    client._open_generation_gate()
    await asyncio.sleep(0.05)
    assert len(calls) == 1 and not task.done(), "the aborted request is held until the gate reopens"
    client._open_generation_gate()
    out = await asyncio.wait_for(task, timeout=1)

    assert [c[0] for c in calls] == [[1, 2], [1, 2, 10, 11]]
    assert calls[1][1]["max_tokens"] == 3
    assert out["response_ids"] == [10, 11, 12]
    assert out["response_logprobs"] == [-0.1, -0.2, -0.3]
    assert out["stop_reason"] == "stop"
    # One abort held: 2 tokens in the aborted segment, prompt (2) + generated (2) re-prefilled.
    assert client.pop_resume_stats() == {
        "aborted_requests": 1,
        "aborted_segment_tokens": 2,
        "resume_prefill_tokens": 4,
        "retried_503": 0,
    }
    assert client.pop_resume_stats()["aborted_requests"] == 0, "popping resets the counters"


@pytest.mark.asyncio
async def test_generate_abort_at_token_budget_is_length():
    client = _client(resume=True)
    client._generate_client.generate = AsyncMock(return_value=_result([10, 11], None, "abort"))
    out = await client._generate_single([1], {"max_tokens": 2}, session_id=None, model="m")
    assert out["stop_reason"] == "length"
    assert out["response_ids"] == [10, 11]
    assert out["response_logprobs"] is None


@pytest.mark.asyncio
async def test_generate_retries_paused_engine_503():
    client = _client(resume=True)
    attempts = []

    async def fake_generate(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise InferenceServerHTTPError("POST", "http://router/g", 503, "engine paused")
        return _result([7], None, "stop")

    client._generate_client.generate = fake_generate
    out = await asyncio.wait_for(client._generate_single([1], {"max_tokens": 4}, session_id=None, model="m"), timeout=5)
    assert out["response_ids"] == [7] and len(attempts) == 2


@pytest.mark.asyncio
async def test_generate_surfaces_abort_when_resume_disabled():
    client = _client(resume=False)
    client._generate_client.generate = AsyncMock(return_value=_result([10], None, "abort"))
    out = await client._generate_single([1], {"max_tokens": 4}, session_id=None, model="m")
    assert out["stop_reason"] == "abort"


@pytest.mark.asyncio
async def test_pause_and_wake_drive_the_gate():
    client = _client(resume=True)
    client._call_all_servers = AsyncMock(return_value={})
    assert client._get_generation_gate().is_set()
    await client.pause(mode="abort")
    assert not client._get_generation_gate().is_set()
    await client.wake_up(tags=["weights"])
    assert not client._get_generation_gate().is_set(), "a partial wake keeps the engines paused"
    await client.wake_up(tags=["kv_cache"])
    assert client._get_generation_gate().is_set()
    await client.pause(mode="keep")
    assert client._get_generation_gate().is_set(), "a keep pause freezes requests in the engine"
    await client.pause(mode="wait")
    await client.resume()
    assert client._get_generation_gate().is_set()


# ---------------------------------------------------------------------------
# PartialRolloutRayPPOTrainer step order
# ---------------------------------------------------------------------------


def test_partial_rollout_step_sleeps_trains_checkpoints_then_syncs(monkeypatch):
    """One step runs sleep, train, checkpoint, weight sync, in this order.

    The sleep aborts the in-flight requests before training. The trainer writes the checkpoint while
    the engines sleep. The weight sync wakes the engines and comes last. The callback sequence is the
    same as in the fully-async trainer."""
    cfg = _build_test_cfg()
    cfg.trainer.fully_async.enabled = True
    cfg.trainer.fully_async.num_parallel_generation_workers = 3
    cfg.trainer.policy_mini_batch_size = cfg.trainer.train_batch_size
    cfg.trainer.algorithm.policy_loss_type = "rollout_is"
    cfg.generator.batched = False
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.ckpt_interval = 1

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2

    order = []
    client = MagicMock()
    client.sleep = AsyncMock(side_effect=lambda *a, **k: order.append("sleep"))

    recorder = RecorderCallback()
    trainer = PartialRolloutRayPPOTrainer(
        cfg=cfg,
        tracker=MagicMock(),
        tokenizer=tokenizer,
        train_dataset=DummyDataset(size=4),
        eval_dataset=DummyDataset(size=2),
        inference_engine_client=client,
        generator=MagicMock(),
        callbacks=[recorder],
    )
    assert client.resume_aborted_requests is True

    dispatch_mock = MagicMock()
    dispatch_mock.save_weights_for_sampler = AsyncMock(side_effect=lambda *a, **k: order.append("sync"))
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

    # The order is: the initial sync, two steps, the sleep at the end of training, and the final checkpoint.
    assert order == [
        "sync",
        "sleep",
        "train",
        "ckpt",
        "sync",
        "sleep",
        "train",
        "ckpt",
        "sync",
        "sleep",
        "ckpt",
    ], order
    event_names = [name for name, _ in recorder.events]
    assert event_names.count("on_step_start") == 2
    assert event_names.index("on_save") < event_names.index("on_eval_start"), "checkpoint precedes the post-sync eval"
