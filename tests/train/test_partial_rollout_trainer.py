"""CPU tests for colocated partial rollout.

The tests cover the admission controller of the partial-rollout trainer, the hold-and-resume of
aborted requests in the client, the order of engine sleep, training, checkpoint and weight sync in
one step, and the two stale-group policies ``wait`` and ``drop``."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    InferenceServerHTTPError,
    RemoteGenerateResult,
    RemoteInferenceClient,
)
from skyrl.train.fully_async_trainer import GeneratedOutputGroup, _AsyncDataloader
from skyrl.train.partial_rollout_trainer import (
    PartialRolloutRayPPOTrainer,
    _PartialRolloutAdmissionController,
)
from skyrl.train.utils.async_utils import BackgroundFailure
from tests.train.test_rl_callbacks import (
    _FAKE_CKPT_PATH,
    DummyDataset,
    RecorderCallback,
    _build_test_cfg,
    _stub_training_input,
)


class _ListLoader:
    """The subset of ``StatefulDataLoader`` that ``_AsyncDataloader`` uses, over a list of batches."""

    def __init__(self, batches):
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)

    def state_dict(self):
        return {}

    def load_state_dict(self, _state):
        pass


# ---------------------------------------------------------------------------
# _PartialRolloutAdmissionController
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admission_without_staleness_holds_slot_until_trained():
    """A group holds its slot from submission until the step that trains on it.

    A finished group that waits in the buffer still holds its slot. So running + buffered never exceeds
    the in-flight bound, which is also the buffer size, and the cut fraction is
    1 - mini_batch / in_flight. There is no staleness term."""
    ctrl = _PartialRolloutAdmissionController(max_concurrent_generation_groups=3, mini_batch_size=2)
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
        "retried_503_gate_open": 0,
        "not_resumed_requests": 0,
        "completed_requests": 1,
        "cuts_survived_sum": 1,
        "cuts_survived_max": 1,
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


def _partial_cfg(**partial_rollout):
    cfg = _build_test_cfg()
    cfg.trainer.fully_async.enabled = True
    cfg.trainer.fully_async.num_parallel_generation_workers = 3
    cfg.trainer.policy_mini_batch_size = cfg.trainer.train_batch_size
    cfg.trainer.algorithm.policy_loss_type = "rollout_is"
    cfg.generator.batched = False
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.ckpt_interval = 1
    for k, v in partial_rollout.items():
        setattr(cfg.trainer.fully_async.partial_rollout, k, v)
    return cfg


def _partial_trainer(client, cfg=None, callbacks=(), train_size=4):
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    return PartialRolloutRayPPOTrainer(
        cfg=cfg if cfg is not None else _partial_cfg(),
        tracker=MagicMock(),
        tokenizer=tokenizer,
        train_dataset=DummyDataset(size=train_size),
        eval_dataset=DummyDataset(size=2),
        inference_engine_client=client,
        generator=MagicMock(),
        callbacks=list(callbacks),
    )


def test_partial_rollout_step_sleeps_trains_checkpoints_then_syncs(monkeypatch):
    """One step runs sleep, train, checkpoint, weight sync, in this order.

    The sleep aborts the in-flight requests before training. The trainer writes the checkpoint while
    the engines sleep. The weight sync wakes the engines and comes last. The callback sequence is the
    same as in the fully-async trainer."""
    order = []
    client = MagicMock()
    client.sleep = AsyncMock(side_effect=lambda *a, **k: order.append("sleep"))
    recorder = RecorderCallback()
    trainer = _partial_trainer(client, callbacks=[recorder])
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


# ---------------------------------------------------------------------------
# stale_group_policy: drop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_stops_resuming_at_max_resumes_per_request():
    """With a resume budget of 1, the second abort returns the spliced partial result with stop reason
    ``abort`` instead of a third submission."""
    client = _client(resume=True)
    client.max_resumes_per_request = 1
    results = iter([_result([10], [-0.1], "abort"), _result([11], [-0.2], "abort"), _result([12], [-0.3], "stop")])
    calls = []

    async def fake_generate(**kwargs):
        calls.append(list(kwargs["prompt_token_ids"]))
        return next(results)

    client._generate_client.generate = fake_generate
    out = await asyncio.wait_for(client._generate_single([1, 2], {"max_tokens": 8}, session_id=None, model="m"), 5)
    assert out["stop_reason"] == "abort"
    assert out["response_ids"] == [10, 11]
    assert out["response_logprobs"] == [-0.1, -0.2]
    assert calls == [[1, 2], [1, 2, 10]]
    stats = client.pop_resume_stats()
    assert stats["aborted_requests"] == 2
    assert stats["not_resumed_requests"] == 1
    assert stats["resume_prefill_tokens"] == 3, "only the first abort was resubmitted"
    assert stats["completed_requests"] == 0, "a request that is not resumed does not count as completed"
    assert stats["cuts_survived_max"] == 0


@pytest.mark.asyncio
async def test_abort_before_the_first_token_does_not_count_against_the_budget():
    """A request that the engine aborts while it is still queued has no tokens. The client resubmits it
    without charging the budget, so a request that is not resumed always carries at least one token."""
    client = _client(resume=True)
    client.max_resumes_per_request = 1
    results = iter(
        [
            _result([], [], "abort"),
            _result([10], [-0.1], "abort"),
            _result([11], [-0.2], "abort"),
            _result([12], [-0.3], "stop"),
        ]
    )
    calls = []

    async def fake_generate(**kwargs):
        calls.append(list(kwargs["prompt_token_ids"]))
        return next(results)

    client._generate_client.generate = fake_generate
    out = await asyncio.wait_for(client._generate_single([1, 2], {"max_tokens": 8}, session_id=None, model="m"), 5)
    assert out["stop_reason"] == "abort"
    assert out["response_ids"] == [10, 11]
    assert calls == [[1, 2], [1, 2], [1, 2, 10]], "the empty abort was resubmitted as-is"
    stats = client.pop_resume_stats()
    assert stats["aborted_requests"] == 3
    assert stats["not_resumed_requests"] == 1
    assert stats["resume_prefill_tokens"] == 5


def test_drop_policy_sets_the_client_budget_and_rejects_aborted_groups():
    client = MagicMock()
    trainer = _partial_trainer(client, cfg=_partial_cfg(stale_group_policy="drop", max_cuts_per_group=2))
    assert client.max_resumes_per_request == 2
    assert trainer._epoch_may_exhaust_early() is True

    def group(stop_reasons):
        return GeneratedOutputGroup(
            generator_output={"stop_reasons": stop_reasons, "response_ids": [[1]] * len(stop_reasons)},
            uid="u",
            global_step_when_scheduled=1,
        )

    assert trainer._accept_generated_group(group(["stop", "length"])) is True
    assert trainer._accept_generated_group(group(["stop", "abort"])) is False
    assert trainer._stale_groups_dropped == 1


def test_ignore_policy_accepts_aborted_groups_and_sets_no_budget():
    client = MagicMock()
    client.max_resumes_per_request = None
    trainer = _partial_trainer(client)
    assert client.max_resumes_per_request is None
    assert trainer._epoch_may_exhaust_early() is False
    group = GeneratedOutputGroup(
        generator_output={"stop_reasons": ["abort"], "response_ids": [[1]]}, uid="u", global_step_when_scheduled=1
    )
    assert trainer._accept_generated_group(group) is True


@pytest.mark.asyncio
async def test_worker_drops_a_rejected_group_and_marks_its_prompt_filtered():
    """A rejected group never reaches the buffer. The worker frees its slot and marks the uid filtered,
    so the prompt is not drawn again and the epoch can end early."""
    client = MagicMock()
    trainer = _partial_trainer(client, cfg=_partial_cfg(stale_group_policy="drop", max_cuts_per_group=0))
    trainer.global_step = 1
    seen = []

    async def fake_generate(generator_input, **_kwargs):
        seen.append(generator_input["prompts"][0])
        n = len(generator_input["prompts"])
        reason = "abort" if len(seen) == 1 else "stop"
        return {
            "prompt_token_ids": [[1]] * n,
            "response_ids": [[2]] * n,
            "rewards": [1.0] * n,
            "loss_masks": [[1]] * n,
            "stop_reasons": [reason] * n,
            "rollout_metrics": {},
        }

    trainer.generator.generate = fake_generate
    # The dummy dataset yields bare tuples. The worker needs prompt rows with a uid, so swap in a
    # async dataloader over four such rows.
    rows = [
        [{"uid": f"u{i}", "prompt": [{"role": "user", "content": f"q{i}"}], "env_class": None, "env_extras": {}}]
        for i in range(4)
    ]
    trainer.async_train_dataloader = _AsyncDataloader(_ListLoader(rows), trainer.mini_batch_size)
    buffer: asyncio.Queue = asyncio.Queue(maxsize=trainer._generation_buffer_maxsize())
    failure = BackgroundFailure()
    await asyncio.wait_for(trainer._run_generate_for_a_group_loop(buffer, failure), timeout=10)

    assert len(seen) == 4, "the worker drew every prompt of the epoch once"
    assert buffer.qsize() == 3, "the first group was dropped; the other three were buffered"
    assert len(trainer.async_train_dataloader.get_filtered_uids_list()) == 1
    assert trainer.async_train_dataloader.num_trained() == 0
    assert trainer._stale_groups_dropped == 1
    assert trainer._in_flight_start_steps == {}
    ctrl = trainer._staleness_manager
    assert ctrl._stat.running == 0 and ctrl._stat.submitted == 4 and ctrl._stat.accepted == 3


# ---------------------------------------------------------------------------
# stale_group_policy: wait
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_policy_delays_the_sleep_until_threshold_groups_finish():
    """Under ``wait`` the engines do not sleep while a group that started at or before
    ``global_step - max_cuts_per_group`` is still in flight."""
    client = MagicMock()
    client.sleep = AsyncMock()
    trainer = _partial_trainer(client, cfg=_partial_cfg(stale_group_policy="wait", max_cuts_per_group=2))
    trainer.global_step = 5
    trainer._generation_failure = BackgroundFailure()
    trainer._in_flight_start_steps = {"old": 3, "older": 2, "young": 4}
    trainer._in_flight_changed = asyncio.Event()

    step = asyncio.create_task(trainer._before_training_step())
    await asyncio.sleep(0.01)
    assert not step.done() and client.sleep.await_count == 0

    # One stale group finishes: still one at the threshold.
    trainer._in_flight_start_steps.pop("older")
    trainer._in_flight_changed.set()
    await asyncio.sleep(0.01)
    assert not step.done() and client.sleep.await_count == 0

    # The last stale group finishes. The young one (staleness 1 < 2 after the cut) may stay in flight.
    trainer._in_flight_start_steps.pop("old")
    trainer._in_flight_changed.set()
    await asyncio.wait_for(step, timeout=5)
    assert client.sleep.await_count == 1
    assert trainer._in_flight_start_steps == {"young": 4}
    assert trainer.all_metrics["partial_rollout/stale_groups_waited"] == 2
    assert "wait_for_stale_groups" in trainer.all_timings


@pytest.mark.asyncio
async def test_wait_policy_raises_when_a_worker_fails_during_the_wait():
    client = MagicMock()
    client.sleep = AsyncMock()
    trainer = _partial_trainer(client, cfg=_partial_cfg(stale_group_policy="wait", max_cuts_per_group=0))
    trainer.global_step = 2
    failure = BackgroundFailure()
    trainer._generation_failure = failure
    trainer._in_flight_start_steps = {"old": 2}
    trainer._in_flight_changed = asyncio.Event()

    step = asyncio.create_task(trainer._before_training_step())
    await asyncio.sleep(0.01)
    err = RuntimeError("generator crashed")
    failure.record(err, "generation worker")
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(step, timeout=5)
    assert client.sleep.await_count == 0


def test_rejects_an_unknown_stale_group_policy():
    with pytest.raises(AssertionError, match="stale_group_policy"):
        _partial_trainer(MagicMock(), cfg=_partial_cfg(stale_group_policy="resample"))
