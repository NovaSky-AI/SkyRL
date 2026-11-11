"""CPU tests for the hold-and-resume of aborted requests in ``RemoteInferenceClient``.

The tests cover ``resume_aborted_requests``, ``max_resumes_per_request``, the generation gate that
pause, resume and wake drive, and ``sleep_with_frozen_requests``."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    InferenceServerHTTPError,
    RemoteGenerateResult,
    RemoteInferenceClient,
)


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


@pytest.mark.asyncio
async def test_sleep_with_frozen_requests_keeps_the_gate_open_until_resume():
    """The client pauses in keep mode, sleeps the allocator, and aborts no request."""
    client = _client(resume=True)
    calls = []

    async def _call(endpoint, json=None, method="POST", params=None):
        calls.append((endpoint, json, params))
        return {}

    client._call_all_servers = _call
    assert client.requests_frozen_asleep is False
    await client.sleep_with_frozen_requests(offload_kv=True)
    assert calls == [
        ("/pause", None, {"mode": "keep", "clear_cache": "false"}),
        ("/collective_rpc", {"method": "skyrl_sleep_for_weight_sync", "kwargs": {"offload_kv": True}}, None),
    ]
    assert client.requests_frozen_asleep is True
    assert client._get_generation_gate().is_set(), "a keep pause holds no request in the client"
    await client.resume_generation()
    assert client.requests_frozen_asleep is False
