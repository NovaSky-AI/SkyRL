"""SDK retries replay the original call instead of sampling again."""

from __future__ import annotations

import asyncio

import pytest

from skycap.retry import RetryCache
from tests.conftest import Stack, openai_client
from tests.test_tokens import token_stack, user

URL_BODY = {"model": "policy", "messages": [{"role": "user", "content": "q"}]}


async def _post(stack: Stack, base_url: str, body: dict, headers: dict | None = None) -> tuple[int, bytes]:
    async with stack.http.post(f"{base_url}/chat/completions", json=body, headers=headers or {}) as response:
        return response.status, await response.read()


async def test_a_retry_after_the_original_finished_is_replayed(stack: Stack) -> None:
    created = await stack.create()
    first = await _post(stack, created["base_url"], URL_BODY, {"x-stainless-retry-count": "0"})
    again = await _post(stack, created["base_url"], URL_BODY, {"x-stainless-retry-count": "1"})

    assert again == first
    assert len(stack.upstream.requests) == 1
    document = await stack.document(created["id"])
    assert len(document["nodes"][1]["calls"]) == 1
    assert document["retries"] == {"replayed": 1, "coalesced": 0}


async def test_a_retry_of_a_call_still_running_waits_for_it(stack: Stack) -> None:
    created = await stack.create()
    slow = {**URL_BODY, "mock": {"delay": 0.3}}
    original = asyncio.create_task(_post(stack, created["base_url"], slow))
    await asyncio.sleep(0.05)
    retried = await _post(stack, created["base_url"], slow, {"x-stainless-retry-count": "1"})

    assert retried == await original
    assert len(stack.upstream.requests) == 1
    assert (await stack.document(created["id"]))["retries"] == {"replayed": 1, "coalesced": 1}


async def test_the_sdk_retrying_on_its_own_gets_one_sample(stack: Stack) -> None:
    """The case this exists for: the SDK times out on a slow call and resends it."""
    created = await stack.create()
    llm = openai_client(created["base_url"], timeout=0.2, max_retries=1)
    reply = await llm.chat.completions.create(
        model="policy", messages=[user("q")], extra_body={"mock": {"delay": 0.35}}
    )

    assert reply.choices[0].message.content == "re: q"
    assert len(stack.upstream.requests) == 1
    document = await stack.document(created["id"])
    assert [n["author"] for n in document["nodes"]] == ["client", "model"]
    assert document["retries"]["replayed"] == 1


async def test_a_repeat_without_a_retry_marker_is_a_resample(stack: Stack) -> None:
    created = await stack.create()
    await _post(stack, created["base_url"], URL_BODY)
    await _post(stack, created["base_url"], URL_BODY)

    assert len(stack.upstream.requests) == 2


async def test_a_retry_of_a_failed_call_runs_again(stack: Stack) -> None:
    created = await stack.create()
    failing = {**URL_BODY, "mock": {"status": 500}}
    await _post(stack, created["base_url"], failing)
    status, _ = await _post(stack, created["base_url"], failing, {"x-stainless-retry-count": "1"})

    assert status == 500
    assert len(stack.upstream.requests) == 2


async def test_an_idempotency_key_replays_and_refuses_a_different_body(stack: Stack) -> None:
    created = await stack.create()
    key = {"Idempotency-Key": "abc"}
    first = await _post(stack, created["base_url"], URL_BODY, key)
    again = await _post(stack, created["base_url"], URL_BODY, key)
    other = await _post(stack, created["base_url"], {**URL_BODY, "temperature": 0.1}, key)

    assert again == first
    assert other[0] == 400
    assert len(stack.upstream.requests) == 1


async def test_a_relayed_text_stream_is_not_cached(stack: Stack) -> None:
    created = await stack.create()
    streamed = {**URL_BODY, "stream": True}
    await _post(stack, created["base_url"], streamed)
    await _post(stack, created["base_url"], streamed, {"x-stainless-retry-count": "1"})

    assert len(stack.upstream.requests) == 2


async def test_token_mode_coalesces_a_retry_onto_one_engine_call() -> None:
    async with token_stack() as tokens:
        # The retry lands after the timeout plus the SDK's 0.375-0.5 s backoff, while the
        # call still runs, and waits less than its own timeout for it.
        tokens.engine.delay = 1.0
        created = await tokens.create()
        llm = openai_client(created["base_url"], timeout=0.4, max_retries=1)
        await llm.chat.completions.create(model="policy", messages=[user("q")], stream=True)

        assert len(tokens.engine.requests) == 1
        trajectory = tokens.server.trajectories[created["id"]]
        assert len(trajectory.graph) == 2 and len(trajectory.graph.nodes[1].calls) == 1
        assert (trajectory.replay.replayed, trajectory.replay.coalesced) == (1, 1)


async def test_the_cache_forgets_old_and_excess_entries() -> None:
    cache = RetryCache(ttl=0.05, max_entries=2)
    for key in ("a", "b", "c"):
        entry = cache.start(key, key)
        cache.complete(key, entry, object())  # type: ignore[arg-type]
    assert cache.get("a") is None
    assert len(cache) == 2
    await asyncio.sleep(0.06)
    assert cache.get("b") is None and cache.get("c") is None


async def test_the_cache_limit_holds_once_a_burst_of_calls_finishes() -> None:
    cache = RetryCache(max_entries=2)
    running = [(key, cache.start(key, key)) for key in ("a", "b", "c")]
    assert len(cache) == 3  # a running call is never dropped
    for key, entry in running:
        cache.complete(key, entry, object())  # type: ignore[arg-type]
    assert len(cache) == 2


async def test_retries_waiting_on_a_failed_call_share_one_new_call(stack: Stack) -> None:
    created = await stack.create()
    body = {**URL_BODY, "mock": {"delay": 0.2, "fail_first": 500}}
    retry = {"x-stainless-retry-count": "1"}
    original = asyncio.create_task(_post(stack, created["base_url"], body))
    await asyncio.sleep(0.05)
    retries = await asyncio.gather(*(_post(stack, created["base_url"], body, retry) for _ in range(2)))

    assert (await original)[0] == 500
    assert retries[0] == retries[1] and retries[0][0] == 200
    assert len(stack.upstream.requests) == 2
    document = await stack.document(created["id"])
    assert len(document["nodes"]) == 2
    assert document["retries"] == {"replayed": 1, "coalesced": 1}


async def test_a_failed_keyed_call_still_refuses_a_different_body(stack: Stack) -> None:
    created = await stack.create()
    key = {"Idempotency-Key": "abc"}
    await _post(stack, created["base_url"], {**URL_BODY, "mock": {"status": 500}}, key)
    status, _ = await _post(stack, created["base_url"], URL_BODY, key)

    assert status == 400
    assert len(stack.upstream.requests) == 1


async def test_an_unreadable_reply_is_not_replayed(stack: Stack) -> None:
    created = await stack.create()
    body = {**URL_BODY, "mock": {"garbage": True}}
    await _post(stack, created["base_url"], body)
    await _post(stack, created["base_url"], body, {"x-stainless-retry-count": "1"})

    assert len(stack.upstream.requests) == 2
    assert (await stack.document(created["id"]))["retries"]["replayed"] == 0


@pytest.mark.parametrize("value", ["0", "", "x"])
async def test_a_first_attempt_is_not_a_retry(stack: Stack, value: str) -> None:
    created = await stack.create()
    await _post(stack, created["base_url"], URL_BODY)
    await _post(stack, created["base_url"], URL_BODY, {"x-stainless-retry-count": value})

    assert len(stack.upstream.requests) == 2
