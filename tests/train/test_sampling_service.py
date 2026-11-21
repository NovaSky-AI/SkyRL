import asyncio
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from skyrl.train.sampling_service import (
    SamplingClient,
    SamplingRequest,
    SamplingService,
)
from skyrl.utils.adaptive_concurrency import (
    FixedConcurrencyPolicy,
    SamplingConcurrencyController,
)


@pytest.mark.asyncio
async def test_service_owns_waiting_requests_and_limits_backend_dispatch():
    backend = MagicMock()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    active = 0
    max_active = 0

    async def chat_completion(payload):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        if not first_started.is_set():
            first_started.set()
            await release_first.wait()
        active -= 1
        return {"id": payload["json"]["session_id"]}

    backend.chat_completion = chat_completion
    controller = SamplingConcurrencyController(policy=FixedConcurrencyPolicy(), initial_limit=1)
    service = SamplingService(backend, controller=controller)

    first = asyncio.create_task(service.chat_completion({"json": {"session_id": "a"}}, attempt_id="a"))
    await first_started.wait()
    second = asyncio.create_task(service.chat_completion({"json": {"session_id": "b"}}, attempt_id="b"))
    for _ in range(10):
        if controller.waiting == 1:
            break
        await asyncio.sleep(0)

    assert service.in_flight == 1
    assert service.pending == 1
    assert controller.in_flight == 1
    assert controller.waiting == 1
    assert max_active == 1

    release_first.set()
    assert await asyncio.gather(first, second) == [{"id": "a"}, {"id": "b"}]
    assert service.in_flight == 0
    assert service.pending == 0
    assert max_active == 1


@pytest.mark.asyncio
async def test_sampling_client_splits_generate_batch_into_service_requests():
    backend = MagicMock()
    seen = []

    async def generate(input_batch, model=None):
        seen.append(input_batch)
        prompt = input_batch["prompt_token_ids"][0]
        return {
            "responses": [str(prompt[-1])],
            "response_ids": [[prompt[-1]]],
            "stop_reasons": ["stop"],
            "response_logprobs": [[-0.1]],
            "prompt_logprobs": None,
            "rollout_expert_indices": None,
            "rollout_sample_support": [prompt[-1]],
        }

    backend.generate = generate
    backend.model_name = "model"
    service = SamplingService(backend)
    client = SamplingClient(service, backend)

    result = await client.generate(
        {
            "prompt_token_ids": [[1, 2], [3, 4]],
            "prompts": None,
            "sampling_params": {"max_tokens": 1},
            "session_ids": ["attempt-a", "attempt-b"],
            "mm_features": None,
            "cache_salt": None,
            "routed_experts_prompt_starts": [7, 9],
            "return_sample_support": True,
        }
    )

    assert result["responses"] == ["2", "4"]
    assert result["response_ids"] == [[2], [4]]
    assert result["response_logprobs"] == [[-0.1], [-0.1]]
    assert result["rollout_sample_support"] == [2, 4]
    assert [batch["routed_experts_prompt_starts"] for batch in seen] == [[7], [9]]
    assert all(batch["return_sample_support"] for batch in seen)


@pytest.mark.asyncio
async def test_client_cancellation_cancels_service_owned_backend_work():
    backend = MagicMock()
    backend.chat_completion = AsyncMock(side_effect=asyncio.Event().wait)
    controller = SamplingConcurrencyController(policy=FixedConcurrencyPolicy(), initial_limit=1)
    service = SamplingService(backend, controller=controller)

    request = asyncio.create_task(service.chat_completion({"json": {}}, attempt_id="attempt"))
    await asyncio.sleep(0)
    request.cancel()

    with pytest.raises(asyncio.CancelledError):
        await request
    assert service.pending == 0
    assert service.in_flight == 0
    assert controller.in_flight == 0


@pytest.mark.asyncio
async def test_explicit_sampling_request_uses_the_same_service_queue():
    backend = MagicMock()
    backend.chat_completion = AsyncMock(return_value={"id": "response"})
    service = SamplingService(backend)
    client = SamplingClient(service, backend)

    result = await client.submit(
        SamplingRequest(
            operation="chat_completion",
            payload={"json": {"messages": []}},
            attempt_id="trajectory-1",
        )
    )

    assert result == {"id": "response"}
    backend.chat_completion.assert_awaited_once()
    await client.aclose_service()


@pytest.mark.asyncio
async def test_endpoint_generators_route_through_service_owned_http_ingress():
    backend = MagicMock()
    backend.model_name = "model"
    backend.get_endpoint_url.return_value = "http://raw-backend.invalid"
    backend.chat_completion = AsyncMock(return_value={"id": "through-service"})
    service = SamplingService(backend)
    client = SamplingClient(service, backend)
    client.start()

    endpoint_url = client.get_endpoint_url()
    assert endpoint_url != backend.get_endpoint_url.return_value
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{endpoint_url}/v1/chat/completions",
            json={"model": "model", "messages": [{"role": "user", "content": "hello"}]},
        ) as response:
            assert response.status == 200
            assert await response.json() == {"id": "through-service"}

    backend.chat_completion.assert_awaited_once()
    await client.aclose_service()
