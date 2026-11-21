import asyncio

import aiohttp
import pytest

from skyrl.train.sampling_service import SamplingClient, SamplingService
from skyrl.utils.adaptive_concurrency import (
    FixedConcurrencyPolicy,
    SamplingConcurrencyController,
)


@pytest.mark.asyncio
async def test_service_stress_preserves_limit_and_exactly_once_execution():
    active = 0
    max_active = 0
    calls: dict[int, int] = {}

    async def chat_completion(payload):
        nonlocal active, max_active
        request_id = payload["json"]["request_id"]
        calls[request_id] = calls.get(request_id, 0) + 1
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep((request_id % 5) * 0.0002)
            return {"request_id": request_id}
        finally:
            active -= 1

    backend = type("Backend", (), {"chat_completion": staticmethod(chat_completion)})()
    controller = SamplingConcurrencyController(policy=FixedConcurrencyPolicy(), initial_limit=13)
    service = SamplingService(backend, controller=controller)

    results = await asyncio.gather(
        *(
            service.chat_completion({"json": {"request_id": request_id}}, attempt_id=str(request_id))
            for request_id in range(500)
        )
    )

    assert results == [{"request_id": request_id} for request_id in range(500)]
    assert calls == {request_id: 1 for request_id in range(500)}
    assert max_active == 13
    assert service.pending == 0
    assert service.in_flight == 0
    assert controller.waiting == 0
    assert controller.in_flight == 0
    await service.aclose()


@pytest.mark.asyncio
async def test_pending_cancellation_stress_leaves_no_orphan_work():
    gate = asyncio.Event()
    active = 0
    max_active = 0
    calls: set[int] = set()

    async def chat_completion(payload):
        nonlocal active, max_active
        request_id = payload["json"]["request_id"]
        calls.add(request_id)
        active += 1
        max_active = max(max_active, active)
        try:
            await gate.wait()
            return {"request_id": request_id}
        finally:
            active -= 1

    backend = type("Backend", (), {"chat_completion": staticmethod(chat_completion)})()
    controller = SamplingConcurrencyController(policy=FixedConcurrencyPolicy(), initial_limit=10)
    service = SamplingService(backend, controller=controller)
    tasks = [
        asyncio.create_task(service.chat_completion({"json": {"request_id": request_id}}, attempt_id=str(request_id)))
        for request_id in range(100)
    ]

    for _ in range(1000):
        if controller.in_flight == 10 and controller.waiting == 90:
            break
        await asyncio.sleep(0)
    assert controller.in_flight == 10
    assert controller.waiting == 90

    cancelled_ids = set(range(20, 100, 3))
    for request_id in cancelled_ids:
        tasks[request_id].cancel()
    await asyncio.sleep(0)
    gate.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert all(isinstance(results[request_id], asyncio.CancelledError) for request_id in cancelled_ids)
    assert calls == set(range(100)) - cancelled_ids
    assert max_active == 10
    assert service.pending == 0
    assert service.in_flight == 0
    assert controller.waiting == 0
    assert controller.in_flight == 0
    await service.aclose()


@pytest.mark.asyncio
async def test_direct_and_http_generators_share_one_admission_window():
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    active = 0
    max_active = 0

    async def chat_completion(payload):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            request_id = payload["json"]["request_id"]
            if request_id == "direct":
                first_started.set()
                await release_first.wait()
            return {"request_id": request_id}
        finally:
            active -= 1

    backend = type(
        "Backend",
        (),
        {
            "model_name": "model",
            "chat_completion": staticmethod(chat_completion),
            "get_endpoint_url": staticmethod(lambda: "http://raw-backend.invalid"),
        },
    )()
    controller = SamplingConcurrencyController(policy=FixedConcurrencyPolicy(), initial_limit=1)
    service = SamplingService(backend, controller=controller)
    client = SamplingClient(service, backend)
    client.start()

    direct = asyncio.create_task(client.chat_completion({"json": {"request_id": "direct"}}))
    await first_started.wait()
    async with aiohttp.ClientSession() as session:
        http = asyncio.create_task(
            session.post(
                f"{client.get_endpoint_url()}/v1/chat/completions",
                json={"request_id": "http"},
            )
        )
        for _ in range(1000):
            if service.pending == 1:
                break
            await asyncio.sleep(0.001)
        assert service.in_flight == 1
        assert service.pending == 1
        assert max_active == 1

        release_first.set()
        assert await direct == {"request_id": "direct"}
        response = await http
        async with response:
            assert response.status == 200
            assert await response.json() == {"request_id": "http"}

    assert max_active == 1
    await client.aclose_service()
