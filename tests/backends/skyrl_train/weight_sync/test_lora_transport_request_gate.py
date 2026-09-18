import asyncio

import pytest

from skyrl.backends.skyrl_train.weight_sync.lora_transport.request_gate import (
    LoRATransportAdmissionGate,
)


@pytest.mark.asyncio
async def test_close_drains_admitted_request_and_blocks_new_request():
    gate = LoRATransportAdmissionGate()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def run_first():
        async with gate.admit():
            first_entered.set()
            await release_first.wait()

    async def run_second():
        async with gate.admit():
            second_entered.set()

    first = asyncio.create_task(run_first())
    await first_entered.wait()
    gate.close()
    drained = asyncio.create_task(gate.wait_until_idle())
    second = asyncio.create_task(run_second())
    await asyncio.sleep(0)

    assert not drained.done()
    assert not second_entered.is_set()

    release_first.set()
    await first
    await drained
    assert not second_entered.is_set()

    gate.open()
    await second
    assert second_entered.is_set()


@pytest.mark.asyncio
async def test_cancelled_request_releases_admission():
    gate = LoRATransportAdmissionGate()
    entered = asyncio.Event()

    async def run_request():
        async with gate.admit():
            entered.set()
            await asyncio.Event().wait()

    request = asyncio.create_task(run_request())
    await entered.wait()
    gate.close()
    request.cancel()

    with pytest.raises(asyncio.CancelledError):
        await request
    await asyncio.wait_for(gate.wait_until_idle(), timeout=1)
