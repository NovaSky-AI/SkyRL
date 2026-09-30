import asyncio
from unittest.mock import AsyncMock, MagicMock

from skyrl.train.fully_async_trainer_sim import _SimDispatch


def test_simulated_dispatch_exposes_timing_interface_without_real_transfer():
    client = MagicMock()
    client.pause_generation = AsyncMock()
    client.resume_generation = AsyncMock()
    dispatch = _SimDispatch(client, sync_sleep=0)

    assert dispatch.get_timing_metrics() == {}
    assert dispatch.finalize_pending_saves("policy") is None
    assert dispatch.finalize_pending_saves("critic") is None
    asyncio.run(dispatch.save_weights_for_sampler())

    client.pause_generation.assert_awaited_once()
    client.resume_generation.assert_awaited_once()
    client.increment_weight_version.assert_called_once()
    assert dispatch.get_timing_metrics() == {}
