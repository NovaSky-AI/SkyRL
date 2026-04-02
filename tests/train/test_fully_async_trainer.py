from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer


def _make_trainer(preserve_inflight_kv_cache_on_weight_update: bool):
    trainer = FullyAsyncRayPPOTrainer.__new__(FullyAsyncRayPPOTrainer)
    trainer.cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            fully_async=SimpleNamespace(
                preserve_inflight_kv_cache_on_weight_update=preserve_inflight_kv_cache_on_weight_update
            )
        )
    )
    trainer.inference_engine_client = AsyncMock()
    trainer.async_sync_policy_weights_to_inference_engines = AsyncMock()
    return trainer


@pytest.mark.asyncio
async def test_sync_policy_weights_preserves_inflight_kv_cache_by_default():
    call_order = []
    trainer = _make_trainer(True)
    trainer.inference_engine_client.pause_generation.side_effect = lambda: call_order.append("pause")
    trainer.async_sync_policy_weights_to_inference_engines.side_effect = lambda: call_order.append("sync")
    trainer.inference_engine_client.resume_generation.side_effect = lambda: call_order.append("resume")

    await trainer._sync_policy_weights_with_inflight_update_policy()

    assert call_order == ["pause", "sync", "resume"]
    trainer.inference_engine_client.reset_prefix_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_policy_weights_can_reset_running_requests_before_sync():
    call_order = []
    trainer = _make_trainer(False)
    trainer.inference_engine_client.pause_generation.side_effect = lambda: call_order.append("pause")
    trainer.inference_engine_client.reset_prefix_cache.side_effect = lambda **kwargs: call_order.append(
        f"reset:{kwargs['reset_running_requests']}"
    )
    trainer.async_sync_policy_weights_to_inference_engines.side_effect = lambda: call_order.append("sync")
    trainer.inference_engine_client.resume_generation.side_effect = lambda: call_order.append("resume")

    await trainer._sync_policy_weights_with_inflight_update_policy()

    assert call_order == ["pause", "reset:True", "sync", "resume"]


@pytest.mark.asyncio
async def test_sync_policy_weights_resumes_generation_if_sync_fails():
    trainer = _make_trainer(True)
    trainer.async_sync_policy_weights_to_inference_engines.side_effect = RuntimeError("sync failed")

    with pytest.raises(RuntimeError, match="sync failed"):
        await trainer._sync_policy_weights_with_inflight_update_policy()

    trainer.inference_engine_client.pause_generation.assert_awaited_once()
    trainer.inference_engine_client.resume_generation.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_policy_weights_resumes_generation_if_reset_fails():
    trainer = _make_trainer(False)
    trainer.inference_engine_client.reset_prefix_cache.side_effect = RuntimeError("reset failed")

    with pytest.raises(RuntimeError, match="reset failed"):
        await trainer._sync_policy_weights_with_inflight_update_policy()

    trainer.inference_engine_client.pause_generation.assert_awaited_once()
    trainer.inference_engine_client.resume_generation.assert_awaited_once()
