"""Tests for prefix cache reset behaviour in `PolicyWorker.broadcast_to_inference_engines`

The worker's send path goes through ``WeightTransferSender.send``, whose default
implementation materializes the chunk stream and calls ``send_chunks``; senders that
pull instead (sharded_rdt) override ``send``. When
``WeightTransferSender.handles_prefix_cache_reset`` is ``True``,
the sender will resets the cache itself as part of its own pause/update sequence
(``DeltaWeightTransferSender`` does this inside ``_apply_receiver_update``), and the worker must skip
this.

uv run --extra dev --extra fsdp -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_prefix_cache_reset.py -m "not megatron"
uv run --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_prefix_cache_reset.py -m "megatron"
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Type
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase


# TODO (sumanthrh): Ideally we avoid all this mocking with an easier way to construct dummy policy workers
def _make_worker(worker_cls, handles_prefix_cache_reset: bool):
    """Build a policy worker with just enough state for broadcast_to_inference_engines."""
    worker = worker_cls.__new__(worker_cls)
    worker._is_lora = False
    worker.cfg = SimpleNamespace(
        fully_async=SimpleNamespace(enabled=False, clear_kv_cache_on_weight_sync=False),
        placement=SimpleNamespace(colocate_all=False),
        policy=SimpleNamespace(megatron_config=SimpleNamespace(lora_config=SimpleNamespace(merge_lora=False))),
    )
    worker._weight_transfer_sender = AsyncMock()
    # AsyncMock would make this attribute a coroutine; the production code reads it as a
    # plain bool, so set it explicitly.
    worker._weight_transfer_sender.handles_prefix_cache_reset = handles_prefix_cache_reset
    # Same reason: the worker reads these as plain bools to decide the
    # expandable-segments toggle and the post-send empty_cache.
    worker._weight_transfer_sender.force_disable_expandable_segments = False
    worker._weight_transfer_sender.empty_cache_after_send = True
    worker._weight_transfer_sender._inference_client = None
    worker.weight_extractor = MagicMock()
    # FSDP dereferences self.model.model before the LoRA branch, so it must exist even
    # though _is_lora is False and the resulting peft_model goes unused.
    worker.model = SimpleNamespace(model=SimpleNamespace())

    @contextmanager
    def _noop_ctx(*args, **kwargs):
        # The worker calls this with force=<sender flag>.
        yield

    worker._expandable_segments_disabled_for_sync = _noop_ctx
    return worker


def _patch_collectives(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *a, **k: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def _ie_cfg():
    return SimpleNamespace(enable_prefix_caching=True, model_dtype="bfloat16")


def get_worker_cls(strategy: str) -> Type[PolicyWorkerBase]:
    if strategy == "fsdp":
        from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import (
            FSDPPolicyWorkerBase,
        )

        return FSDPPolicyWorkerBase
    elif strategy == "megatron":
        from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
            MegatronPolicyWorkerBase,
        )

        return MegatronPolicyWorkerBase
    else:
        raise ValueError(f"Invalid worker cls: {strategy}")


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["fsdp", pytest.param("megatron", marks=pytest.mark.megatron)])
async def test_worker_skips_prefix_cache_reset_when_sender_handles_it(strategy, monkeypatch):
    worker_cls = get_worker_cls(strategy)

    _patch_collectives(monkeypatch)
    worker = _make_worker(worker_cls, handles_prefix_cache_reset=True)
    client = AsyncMock()

    await worker.broadcast_to_inference_engines(client, _ie_cfg())

    client.reset_prefix_cache.assert_not_awaited()
    # The sync still happens, and the sender is still told the cache needs resetting so
    # it can do it at the right point in its own sequence.
    worker._weight_transfer_sender.send.assert_awaited_once()
    assert worker._weight_transfer_sender.send.await_args.kwargs["reset_prefix_cache"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["fsdp", pytest.param("megatron", marks=pytest.mark.megatron)])
async def test_worker_resets_prefix_cache_when_sender_does_not(strategy, monkeypatch):
    worker_cls = get_worker_cls(strategy)

    _patch_collectives(monkeypatch)
    worker = _make_worker(worker_cls, handles_prefix_cache_reset=False)
    client = AsyncMock()

    await worker.broadcast_to_inference_engines(client, _ie_cfg())

    client.reset_prefix_cache.assert_awaited_once_with(reset_running_requests=True)
    worker._weight_transfer_sender.send.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["fsdp", pytest.param("megatron", marks=pytest.mark.megatron)])
async def test_no_reset_when_prefix_caching_disabled(strategy, monkeypatch):
    worker_cls = get_worker_cls(strategy)

    _patch_collectives(monkeypatch)
    worker = _make_worker(worker_cls, handles_prefix_cache_reset=False)
    client = AsyncMock()
    ie_cfg = SimpleNamespace(enable_prefix_caching=False, model_dtype="bfloat16")

    await worker.broadcast_to_inference_engines(client, ie_cfg)

    client.reset_prefix_cache.assert_not_awaited()
    assert worker._weight_transfer_sender.send.await_args.kwargs["reset_prefix_cache"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["fsdp", pytest.param("megatron", marks=pytest.mark.megatron)])
@pytest.mark.parametrize("handles_reset", [False, True])
@pytest.mark.parametrize("enable_prefix_caching", [False, True])
@pytest.mark.parametrize("clear_kv_cache", [False, True])
async def test_async_running_request_reset(strategy, handles_reset, enable_prefix_caching, clear_kv_cache, monkeypatch):
    _patch_collectives(monkeypatch)
    worker = _make_worker(get_worker_cls(strategy), handles_prefix_cache_reset=handles_reset)
    worker.cfg.fully_async.enabled = True
    worker.cfg.fully_async.clear_kv_cache_on_weight_sync = clear_kv_cache
    client = AsyncMock()
    ie_cfg = SimpleNamespace(enable_prefix_caching=enable_prefix_caching, model_dtype="bfloat16")

    await worker.broadcast_to_inference_engines(client, ie_cfg)

    # The sender gets the same policy regardless of who owns the reset, and
    # running requests require invalidation even without reusable prefix blocks.
    assert worker._weight_transfer_sender.send.await_args.kwargs["reset_prefix_cache"] is clear_kv_cache
    if clear_kv_cache and not handles_reset:
        client.reset_prefix_cache.assert_awaited_once_with(reset_running_requests=True)
    else:
        client.reset_prefix_cache.assert_not_awaited()
