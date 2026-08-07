"""Tests for prefix cache reset behaviour in `PolicyWorker.broadcast_to_inference_engines`

When ``WeightTransferSender.handles_prefix_cache_reset`` is ``True``,
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
        policy=SimpleNamespace(megatron_config=SimpleNamespace(lora_config=SimpleNamespace(merge_lora=False))),
    )
    worker._weight_transfer_sender = AsyncMock()
    # AsyncMock would make this attribute a coroutine; the production code reads it as a
    # plain bool, so set it explicitly.
    worker._weight_transfer_sender.handles_prefix_cache_reset = handles_prefix_cache_reset
    worker._weight_transfer_sender._inference_client = None
    worker.weight_extractor = MagicMock()
    # FSDP dereferences self.model.model before the LoRA branch, so it must exist even
    # though _is_lora is False and the resulting peft_model goes unused.
    worker.model = SimpleNamespace(model=SimpleNamespace())

    @contextmanager
    def _noop_ctx():
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
    worker._weight_transfer_sender.send_chunks.assert_awaited_once()
    assert worker._weight_transfer_sender.send_chunks.await_args.kwargs["reset_prefix_cache"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["fsdp", pytest.param("megatron", marks=pytest.mark.megatron)])
async def test_worker_resets_prefix_cache_when_sender_does_not(strategy, monkeypatch):
    worker_cls = get_worker_cls(strategy)

    _patch_collectives(monkeypatch)
    worker = _make_worker(worker_cls, handles_prefix_cache_reset=False)
    client = AsyncMock()

    await worker.broadcast_to_inference_engines(client, _ie_cfg())

    client.reset_prefix_cache.assert_awaited_once_with(reset_running_requests=True)
    worker._weight_transfer_sender.send_chunks.assert_awaited_once()


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
    assert worker._weight_transfer_sender.send_chunks.await_args.kwargs["reset_prefix_cache"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["fsdp", pytest.param("megatron", marks=pytest.mark.megatron)])
async def test_worker_closes_deserialized_remote_client(strategy, monkeypatch):
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )

    worker_cls = get_worker_cls(strategy)
    _patch_collectives(monkeypatch)
    worker = _make_worker(worker_cls, handles_prefix_cache_reset=True)
    client = RemoteInferenceClient(
        proxy_url="http://router",
        server_urls=["http://server"],
        data_parallel_size=1,
    )
    client.teardown = AsyncMock()

    await worker.broadcast_to_inference_engines(client, _ie_cfg())

    client.teardown.assert_awaited_once()
