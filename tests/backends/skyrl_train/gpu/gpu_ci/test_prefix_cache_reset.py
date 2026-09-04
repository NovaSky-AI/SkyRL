"""Tests for prefix cache reset behaviour in ``PolicyWorker.broadcast_to_inference_engines``.

The worker calls the trainer engine's ``send_weights()`` inside a memory bracket
whose three decisions come from ``getattr`` capability probes on the engine
(``skyrl_handles_prefix_cache_reset``,
``skyrl_force_disable_expandable_segments``, ``skyrl_empty_cache_after_send``).
Two of the four engines are vLLM's own classes and cannot declare SkyRL
attributes, so the *absence* of a flag is the common case and must mean the
default.

An engine declaring ``skyrl_handles_prefix_cache_reset`` resets the cache itself,
at the right point in its own pause/update sequence, and the worker must skip its
own concurrent reset.

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


class _FakeEngine:
    """A trainer engine with only what the worker's send bracket touches.

    A plain class, not a Mock: a Mock answers every attribute, so it could never
    exercise the probe defaults.
    """

    def __init__(self, **flags) -> None:
        self.sends = 0
        self.reset_prefix_cache_told = None
        for name, value in flags.items():
            setattr(self, f"skyrl_{name}", value)

    def send_weights(self) -> None:
        self.sends += 1


class _DeltaLikeEngine(_FakeEngine):
    """Also implements the per-round setter, as the delta engine does."""

    def skyrl_set_reset_prefix_cache(self, reset: bool) -> None:
        self.reset_prefix_cache_told = reset


# TODO (sumanthrh): Ideally we avoid all this mocking with an easier way to construct dummy policy workers
def _make_worker(worker_cls, engine):
    """Build a policy worker with just enough state for broadcast_to_inference_engines."""
    worker = worker_cls.__new__(worker_cls)
    worker._is_lora = False
    worker.cfg = SimpleNamespace(
        fully_async=SimpleNamespace(enabled=False, clear_kv_cache_on_weight_sync=False),
        placement=SimpleNamespace(colocate_all=False),
        policy=SimpleNamespace(megatron_config=SimpleNamespace(lora_config=SimpleNamespace(merge_lora=False))),
    )
    worker._weight_sync_engine = engine
    worker._weight_sync_inference_client = None
    # FSDP dereferences self.model.model before the LoRA branch, so it must exist even
    # though _is_lora is False and the resulting peft_model goes unused.
    worker.model = SimpleNamespace(model=SimpleNamespace())

    @contextmanager
    def _noop_ctx(*args, **kwargs):
        # The worker calls this with force=<engine capability>.
        yield

    worker._expandable_segments_disabled_for_sync = _noop_ctx
    return worker


def _patch_collectives(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *a, **k: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def _ie_cfg(enable_prefix_caching: bool = True):
    return SimpleNamespace(enable_prefix_caching=enable_prefix_caching, model_dtype="bfloat16")


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


STRATEGIES = ["fsdp", pytest.param("megatron", marks=pytest.mark.megatron)]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", STRATEGIES)
async def test_worker_skips_prefix_cache_reset_when_engine_handles_it(strategy, monkeypatch):
    _patch_collectives(monkeypatch)
    engine = _DeltaLikeEngine(handles_prefix_cache_reset=True)
    worker = _make_worker(get_worker_cls(strategy), engine)
    client = AsyncMock()

    await worker.broadcast_to_inference_engines(client, _ie_cfg())

    client.reset_prefix_cache.assert_not_awaited()
    # The sync still happens, and the engine is still told the cache needs
    # resetting so it can do it at the right point in its own sequence.
    assert engine.sends == 1
    assert engine.reset_prefix_cache_told is True


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", STRATEGIES)
async def test_worker_resets_prefix_cache_when_engine_does_not(strategy, monkeypatch):
    _patch_collectives(monkeypatch)
    # No flags at all -- the shape of vLLM's own NCCL / IPC trainer engines.
    engine = _FakeEngine()
    worker = _make_worker(get_worker_cls(strategy), engine)
    client = AsyncMock()

    await worker.broadcast_to_inference_engines(client, _ie_cfg())

    client.reset_prefix_cache.assert_awaited_once_with(reset_running_requests=True)
    assert engine.sends == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", STRATEGIES)
async def test_no_reset_when_prefix_caching_disabled(strategy, monkeypatch):
    _patch_collectives(monkeypatch)
    engine = _DeltaLikeEngine()
    worker = _make_worker(get_worker_cls(strategy), engine)
    client = AsyncMock()

    await worker.broadcast_to_inference_engines(client, _ie_cfg(enable_prefix_caching=False))

    client.reset_prefix_cache.assert_not_awaited()
    assert engine.reset_prefix_cache_told is False


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", STRATEGIES)
async def test_expandable_segments_force_comes_from_the_engine(strategy, monkeypatch):
    """sharded_rdt asks for the toggle unconditionally; everything else leaves it
    to ``colocate_all``. Default False for an engine that declares nothing."""
    _patch_collectives(monkeypatch)
    seen = []

    for engine in (_FakeEngine(), _FakeEngine(force_disable_expandable_segments=True)):
        worker = _make_worker(get_worker_cls(strategy), engine)

        @contextmanager
        def _record(force=False):
            seen.append(force)
            yield

        worker._expandable_segments_disabled_for_sync = _record
        await worker.broadcast_to_inference_engines(AsyncMock(), _ie_cfg(enable_prefix_caching=False))

    assert seen == [False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", STRATEGIES)
async def test_empty_cache_after_send_defaults_on_and_can_be_declined(strategy, monkeypatch):
    """The default must be True -- that is what vLLM's own engines get."""
    _patch_collectives(monkeypatch)
    for engine, expected in ((_FakeEngine(), 2), (_FakeEngine(empty_cache_after_send=False), 1)):
        calls = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
        worker = _make_worker(get_worker_cls(strategy), engine)
        await worker.broadcast_to_inference_engines(AsyncMock(), _ie_cfg(enable_prefix_caching=False))
        # One before the send always; one after only if the engine allows it.
        assert len(calls) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", STRATEGIES)
async def test_lora_sync_skips_the_engine_entirely(strategy, monkeypatch):
    """A LoRA adapter sync writes safetensors and calls the LoRA route; there is
    no tensor transfer, so it must not drive the engine."""
    _patch_collectives(monkeypatch)
    engine = _FakeEngine()
    worker = _make_worker(get_worker_cls(strategy), engine)
    worker._is_lora = True
    worker.model = SimpleNamespace(model=SimpleNamespace(peft_config={"default": {}}))
    worker._resolve_lora_sync_target = lambda model_id: ("adapter", "/tmp/lora")
    saved = MagicMock()
    worker._save_lora_adapters_and_sync = AsyncMock(side_effect=saved)

    client = AsyncMock()
    await worker.broadcast_to_inference_engines(client, _ie_cfg())

    assert engine.sends == 0
    worker._save_lora_adapters_and_sync.assert_awaited_once()
    # A LoRA sync still invalidates the prefix cache: the adapter changes what
    # the cached prefixes decode to.
    client.reset_prefix_cache.assert_awaited_once_with(reset_running_requests=True)
