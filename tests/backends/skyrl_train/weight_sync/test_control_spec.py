"""Provider-independent trainer construction and lifecycle contracts."""

import pickle
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)


@pytest.mark.asyncio
async def test_direct_control_spec_is_picklable_and_builds_existing_client():
    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        SkyrlWeightSyncClient,
    )

    client = RemoteInferenceClient("http://router", ["http://worker"], 1)
    spec = await client.weight_sync_control_spec("GPU-test")
    spec = pickle.loads(pickle.dumps(spec))
    assert spec.fail_closed_transfers is False
    assert not hasattr(spec, "validate_sender")
    blocking = spec.build_client(None)
    try:
        assert isinstance(blocking, SkyrlWeightSyncClient)
        assert blocking.fail_closed_transfers is False
    finally:
        blocking.close()
    assert client.owns_weight_update_lifecycle is False
    assert not hasattr(client, "owns_cache_invalidation")
    assert not hasattr(client, "weight_update_transaction")


@pytest.mark.asyncio
@pytest.mark.vllm
async def test_third_provider_controls_worker_setup_and_update_bracket(monkeypatch):
    pytest.importorskip("vllm")
    import torch

    from skyrl.backends.skyrl_train.weight_sync import weight_senders
    from skyrl.backends.skyrl_train.workers.worker import Worker
    from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

    events = []
    spec = object()

    class ThirdProvider(RemoteInferenceClient):
        owns_weight_update_lifecycle = True

        async def get_world_size(self):
            return 1, 1

        async def weight_sync_control_spec(self, device_uuid):
            assert device_uuid == "GPU-third"
            events.append("spec")
            return spec

        @asynccontextmanager
        async def weight_update_transaction(self):
            events.append("pause")
            yield
            events.append("commit")

    client = ThirdProvider("http://third", ["http://third-worker"], 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="GPU-third"))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    build = Mock(return_value=object())
    monkeypatch.setattr(weight_senders, "build_trainer_engine", build)

    async def run_sync(fn, **kwargs):
        return fn(**kwargs)

    worker = Worker.__new__(Worker)
    worker.cfg = SimpleNamespace(
        placement=SimpleNamespace(colocate_all=False), policy=SimpleNamespace(model=SimpleNamespace(path="model"))
    )
    worker._weight_sync_thread = run_sync
    worker._build_weight_source = Mock()
    worker._build_draft_weight_source = Mock()
    await worker.init_weight_sync_state(client, SimpleNamespace())
    assert build.call_args.kwargs["control_spec"] is spec
    assert worker._reset_prefix_cache_task(client, SimpleNamespace()) is None

    dispatch = WorkerDispatch.__new__(WorkerDispatch)
    dispatch.cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            strategy="fsdp", policy=SimpleNamespace(model=SimpleNamespace(lora=SimpleNamespace(rank=0)))
        )
    )
    dispatch.colocate_all = False
    dispatch._inference_engine_client = client
    dispatch._prepare_for_weight_sync = AsyncMock()
    dispatch.ensure_active_adapter = Mock()
    dispatch._broadcast_to_inference_engines = Mock(side_effect=lambda *args, **kwargs: events.append("broadcast"))
    dispatch._finish_weight_sync = Mock()
    await dispatch.save_weights_for_sampler()
    assert events == ["spec", "pause", "broadcast", "commit"]


@pytest.mark.parametrize("fail_at", ["factory", "post_init"])
@pytest.mark.parametrize("provider", ["direct", "third"])
@pytest.mark.vllm
def test_factory_closes_every_provider_client_after_failure(monkeypatch, fail_at, provider):
    pytest.importorskip("vllm")
    from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

    from skyrl.backends.skyrl_train.weight_sync import weight_senders

    client = SimpleNamespace(close=Mock())
    spec = SimpleNamespace(fail_closed_transfers=False, build_client=Mock(return_value=client))
    if provider == "direct":
        monkeypatch.setattr(weight_senders.DirectWeightSyncControlSpec, "build_client", lambda *args: client)

    class BrokenEngine:
        def __setattr__(self, name, value):
            raise RuntimeError("post init failed")

    def trainer_init(*args, **kwargs):
        if fail_at == "factory":
            raise RuntimeError("factory failed")
        return BrokenEngine()

    monkeypatch.setattr(WeightTransferTrainerFactory, "trainer_init", trainer_init)
    monkeypatch.setattr(weight_senders, "_build_sender_init_info", lambda **kwargs: (object(), None))
    config = SimpleNamespace(weight_sync_backend="nccl", model_dtype="bfloat16", speculative_config=None)
    with pytest.raises(RuntimeError, match="failed"):
        weight_senders.build_trainer_engine(
            ie_cfg=config,
            colocate_all=False,
            rank=0,
            inference_world_size=1,
            source_factory=lambda *args: object(),
            draft_source_factory=None,
            server_urls=["http://worker"],
            data_parallel_size=1,
            control_spec=spec if provider == "third" else None,
        )
    client.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.vllm
@pytest.mark.parametrize("engine_owns_reset", [False, True])
@pytest.mark.parametrize("provider_owns_lifecycle", [False, True])
async def test_native_engine_cache_reset_capability_is_preserved(
    monkeypatch, engine_owns_reset, provider_owns_lifecycle
):
    pytest.importorskip("vllm")
    import torch

    from skyrl.backends.skyrl_train.workers.worker import Worker

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    engine = SimpleNamespace(
        skyrl_handles_prefix_cache_reset=engine_owns_reset,
        skyrl_force_disable_expandable_segments=False,
        skyrl_empty_cache_after_send=True,
        send_weights=Mock(),
    )
    worker = SimpleNamespace(
        cfg=SimpleNamespace(placement=SimpleNamespace(colocate_all=False)),
        _weight_sync_engine=engine,
        _should_reset_prefix_cache=lambda config: True,
        _weight_sync_thread=AsyncMock(),
        _expandable_segments_disabled_for_sync=lambda **kwargs: nullcontext(),
    )
    worker._reset_prefix_cache_task = lambda client, config: Worker._reset_prefix_cache_task(worker, client, config)
    client = SimpleNamespace(
        owns_weight_update_lifecycle=provider_owns_lifecycle,
        reset_prefix_cache=AsyncMock(),
    )
    await Worker._sync_weights_to_inference_engines(worker, client, SimpleNamespace())
    worker._weight_sync_thread.assert_awaited_once_with(engine.send_weights)
    if engine_owns_reset or provider_owns_lifecycle:
        client.reset_prefix_cache.assert_not_awaited()
    else:
        client.reset_prefix_cache.assert_awaited_once_with(reset_running_requests=True)
