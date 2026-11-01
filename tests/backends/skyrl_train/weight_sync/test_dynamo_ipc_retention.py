"""CPU lifetime checks for a native packed producer after ambiguous Dynamo ACKs."""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("vllm")
pytestmark = pytest.mark.vllm

from skyrl.backends.skyrl_train.weight_sync import (  # noqa: E402
    ipc_lifetime,
    weight_senders,
)


class _Buffer:
    pass


def test_registry_membership_compares_identity_without_releasing_producers():
    class Producer:
        def __eq__(self, other):
            raise AssertionError("GPU producer membership must not compare tensor values")

    registry = ipc_lifetime.FailedIPCProducerRegistry()
    producer = Producer()
    other = Producer()
    registry.retain(producer)
    assert producer in registry
    assert other not in registry
    assert len(registry) == 1


class _StrictClient:
    fail_closed_transfers = True

    def __init__(self):
        self.start_weight_update = Mock()
        self.update_weights = Mock()
        self.finish_weight_update = Mock()


def _engine(monkeypatch, client):
    from vllm.distributed.weight_transfer import ipc_engine

    refs, released = [], []

    def producer(**kwargs):
        buffer = _Buffer()
        refs.append(weakref.ref(buffer))
        try:
            yield SimpleNamespace(
                names=["w"],
                dtype_names=["float32"],
                shapes=[[1]],
                ipc_handle={"GPU-test": ("handle",)},
                tensor_sizes=[4],
            )
        finally:
            released.append(True)

    monkeypatch.setattr(ipc_engine, "packed_ipc_producer", producer)
    monkeypatch.setattr(ipc_lifetime, "FAILED_IPC_PRODUCERS", ipc_lifetime.FailedIPCProducerRegistry())
    _, cls = weight_senders.get_skyrl_ipc_trainer()
    monkeypatch.setattr(cls, "_all_gather_and_merge_handles", lambda self, handles: handles)
    monkeypatch.setattr(cls, "_post_send_sync", staticmethod(lambda: None))
    engine = object.__new__(cls)
    engine.client = client
    engine.source = []
    engine.is_sender = True
    engine.packed = True
    engine.gpu_uuid = "GPU-test"
    engine.packed_buffer_size_bytes = 16
    return engine, refs, released


def test_ambiguous_dynamo_failure_retains_buffer_even_after_engine_collection(monkeypatch):
    client = _StrictClient()
    client.update_weights.side_effect = RuntimeError("uncertain install")
    engine, refs, released = _engine(monkeypatch, client)
    with pytest.raises(RuntimeError, match="uncertain"):
        engine.send_weights()
    client.update_weights.side_effect = None
    assert refs[0]() is not None
    assert released == []
    assert len(ipc_lifetime.FAILED_IPC_PRODUCERS) == 1
    engine_ref = weakref.ref(engine)
    del engine
    gc.collect()
    assert engine_ref() is None
    assert refs[0]() is not None
    assert released == []


def test_uncertain_ipc_engine_cannot_start_another_session(monkeypatch):
    client = _StrictClient()
    client.update_weights.side_effect = RuntimeError("uncertain install")
    engine, _refs, _released = _engine(monkeypatch, client)
    with pytest.raises(RuntimeError, match="uncertain"):
        engine.send_weights()
    client.update_weights.side_effect = None
    with pytest.raises(RuntimeError, match="restart"):
        engine.send_weights()
    client.start_weight_update.assert_called_once()


def test_successful_dynamo_send_releases_the_native_producer(monkeypatch):
    client = _StrictClient()
    engine, refs, released = _engine(monkeypatch, client)
    engine.send_weights()
    gc.collect()
    assert refs[0]() is None
    assert released == [True]
    assert len(ipc_lifetime.FAILED_IPC_PRODUCERS) == 0
    client.finish_weight_update.assert_called_once()


def test_direct_ipc_keeps_native_failure_behavior(monkeypatch):
    client = SimpleNamespace(
        fail_closed_transfers=False,
        start_weight_update=Mock(),
        update_weights=Mock(side_effect=RuntimeError("direct failed")),
        finish_weight_update=Mock(),
    )
    engine, refs, released = _engine(monkeypatch, client)
    with pytest.raises(RuntimeError, match="direct failed"):
        engine.send_weights()
    gc.collect()
    assert refs[0]() is None
    assert released == [True]
    assert len(ipc_lifetime.FAILED_IPC_PRODUCERS) == 0


def test_strict_ipc_rejects_unpacked_buffers_before_start(monkeypatch):
    client = _StrictClient()
    engine, _refs, _released = _engine(monkeypatch, client)
    engine.packed = False
    with pytest.raises(RuntimeError, match="packed"):
        engine.send_weights()
    client.start_weight_update.assert_not_called()


def test_failure_before_export_latches_without_a_quarantined_producer(monkeypatch):
    client = _StrictClient()
    client.start_weight_update.side_effect = RuntimeError("start failed")
    engine, refs, _released = _engine(monkeypatch, client)
    with pytest.raises(RuntimeError, match="start failed"):
        engine.send_weights()
    assert refs == []
    assert len(ipc_lifetime.FAILED_IPC_PRODUCERS) == 0
    with pytest.raises(RuntimeError, match="restart"):
        engine.send_weights()
