"""The trainer releases its CUDA cache before the producer sidecar reserves serve rings, CPU-only.

The sidecar is a separate process on the trainer rank's GPU, so it cannot use memory held in the
rank's caching allocator.
"""

from functools import cached_property

import pytest
import torch

pytest.importorskip("vllm", reason="the SkyRL sharded_rdt engines import vllm at module scope")

pytestmark = pytest.mark.vllm

import skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_trainer as trainer_mod  # noqa: E402
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_base import (  # noqa: E402
    GroupedWeightSource,
    ParamMeta,
)
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_trainer import (  # noqa: E402
    SkyRLShardedRDTTrainerInitInfo,
    SkyRLShardedRDTTrainerWeightTransferEngine,
    _RDTProducerServer,
)

GIB = 1 << 30


class _TinySource(GroupedWeightSource):
    def __init__(self):
        self._t = {"model.layers.0.w": torch.zeros(4), "model.layers.1.w": torch.zeros(4)}

    @cached_property
    def metadata(self):
        return [ParamMeta(n, t.dtype, tuple(t.shape)) for n, t in self._t.items()]

    def iter_full(self):
        yield from self._t.items()


class TestTrainerReleasesCacheBeforeSidecar:
    def test_cache_is_released_before_the_sidecar_spawns_and_the_worker_init(self, monkeypatch):
        order: list[str] = []
        cls = SkyRLShardedRDTTrainerWeightTransferEngine
        monkeypatch.setattr(trainer_mod, "_release_cuda_cache_for_sidecar", lambda rank: order.append("release"))
        monkeypatch.setattr(cls, "_spawn_server", lambda self, names: order.append("spawn"))
        monkeypatch.setattr(cls, "_all_gather_server_names", lambda self, w, r: order.append("barrier") or ["srv"])
        monkeypatch.setattr(cls, "_build_worker_init_info", lambda self, names: SkyRLShardedRDTTrainerInitInfo)

        class _Client:
            def init_weight_transfer_engine(self, payload):
                order.append("worker_init")

        monkeypatch.setattr(trainer_mod, "asdict", lambda x: {})
        info = SkyRLShardedRDTTrainerInitInfo(rank=0, num_consumers=8)
        cls.trainer_init(info, client=_Client(), source=_TinySource())
        assert order == ["release", "spawn", "barrier", "worker_init"]

    def test_release_empties_the_cache_and_reports(self, monkeypatch):
        state = {"reserved": 170 * GIB}
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: state["reserved"])
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 90 * GIB)

        def empty():
            state["reserved"] = 91 * GIB

        monkeypatch.setattr(torch.cuda, "empty_cache", empty)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (80 * GIB, 178 * GIB))
        info = trainer_mod._release_cuda_cache_for_sidecar(0)
        assert info == {
            "allocated_gib": 90.0,
            "reserved_before_gib": 170.0,
            "reserved_after_gib": 91.0,
            "device_free_gib": 80.0,
            "device_total_gib": 178.0,
        }

    def test_release_is_a_noop_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert trainer_mod._release_cuda_cache_for_sidecar(0) is None


class _Sized:
    """Stands in for a registered serve slot: the reservation only reads numel()."""

    def __init__(self, n):
        self.n = n

    def numel(self):
        return self.n


def test_an_oom_reserving_a_ring_names_the_cause(monkeypatch):
    import gc

    monkeypatch.setattr("torch.accelerator.current_device_index", lambda *a, **kw: 0, raising=False)
    monkeypatch.setattr(gc, "freeze", lambda: None)
    server = _RDTProducerServer(num_rdt_buffers=4, buffer_presize_gb=0.0, gather_lookahead=1)
    made: list[int] = []

    def new_buffer(nbytes):
        if len(made) == 2:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 512.00 MiB.")
        made.append(nbytes)
        return _Sized(nbytes)

    monkeypatch.setattr(server, "_new_serve_buffer", new_buffer)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (332 << 20, 178 * GIB))
    with pytest.raises(RuntimeError, match="empty_cache") as exc:
        server.reserve_serve_buffer(0, 300 << 20)
    assert isinstance(exc.value.__cause__, torch.OutOfMemoryError)
    assert "slot 2" in str(exc.value) and "GiB already held" in str(exc.value)
    server.shutdown()
