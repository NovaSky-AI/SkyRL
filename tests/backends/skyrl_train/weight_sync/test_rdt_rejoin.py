"""Rejoining a restarted inference engine to the RDT weight sync (Part 2, P6).

Re-initializing one consumer mid-run has a hard constraint: the driver is the caller,
and the training ranks are wherever their own step left them. So the payload the
consumer needs must be obtainable WITHOUT a collective, or the fetch deadlocks against
ranks that are not at a matching call.

``_build_worker_init_info`` turns out to be pure over state ``trainer_init`` already
retained -- the only thing it was missing is the producer server names, which
``trainer_init`` computed and then dropped. Caching them is the whole mechanism; these
tests pin that it is genuinely collective-free and that the payload is stable, because
"static for the run" is what justifies fetching it once.

The second half covers the consumer's own idempotency. ``_producer_actors`` /
``_produce_methods`` are append-only lists that every owner index is a POSITION in, so
a second init on a live engine would double their length and shift every index --
silently, with the symptom appearing later as one consumer pulling another's slice.

CPU-only: the engine and trainer objects are built field-by-field, since their real
inits want Ray, NCCL and a GPU.

Run:
    uv run --extra dev --extra fsdp pytest tests/backends/skyrl_train/weight_sync/test_rdt_rejoin.py
"""

from types import SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.weight_sync.rdt_send import RdtWeightSyncSender
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_base import ParamMeta
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_trainer import (
    ShardedRDTTrainerInitInfo,
    ShardedRDTTrainerWeightTransferEngine,
)


def _engine(server_names=("srv-a", "srv-b"), num_consumers=4):
    """A trainer engine carrying only the retained state the payload is built from."""
    import torch

    e = ShardedRDTTrainerWeightTransferEngine.__new__(ShardedRDTTrainerWeightTransferEngine)
    e._meta = [
        ParamMeta(name="model.embed_tokens.weight", dtype=torch.bfloat16, shape=(8, 4)),
        ParamMeta(name="model.layers.0.mlp.w.weight", dtype=torch.bfloat16, shape=(4, 4)),
        ParamMeta(name="lm_head.weight", dtype=torch.bfloat16, shape=(8, 4)),
    ]
    e._groups = [["model.embed_tokens.weight"], ["model.layers.0.mlp.w.weight"], ["lm_head.weight"]]
    e._group_owners = [[0], [1], [0]]
    e._name_ep_rank = None
    e._producer_ep_ranks = None
    # rank 0 IS the sender (`is_sender` is derived from rank on the base class, not a
    # constructor field), so this engine is the one that drives the control plane.
    e._init_info = ShardedRDTTrainerInitInfo(
        rank=0,
        num_consumers=num_consumers,
        trainer_actor_namespace="ns",
    )
    e.is_sender = True
    e._server_names = list(server_names) if server_names is not None else None
    return e


class TestWorkerInitPayload:
    def test_the_payload_carries_the_cached_producer_names(self):
        """The one thing trainer_init used to drop. Without it a rejoin needs another
        all-gather, which the driver cannot participate in."""
        e = _engine(server_names=("srv-a", "srv-b"))
        payload = e.get_worker_init_payload()
        assert payload["trainer_actor_names"] == ["srv-a", "srv-b"]
        assert payload["trainer_actor_namespace"] == "ns"

    def test_the_payload_describes_the_provisioned_geometry(self):
        e = _engine(num_consumers=4)
        payload = e.get_worker_init_payload()
        assert payload["num_consumers"] == 4
        assert payload["group_lens"] == [1, 1, 1]
        assert payload["group_owners"] == [[0], [1], [0]]
        assert payload["names"] == [m.name for m in e._meta]
        assert payload["dtype_names"] == ["bfloat16"] * 3
        assert payload["shapes"] == [[8, 4], [4, 4], [8, 4]]

    def test_the_payload_is_stable_across_calls(self):
        """What justifies fetching it ONCE and caching it on the driver. If it drifted,
        a rejoining engine would bake against a different picture than its peers."""
        e = _engine()
        assert e.get_worker_init_payload() == e.get_worker_init_payload()

    def test_it_touches_no_collective(self, monkeypatch):
        """The property that makes a mid-run rejoin possible at all: the driver calls
        this while every rank sits in its own training step, so any torch.distributed
        or Ray call here would deadlock rather than fail."""
        import torch

        def _boom(*a, **kw):
            raise AssertionError("get_worker_init_payload must not touch torch.distributed")

        for name in ("all_gather_object", "barrier", "all_gather", "broadcast"):
            if hasattr(torch.distributed, name):
                monkeypatch.setattr(torch.distributed, name, _boom)
        e = _engine()
        assert e.get_worker_init_payload()["num_consumers"] == 4

    def test_it_refuses_before_trainer_init(self):
        """Better a clear error than a payload with no producers in it, which the
        consumer would only fail on much later, inside its bake."""
        e = _engine(server_names=None)
        with pytest.raises(RuntimeError, match="requires trainer_init"):
            e.get_worker_init_payload()


class TestSenderExposure:
    @staticmethod
    def _sender(engine):
        s = RdtWeightSyncSender.__new__(RdtWeightSyncSender)
        s._engine = engine
        return s

    def test_the_sender_forwards_the_payload(self):
        s = self._sender(_engine())
        payload = s.get_worker_init_info()
        assert payload is not None and payload["trainer_actor_names"] == ["srv-a", "srv-b"]

    def test_a_non_sender_rank_returns_none(self):
        """None means "ask rank 0", not an error -- every rank holds a sender, but only
        rank 0's engine drives the control plane."""
        e = _engine()
        e.is_sender = False
        assert self._sender(e).get_worker_init_info() is None

    def test_an_uninitialized_sender_returns_none(self):
        assert self._sender(None).get_worker_init_info() is None


class TestConsumerInitIdempotency:
    """A rejoining engine calls ``init_transfer_engine`` a SECOND time.

    ``_producer_actors``/``_produce_methods`` are append-only and every owner index is
    a position in them, so without a reset the second init doubles their length and
    shifts every index -- a silent mis-route, not an error. Driven here as the pure
    list discipline rather than through the real init, which needs Ray and a GPU.
    """

    def test_re_binding_replaces_rather_than_appends(self):
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_engine import (
            ShardedRDTWeightTransferEngine,
        )

        e = ShardedRDTWeightTransferEngine.__new__(ShardedRDTWeightTransferEngine)
        e._producer_actors = ["actor-a", "actor-b"]
        e._produce_methods = ["m-a", "m-b"]

        # The reset that `init_transfer_engine` now performs before re-binding.
        e._producer_actors = []
        e._produce_methods = []
        for name in ("actor-a", "actor-b"):
            e._producer_actors.append(name)
            e._produce_methods.append(f"m-{name[-1]}")

        assert e._producer_actors == ["actor-a", "actor-b"]
        assert e._produce_methods == ["m-a", "m-b"]

    def test_the_reset_is_present_in_the_init_path(self):
        """Guards the actual product line rather than a restatement of it: without the
        reset, a rejoin doubles the producer lists. Asserted on the source because the
        surrounding init needs Ray, CUDA and a live vLLM worker.

        `_resolve_producers` is where the binding happens (`init_transfer_engine`
        delegates to it), and it is reached on every init -- including a rejoin's."""
        import inspect

        from skyrl.backends.skyrl_train.weight_sync import sharded_rdt_engine

        cls = sharded_rdt_engine.ShardedRDTWeightTransferEngine
        assert "_resolve_producers" in inspect.getsource(cls.init_transfer_engine), (
            "init_transfer_engine no longer delegates producer binding to _resolve_producers; "
            "re-point this test at whatever does the binding now."
        )
        src = inspect.getsource(cls._resolve_producers)
        reset = src.index("self._producer_actors = []")
        append = src.index("self._producer_actors.append(")
        assert reset < append, "the producer lists must be cleared BEFORE they are re-bound"
        assert "self._produce_methods = []" in src


def test_the_driver_caches_the_payload_at_init_weight_sync_state(monkeypatch):
    """The fetch happens once, from rank 0, and lands on the supervisor.

    Fetching per rejoin instead would put a round trip into the training ranks on the
    sync boundary's critical path; fetching before ``init_weight_sync_state`` would find
    nothing, since the producers have not rendezvoused yet.
    """
    from skyrl.backends.skyrl_train.workers import worker_dispatch as wd

    calls = []

    class _Group:
        def async_run_ray_method(self, mode, method, *a, **kw):
            calls.append(method)
            return [f"ref:{method}"]

    class _Supervisor:
        enabled = True

        def __init__(self):
            self.payload = "unset"

        def set_rdt_worker_init(self, p):
            self.payload = p

    sup = _Supervisor()
    client = SimpleNamespace(engine_supervisor=sup)

    d = wd.WorkerDispatch.__new__(wd.WorkerDispatch)
    d._actor_groups = {"policy": _Group()}
    d.cfg = SimpleNamespace(generator=SimpleNamespace(inference_engine=SimpleNamespace()))
    monkeypatch.setattr(wd.ray, "get", lambda refs: [{"num_consumers": 4}])

    d.init_weight_sync_state(client)

    assert calls == ["init_weight_sync_state", "get_rdt_worker_init_info"], calls
    assert sup.payload == {"num_consumers": 4}


def test_a_failed_fetch_does_not_fail_the_run(monkeypatch):
    """Losing the payload costs the ability to REJOIN engines, not the ability to
    survive losing them -- so it must not turn a healthy start into a crash."""
    from skyrl.backends.skyrl_train.workers import worker_dispatch as wd

    class _Group:
        def async_run_ray_method(self, mode, method, *a, **kw):
            return ["ref"]

    class _Supervisor:
        enabled = True
        payload = "unset"

        def set_rdt_worker_init(self, p):
            self.payload = p

    sup = _Supervisor()
    calls = {"n": 0}

    def _get(refs):
        calls["n"] += 1
        if calls["n"] == 1:
            return [None]  # init_weight_sync_state
        raise RuntimeError("actor died")

    d = wd.WorkerDispatch.__new__(wd.WorkerDispatch)
    d._actor_groups = {"policy": _Group()}
    d.cfg = SimpleNamespace(generator=SimpleNamespace(inference_engine=SimpleNamespace()))
    monkeypatch.setattr(wd.ray, "get", _get)

    d.init_weight_sync_state(SimpleNamespace(engine_supervisor=sup))
    assert sup.payload == "unset"


def test_no_supervisor_means_no_fetch(monkeypatch):
    """Part 1 and non-FT runs must not pay for a Ray round trip they cannot use."""
    from skyrl.backends.skyrl_train.workers import worker_dispatch as wd

    calls = []

    class _Group:
        def async_run_ray_method(self, mode, method, *a, **kw):
            calls.append(method)
            return ["ref"]

    d = wd.WorkerDispatch.__new__(wd.WorkerDispatch)
    d._actor_groups = {"policy": _Group()}
    d.cfg = SimpleNamespace(generator=SimpleNamespace(inference_engine=SimpleNamespace()))
    monkeypatch.setattr(wd.ray, "get", lambda refs: [None])

    d.init_weight_sync_state(SimpleNamespace(engine_supervisor=None))
    assert calls == ["init_weight_sync_state"]
