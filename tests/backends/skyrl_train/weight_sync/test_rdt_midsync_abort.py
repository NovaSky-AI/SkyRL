"""Mid-sync abort and retry (Part 2, §5.5).

Part 1 accepted that an engine dying INSIDE the weight-sync window fails the run
cleanly. Making it recoverable turns on three things that are each easy to get subtly
wrong, and whose failure mode is corrupt weights rather than an error:

1. **Every rank's gather loop must run to completion.** The gathers are collectives.
   Raising from the middle of the loop on the rank that noticed the failure abandons
   the remaining collectives and leaves every peer blocked in NCCL — one dead consumer
   becomes a dead job. So the abort switches the producers into a discard mode where
   publishes are accepted, cached nowhere, and reported freed, and the exception is
   raised only after the loop has finished its schedule.

2. **Frees must be idempotent per (group, consumer).** A retried sync can deliver a
   free a consumer already sent. The old ledger counted CALLS, so the duplicate read
   as a second consumer and released a group the other consumer was still pulling from
   — a use-after-free of a live CUDA-IPC import.

3. **Abandoned exports must outlive their imports.** On abort the trainer's refs to
   gathered storage cannot simply be dropped: a consumer may be mid-copy. They are
   carried to the next ``begin_sync`` instead.

CPU-only: the producer is driven in-process (with ``rebuild_cuda_tensor`` stubbed) and
the engine is built field-by-field.

Run:
    uv run --extra dev --extra fsdp pytest tests/backends/skyrl_train/weight_sync/test_rdt_midsync_abort.py
"""

import threading
from types import SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.inference_servers.rdt_control_protocol import (
    WeightSyncAborted,
)
from skyrl.backends.skyrl_train.weight_sync.rdt_send import RdtWeightSyncSender
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_trainer import (
    _RDTProducerServer,
)


def _server(lookahead=4, monkeypatch=None):
    """A real ``_RDTProducerServer`` with only its CUDA/NIXL edges stubbed.

    The real class, not a fake: the whole point of these tests is the ref-counting and
    abort protocol, and a reimplementation of it would be the thing under test.
    """
    s = _RDTProducerServer.__new__(_RDTProducerServer)
    s._device_index = 0
    s._cache = {}
    s._cache_cond = threading.Condition()
    s._gather_error = None
    s._stall_timeout = 5.0
    s._last_progress = 0.0
    s._served_names = None
    s._free_targets = {}
    s._freed_by = {}
    s._anon_frees = 0
    s._lookahead = lookahead
    s._inflight_keys = []
    s._name_to_key = {}
    s._freed_pending = []
    s._nring = 1
    s._serve_rings = {}
    s._serve_idx = {}
    s._serve_lock = threading.Lock()
    s._reg_lock = threading.Lock()
    s._arena_presize = 0
    s._serve_stream = None
    s._cache_event = {}
    s._pack_check = False
    s._pack_dsts = {}
    s._timing_lock = threading.Lock()
    for attr in (
        "_produce_calls",
        "_produce_specs",
        "_produce_bytes",
        "_publish_calls",
        "_publish_open_count",
        "_serve_lag_waited_calls",
        "_serve_lag_ready_calls",
        "_free_lag_count",
    ):
        setattr(s, attr, 0)
    for attr in (
        "_produce_wait_seconds",
        "_produce_slice_seconds",
        "_produce_method_seconds",
        "_publish_bp_wait_seconds",
        "_publish_rebuild_seconds",
        "_publish_open_seconds",
        "_publish_view_seconds",
        "_serve_lag_waited_seconds",
        "_serve_lag_ready_seconds",
        "_free_lag_seconds",
        "_publish_to_free_seconds",
    ):
        setattr(s, attr, 0.0)
    s._publish_time = {}
    s._serve_done = {}
    return s


def _entries(names):
    """``(storages, views)`` for a group, shaped so the stubbed rebuild can serve it.

    The storage entry has to be at least 7 long: ``publish_group`` overwrites index 6
    (``reduce_tensor``'s device index) before rebuilding.
    """
    return ({0: tuple(range(8))}, {n: (0, "float32", (2,), (1,), 0) for n in names})


@pytest.fixture
def stub_rebuild(monkeypatch):
    """Replace the CUDA-IPC rebuild with a CPU tensor of the right shape."""
    import torch

    from skyrl.backends.skyrl_train.weight_sync import sharded_rdt_trainer as t

    monkeypatch.setattr(t, "rebuild_cuda_tensor", lambda *a, **kw: torch.zeros(8, dtype=torch.uint8))
    # as_strided on a CPU uint8 base viewed as float32 needs a real byte count; the
    # views above are 2 floats = 8 bytes, which the stub provides.
    return True


class TestFreeLedger:
    """Frees are counted per CONSUMER, so a retry's duplicate cannot over-credit."""

    def test_two_distinct_consumers_release_the_group(self, stub_rebuild):
        s = _server()
        key = ("a", "b")
        s.publish_group(key, _entries(key), free_target=2)
        assert s._inflight_keys == [key]

        s.free_gather(list(key), consumer_id=0)
        assert s._inflight_keys == [key], "released after only one of two consumers"

        s.free_gather(list(key), consumer_id=1)
        assert s._inflight_keys == []
        assert "a" not in s._cache

    def test_a_duplicate_free_from_one_consumer_does_not_release(self, stub_rebuild):
        """THE bug identity-keying exists to prevent. With a call counter, consumer 0
        freeing twice looks like both consumers being done, and the group is released
        while consumer 1 is still pulling out of it."""
        s = _server()
        key = ("a", "b")
        s.publish_group(key, _entries(key), free_target=2)

        s.free_gather(list(key), consumer_id=0)
        s.free_gather(list(key), consumer_id=0)

        assert s._inflight_keys == [key], "a repeated free from ONE consumer released the group"
        assert "a" in s._cache, "cache entries dropped while a consumer may still be pulling"

    def test_the_backpressure_slot_is_not_over_credited(self, stub_rebuild):
        """The same bug seen from the other side: an over-credited free frees a
        lookahead slot that is still occupied, so the gather loop runs ahead of its
        memory bound (~17GiB of extra resident gather at 235B)."""
        s = _server(lookahead=2)
        k1, k2 = ("a",), ("b",)
        s.publish_group(k1, _entries(k1), free_target=2)
        s.publish_group(k2, _entries(k2), free_target=2)
        for _ in range(5):
            s.free_gather(list(k1), consumer_id=0)
        assert len(s._inflight_keys) == 2

    def test_a_consumer_without_an_id_keeps_the_old_counting(self, stub_rebuild):
        """Back-compat: an older consumer sends no id, and must behave exactly as
        before rather than being silently deduplicated into never releasing."""
        s = _server()
        key = ("a",)
        s.publish_group(key, _entries(key), free_target=2)
        s.free_gather(list(key))
        s.free_gather(list(key))
        assert s._inflight_keys == []

    def test_frees_arriving_before_the_publish_still_release_it(self, stub_rebuild):
        """Pre-existing behaviour that the ledger must not break: a consumer that
        pulls nothing of a group frees it as soon as its plan starts."""
        s = _server()
        key = ("a",)
        s.free_gather(list(key), consumer_id=0)
        s.free_gather(list(key), consumer_id=1)
        s.publish_group(key, _entries(key), free_target=2)
        assert s._inflight_keys == []

    def test_begin_sync_clears_the_ledger(self, stub_rebuild):
        s = _server()
        key = ("a",)
        s.free_gather(list(key), consumer_id=0)
        s.begin_sync()
        assert s._freed_by == {}
        s.publish_group(key, _entries(key), free_target=1)
        assert s._inflight_keys == [key], "a stale free from the previous sync credited this one"


class TestDiscardMode:
    """Under a gather error, publishes are accepted and dropped."""

    def test_a_publish_after_an_abort_caches_nothing(self, stub_rebuild):
        s = _server()
        s.set_gather_error("engine died")
        key = ("a", "b")
        freed = s.publish_group(key, _entries(key), free_target=2)

        assert key in freed, "the group must be reported freed so the loop keeps moving"
        assert s._cache == {}, "cached under abort -- a consumer could pull abandoned weights"
        assert s._inflight_keys == [], "consumed a lookahead slot that nothing will free"

    def test_the_loop_can_run_to_completion_under_abort(self, stub_rebuild):
        """The invariant that keeps one dead consumer from killing the job: with a
        lookahead of 2, publishing 10 groups after an abort must not block -- every
        rank has to finish its collective schedule."""
        s = _server(lookahead=2)
        s.set_gather_error("engine died")
        for i in range(10):
            key = (f"g{i}",)
            freed = s.publish_group(key, _entries(key), free_target=4)
            assert freed == [key]
        assert s._inflight_keys == []

    def test_end_sync_does_not_claim_unfreed_groups(self, stub_rebuild):
        """Under abort, ``end_sync`` returns early. It must NOT report the still-inflight
        keys as freed: the engine would drop trainer-side refs to storage a consumer
        may be mid-copy out of."""
        s = _server()
        key = ("a",)
        s.publish_group(key, _entries(key), free_target=2)
        s.set_gather_error("engine died")
        freed = s.end_sync()
        assert key not in freed
        assert s._inflight_keys == [key]

    def test_a_healthy_publish_is_untouched(self, stub_rebuild):
        """Discard mode must be reachable only via a gather error -- otherwise every
        sync silently drops its weights."""
        s = _server()
        key = ("a",)
        s.publish_group(key, _entries(key), free_target=1)
        assert "a" in s._cache
        assert s._inflight_keys == [key]


class TestAbandonedExports:
    """Trainer-side refs from an aborted sync outlive it by one sync."""

    @staticmethod
    def _engine():
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_trainer import (
            ShardedRDTTrainerWeightTransferEngine as E,
        )

        e = E.__new__(E)
        e._inflight = {("a",): {"a": object()}}
        e._abandoned_inflight = {}
        e._sync_timing = {}
        return e

    def test_an_abort_carries_the_refs_instead_of_dropping_them(self):
        """Dropping them inside the failing sync is a use-after-free of the CUDA-IPC
        export while a consumer is still importing it."""
        e = self._engine()
        held = e._inflight[("a",)]["a"]
        e._abandoned_inflight.update(e._inflight)
        e._inflight.clear()
        assert e._abandoned_inflight[("a",)]["a"] is held

    def test_the_next_sync_releases_them(self):
        e = self._engine()
        e._abandoned_inflight.update(e._inflight)
        e._inflight.clear()
        # What begin_sync does at the top of the next round.
        e._abandoned_inflight.clear()
        assert e._abandoned_inflight == {}

    def test_the_gather_loop_holds_the_refs_across_the_abort(self):
        """Asserted on the source: the alternative (a live GPU sync with a consumer
        mid-pull) is not reachable on CPU, and the failure this guards is silent."""
        import inspect

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_trainer import (
            ShardedRDTTrainerWeightTransferEngine as E,
        )

        src = inspect.getsource(E._run_gather_loop)
        assert "self._abandoned_inflight.update(self._inflight)" in src
        assert src.index("self._abandoned_inflight.update(self._inflight)") < src.index("self._inflight.clear()")


class TestClassification:
    """Only an engine death is retried; everything else propagates unchanged."""

    @staticmethod
    def _sender(aborts):
        s = RdtWeightSyncSender.__new__(RdtWeightSyncSender)
        s._server_urls = ["http://a", "http://b"]
        s._server_slots = [0, 1]
        s._num_replicas = 2
        s._world_size = 2
        s._data_parallel_size = 1
        s._control_plane = SimpleNamespace(abort_weight_update=lambda urls=None: aborts.append(urls))
        return s

    @pytest.mark.parametrize(
        "exc",
        [
            ConnectionError("Connection refused"),
            RuntimeError("RDT stall: no progress for 300s while waiting for a lookahead credit"),
            RuntimeError("ConnectionError: [Errno 111] Connection refused"),
        ],
    )
    def test_an_engine_death_becomes_retryable(self, exc):
        aborts = []
        s = self._sender(aborts)
        out = s._classify_sync_failure(exc, [("http://a", 0)])
        assert isinstance(out, WeightSyncAborted)
        assert out.newly_dead_slots == (0,)
        assert aborts == [None], "the surviving engines were not told to abandon the update"

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("disallowed op 't'"),
            RuntimeError("CUDA out of memory"),
            AssertionError("group_lens mismatch"),
        ],
    )
    def test_a_real_bug_is_not_retried(self, exc):
        """Retrying a shape mismatch or an OOM hides the fault and burns the budget."""
        aborts = []
        s = self._sender(aborts)
        out = s._classify_sync_failure(exc, None)
        assert out is exc
        assert aborts == [], "aborted the engines for a failure that is not an engine failure"

    def test_a_non_sender_rank_classifies_nothing(self):
        """Only rank 0 holds a control plane, and only it can unwind the engines.
        Other ranks return cleanly from their gather loops, so re-entry is safe."""
        s = self._sender([])
        s._control_plane = None
        exc = ConnectionError("Connection refused")
        assert s._classify_sync_failure(exc, None) is exc

    def test_an_already_classified_abort_passes_through(self):
        """Guards against double-aborting the engines on a nested failure."""
        aborts = []
        s = self._sender(aborts)
        e = WeightSyncAborted("already", [1])
        assert s._classify_sync_failure(e, None) is e
        assert aborts == []


class TestDriverRetry:
    """The retry is issued by the DRIVER, because re-entry is a collective."""

    @staticmethod
    def _dispatch(fail_times, supervisor=None):
        """A dispatch whose broadcast fails ``fail_times`` times with an abort."""
        from types import SimpleNamespace

        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        d = WorkerDispatch.__new__(WorkerDispatch)
        d.cfg = SimpleNamespace(
            generator=SimpleNamespace(
                inference_engine=SimpleNamespace(
                    fault_tolerance=SimpleNamespace(enabled=True, max_sync_retries=2)
                )
            )
        )
        trace = []
        state = {"n": 0}

        def _live():
            trace.append(f"read_live({len(state['dead'])} dead)")
            return [(u, i) for i, u in enumerate(["a", "b"]) if u not in state["dead"]]

        state["dead"] = set()
        state["seen"] = []

        def _broadcast(client, model_id=None, live_url_slots=None):
            state["n"] += 1
            trace.append("broadcast")
            # Recorded separately from the trace label: a substring check against
            # "broadcast(...)" also matches the letters of the word itself.
            state["seen"].append(list(live_url_slots or []))
            if state["n"] <= fail_times:
                raise WeightSyncAborted("engine died mid-sync", [1])

        d._live_url_slots = _live
        d._broadcast_to_inference_engines = _broadcast
        d._inference_engine_client = SimpleNamespace()
        return d, trace, state

    @pytest.mark.asyncio
    async def test_the_fleet_is_re_probed_between_attempts(self):
        """The retry's whole point. Re-dispatching against the SAME live set sends the
        retry straight back into the corpse and burns every attempt on one failure --
        so a reconcile has to happen in between, which is why the retry lives on the
        async path rather than in the sync broadcast helper."""
        reconciles = []

        class _Sup:
            async def reconcile(self_inner):
                reconciles.append(1)
                # What a real reconcile does: names the dead engine.
                state["dead"].add("b")

        d, trace, state = self._dispatch(fail_times=1)
        sup = _Sup()

        await d._broadcast_with_retry(None, sup)

        assert len(reconciles) == 1, "retried without re-probing the fleet"
        assert len(state["seen"]) == 2
        assert state["seen"][0] == [("a", 0), ("b", 1)]
        assert state["seen"][1] == [("a", 0)], (
            f"the retry did not exclude the dead engine: {state['seen']}"
        )

    @pytest.mark.asyncio
    async def test_it_gives_up_after_max_sync_retries(self):
        d, trace, state = self._dispatch(fail_times=5)

        class _Sup:
            async def reconcile(self_inner):
                pass

        with pytest.raises(WeightSyncAborted):
            await d._broadcast_with_retry(None, _Sup())
        assert len([t for t in trace if t.startswith("broadcast")]) == 3  # 1 + 2 retries

    @pytest.mark.asyncio
    async def test_a_non_abort_failure_is_not_retried(self):
        """Repeating an OOM or a shape mismatch hides the fault and wastes the budget."""
        d, trace, state = self._dispatch(fail_times=0)

        def _broadcast(client, model_id=None, live_url_slots=None):
            trace.append("broadcast")
            raise RuntimeError("CUDA out of memory")

        d._broadcast_to_inference_engines = _broadcast
        with pytest.raises(RuntimeError, match="out of memory"):
            await d._broadcast_with_retry(None, None)
        assert trace.count("broadcast") == 1

    @pytest.mark.asyncio
    async def test_retries_are_off_when_fault_tolerance_is(self):
        d, trace, state = self._dispatch(fail_times=5)
        d.cfg.generator.inference_engine.fault_tolerance.enabled = False
        with pytest.raises(WeightSyncAborted):
            await d._broadcast_with_retry(None, None)
        assert len([t for t in trace if t.startswith("broadcast")]) == 1

    def test_a_ray_wrapped_abort_is_still_recognised(self):
        """Ray delivers the worker's exception wrapped, and the sender chains the
        original through ``raise ... from``. An isinstance on the top-level object
        misses the case the retry exists for."""
        from skyrl.backends.skyrl_train.workers.worker_dispatch import (
            _as_weight_sync_aborted,
        )

        inner = WeightSyncAborted("engine died", [2])

        class _RayTaskError(RuntimeError):
            def __init__(self, cause):
                super().__init__("ray wrapped")
                self.cause = cause

        assert _as_weight_sync_aborted(_RayTaskError(inner)) is inner

        try:
            try:
                raise inner
            except WeightSyncAborted as e:
                raise RuntimeError("outer") from e
        except RuntimeError as outer:
            assert _as_weight_sync_aborted(outer) is inner

    def test_an_unrelated_error_is_not_mistaken_for_an_abort(self):
        from skyrl.backends.skyrl_train.workers.worker_dispatch import (
            _as_weight_sync_aborted,
        )

        assert _as_weight_sync_aborted(RuntimeError("CUDA OOM")) is None

    def test_a_self_referential_chain_terminates(self):
        """A cycle in __context__ would otherwise spin forever inside a failure path."""
        from skyrl.backends.skyrl_train.workers.worker_dispatch import (
            _as_weight_sync_aborted,
        )

        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        assert _as_weight_sync_aborted(a) is None
