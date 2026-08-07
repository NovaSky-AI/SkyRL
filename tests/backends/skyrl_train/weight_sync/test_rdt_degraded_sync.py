"""Degraded weight sync: the driver's live URL set reaching the producers.

Two hops, each with its own way of going quietly wrong:

  * ``RdtWeightSyncSender._live_consumer_ids`` maps live server URLs to the
    consumer ids those servers own. Every index must come from the PROVISIONED
    snapshot — re-deriving the geometry from a shrunken list would renumber every
    surviving consumer, and each would then pull another consumer's slice with
    nothing downstream to notice.
  * ``ShardedRDTTrainerWeightTransferEngine._live_consumers`` recomputes this
    sync's free targets from that set and restores the provisioned ones after,
    so a degraded sync cannot leak into the next one.

CPU-only: both are pure functions over already-resolved geometry, so the engine
and sender are built field-by-field rather than through their real inits (which
want Ray, NCCL and a GPU).

Run:
    uv run --extra dev --extra fsdp pytest tests/backends/skyrl_train/weight_sync/test_rdt_degraded_sync.py
"""

import pytest

from skyrl.backends.skyrl_train.weight_sync.rdt_send import RdtWeightSyncSender
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import RdtRouter
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_trainer import (
    ShardedRDTTrainerInitInfo,
    ShardedRDTTrainerWeightTransferEngine,
)


def _sender(urls, world_size, dp=1) -> RdtWeightSyncSender:
    """A sender carrying only the provisioned snapshot the mapping reads."""
    s = RdtWeightSyncSender.__new__(RdtWeightSyncSender)
    s._server_urls = list(urls)
    s._world_size = world_size
    s._data_parallel_size = dp
    return s


class TestLiveConsumerIds:
    def test_none_stays_none(self):
        """The whole-fleet path must reach the engine as ``None`` so a non-degraded
        sync takes exactly the pre-fault-tolerance code path."""
        s = _sender(["http://a", "http://b"], world_size=4)
        assert s._live_consumer_ids(None) is None

    def test_a_full_live_list_collapses_to_none(self):
        s = _sender(["http://a", "http://b"], world_size=4)
        assert s._live_consumer_ids(["http://a", "http://b"]) is None
        assert s._live_consumer_ids(["http://b", "http://a"]) is None

    def test_one_dead_engine_drops_exactly_its_block(self):
        """2 engines x TP2 = 4 consumers. Engine b owns 2 and 3."""
        s = _sender(["http://a", "http://b"], world_size=4)
        assert s._live_consumer_ids(["http://a"]) == {0, 1}
        assert s._live_consumer_ids(["http://b"]) == {2, 3}

    def test_the_surviving_blocks_keep_their_ids(self):
        """The invariant everything else rests on: a survivor's consumer ids are a
        function of its position in the PROVISIONED list, not of how many engines
        are still up. 4 engines x TP2; kill the middle two."""
        s = _sender([f"http://e{i}" for i in range(4)], world_size=8)
        assert s._live_consumer_ids(["http://e0", "http://e3"]) == {0, 1, 6, 7}

    def test_tp1_engines_map_one_to_one(self):
        s = _sender([f"http://e{i}" for i in range(4)], world_size=4)
        assert s._live_consumer_ids(["http://e0", "http://e2"]) == {0, 2}

    def test_data_parallel_servers_share_a_replica_block(self):
        """dp=2: the two servers of a deployment share one replica_rank, so the
        block is the deployment's, not the server's."""
        s = _sender([f"http://e{i}" for i in range(4)], world_size=4, dp=2)
        # e0/e1 are deployment 0 (consumers 0,1); e2/e3 are deployment 1 (2,3).
        assert s._live_consumer_ids(["http://e0", "http://e1"]) == {0, 1}
        assert s._live_consumer_ids(["http://e2", "http://e3"]) == {2, 3}

    def test_duplicates_are_harmless(self):
        s = _sender(["http://a", "http://b"], world_size=4)
        assert s._live_consumer_ids(["http://a", "http://a"]) == {0, 1}

    def test_an_unprovisioned_url_is_a_hard_error(self):
        """A URL nobody provisioned means a server was re-created, which moves the
        geometry — the one thing this path must never absorb silently."""
        s = _sender(["http://a", "http://b"], world_size=4)
        with pytest.raises(RuntimeError, match="outside the provisioned set"):
            s._live_consumer_ids(["http://a", "http://new"])

    def test_an_empty_live_set_is_a_hard_error(self):
        s = _sender(["http://a", "http://b"], world_size=4)
        with pytest.raises(RuntimeError, match="no inference server is alive"):
            s._live_consumer_ids([])


class TestDegradationIsVisible:
    """A degraded sync must announce itself somewhere an operator will actually look.

    The vendored engine's ``[rdt-degraded]`` goes through vLLM's ``init_logger``,
    which in a SkyRL policy worker reaches no log file and no console -- confirmed on
    a GPU run where the degraded path provably ran (the two producers bound to the
    dead consumer went to zero produce calls) while the line appeared in none of the
    worker logs, infra logs, or driver output. loguru from the same process is
    captured, so the announcement is made there too.
    """

    @pytest.mark.asyncio
    async def test_a_degraded_sync_logs_through_loguru(self, monkeypatch):
        import loguru

        from skyrl.backends.skyrl_train.weight_sync import rdt_send as rs

        seen = []
        monkeypatch.setattr(rs, "_loguru", loguru.logger.bind())
        sink_id = loguru.logger.add(lambda m: seen.append(str(m)), level="WARNING")
        try:
            s = _sender(["http://a", "http://b"], world_size=4)
            s._engine = _StubEngine()
            s._control_plane = None
            await s.send(object(), ["http://a"])
        finally:
            loguru.logger.remove(sink_id)

        assert any("rdt-degraded" in m for m in seen), seen
        assert any("2/4 live consumers" in m for m in seen), seen

    @pytest.mark.asyncio
    async def test_a_whole_fleet_says_nothing(self, monkeypatch):
        """Silence is the signal that nothing is degraded; a per-sync line on a
        healthy run would train operators to ignore it."""
        import loguru

        from skyrl.backends.skyrl_train.weight_sync import rdt_send as rs

        seen = []
        monkeypatch.setattr(rs, "_loguru", loguru.logger.bind())
        sink_id = loguru.logger.add(lambda m: seen.append(str(m)), level="WARNING")
        try:
            s = _sender(["http://a", "http://b"], world_size=4)
            s._engine = _StubEngine()
            s._control_plane = None
            await s.send(object(), None)
        finally:
            loguru.logger.remove(sink_id)

        assert not any("rdt-degraded" in m for m in seen), seen


class _StubEngine:
    """Just enough engine for ``send`` to run without Ray, torch or a GPU."""

    def __init__(self):
        self.live_cids = "unset"

    def send_weights(self, live_consumer_ids=None):
        self.live_cids = live_consumer_ids


def _engine(num_consumers, world, rank, group_owners=None, num_groups=4):
    """An engine with just the routing state ``_live_consumers`` reads."""
    e = ShardedRDTTrainerWeightTransferEngine.__new__(ShardedRDTTrainerWeightTransferEngine)
    e._init_info = ShardedRDTTrainerInitInfo(rank=rank, num_consumers=num_consumers)
    router = RdtRouter(world, num_consumers, group_owners, num_groups=num_groups)
    e._router = router
    e._owned_idx = router.owned_groups(rank) if group_owners else list(range(num_groups))
    e._free_targets = {gi: router.free_target(rank, gi) for gi in e._owned_idx}
    e._world_and_rank = lambda: (world, rank)
    return e


class TestLiveConsumersScope:
    def test_none_leaves_the_free_targets_untouched(self):
        e = _engine(num_consumers=4, world=4, rank=0)
        before = dict(e._free_targets)
        with e._live_consumers(None):
            assert e._free_targets == before
        assert e._free_targets == before

    def test_targets_shrink_inside_the_scope(self):
        """2 producers, 8 consumers: rank 0 serves consumers 0-3. Kill two of them
        and its target for every group drops from 4 to 2."""
        e = _engine(num_consumers=8, world=2, rank=0)
        assert set(e._free_targets.values()) == {4}
        with e._live_consumers([0, 1, 4, 5, 6, 7]):
            assert set(e._free_targets.values()) == {2}

    def test_the_provisioned_targets_are_restored(self):
        """A degraded sync must not leak into the next one — if the fleet comes back
        (or simply if the next sync is dispatched with ``None``), the targets have to
        be the provisioned ones again."""
        e = _engine(num_consumers=8, world=2, rank=0)
        before = dict(e._free_targets)
        with e._live_consumers([4, 5, 6, 7]):
            assert set(e._free_targets.values()) == {0}
        assert e._free_targets == before

    def test_targets_are_restored_even_when_the_sync_raises(self):
        e = _engine(num_consumers=8, world=2, rank=0)
        before = dict(e._free_targets)
        with pytest.raises(RuntimeError):
            with e._live_consumers([4, 5, 6, 7]):
                raise RuntimeError("gather failed")
        assert e._free_targets == before

    def test_a_dead_consumer_block_zeroes_the_producer(self):
        """Rank 0's entire consumer block is gone: every one of its groups becomes
        gather-and-drop. It still runs each gather collective (the peers need it) but
        publishes nothing, which is what keeps the backpressure slots free."""
        e = _engine(num_consumers=8, world=2, rank=0)
        with e._live_consumers([4, 5, 6, 7]):
            assert all(t == 0 for t in e._free_targets.values())

    def test_pipeline_local_ownership_keeps_its_group_set(self):
        """PP-local: rank 8 owns groups 3-5 only. Degrading changes the TARGETS, never
        the owned groups — the gather schedule is a collective and must not move."""
        owners = [list(range(8))] * 3 + [list(range(8, 16))] * 3
        e = _engine(num_consumers=8, world=16, rank=8, group_owners=owners, num_groups=6)
        assert e._owned_idx == [3, 4, 5]
        with e._live_consumers([0, 1, 2, 3]):
            assert sorted(e._free_targets) == [3, 4, 5]
        assert sorted(e._free_targets) == [3, 4, 5]

    def test_targets_match_the_router_over_the_live_set(self):
        owners = [list(range(8))] * 3 + [list(range(8, 16))] * 3
        e = _engine(num_consumers=8, world=16, rank=8, group_owners=owners, num_groups=6)
        live = [0, 2, 4, 6]
        with e._live_consumers(live):
            for gi in e._owned_idx:
                assert e._free_targets[gi] == e._router.free_target(8, gi, live)

    def test_a_full_live_set_matches_the_provisioned_targets(self):
        e = _engine(num_consumers=8, world=2, rank=1)
        before = dict(e._free_targets)
        with e._live_consumers(list(range(8))):
            assert e._free_targets == before
