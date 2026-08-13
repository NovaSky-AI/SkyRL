"""Degraded weight sync: the driver's live URL set reaching the producers.

Two hops, each with its own way of going quietly wrong:

  * ``RdtWeightSyncSender._live_consumer_ids`` maps the live servers' SLOTS to the
    consumer ids those servers own. Both inputs must come from PROVISION —
    re-deriving the geometry from a shrunken list would renumber every surviving
    consumer, and each would then pull another consumer's slice with nothing
    downstream to notice. Slots rather than URLs because a restarted engine returns
    on a re-reserved port: the URL is an address, the slot is the identity.
  * ``ShardedRDTTrainerWeightTransferEngine.send_weights`` turns that set into
    this sync's free-barrier TARGET — the live consumer COUNT handed to the
    producer server's ``begin_sync``. The provisioned geometry (router,
    ownership, ``served_names``) is frozen; degrading only lowers one integer.

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


def _sender(urls, world_size, dp=1, slots=None) -> RdtWeightSyncSender:
    """A sender carrying only the provisioned snapshot the mapping reads."""
    s = RdtWeightSyncSender.__new__(RdtWeightSyncSender)
    s._server_urls = list(urls)
    s._server_slots = list(slots) if slots is not None else [i // dp for i in range(len(urls))]
    s._num_replicas = max(1, len(set(s._server_slots)))
    s._world_size = world_size
    s._data_parallel_size = dp
    return s


def _live(sender, *urls):
    """The driver's ``(url, slot)`` view restricted to ``urls``."""
    by_url = dict(zip(sender._server_urls, sender._server_slots))
    return [(u, by_url[u]) for u in urls]


class TestLiveConsumerIds:
    def test_none_stays_none(self):
        """The whole-fleet path must reach the engine as ``None`` so a non-degraded
        sync takes exactly the pre-fault-tolerance code path."""
        s = _sender(["http://a", "http://b"], world_size=4)
        assert s._live_consumer_ids(None) is None

    def test_a_full_live_list_collapses_to_none(self):
        s = _sender(["http://a", "http://b"], world_size=4)
        assert s._live_consumer_ids(_live(s, "http://a", "http://b")) is None
        assert s._live_consumer_ids(_live(s, "http://b", "http://a")) is None

    def test_one_dead_engine_drops_exactly_its_block(self):
        """2 engines x TP2 = 4 consumers. Engine b owns 2 and 3."""
        s = _sender(["http://a", "http://b"], world_size=4)
        assert s._live_consumer_ids(_live(s, "http://a")) == {0, 1}
        assert s._live_consumer_ids(_live(s, "http://b")) == {2, 3}

    def test_the_surviving_blocks_keep_their_ids(self):
        """The invariant everything else rests on: a survivor's consumer ids are a
        function of its SLOT, not of how many engines are still up. 4 engines x TP2;
        kill the middle two."""
        s = _sender([f"http://e{i}" for i in range(4)], world_size=8)
        assert s._live_consumer_ids(_live(s, "http://e0", "http://e3")) == {0, 1, 6, 7}

    def test_tp1_engines_map_one_to_one(self):
        s = _sender([f"http://e{i}" for i in range(4)], world_size=4)
        assert s._live_consumer_ids(_live(s, "http://e0", "http://e2")) == {0, 2}

    def test_data_parallel_servers_share_a_replica_block(self):
        """dp=2: the two servers of a deployment share one replica_rank, so the
        block is the deployment's, not the server's."""
        s = _sender([f"http://e{i}" for i in range(4)], world_size=4, dp=2)
        # e0/e1 are deployment 0 (consumers 0,1); e2/e3 are deployment 1 (2,3).
        assert s._live_consumer_ids(_live(s, "http://e0", "http://e1")) == {0, 1}
        assert s._live_consumer_ids(_live(s, "http://e2", "http://e3")) == {2, 3}

    def test_duplicates_are_harmless(self):
        s = _sender(["http://a", "http://b"], world_size=4)
        assert s._live_consumer_ids([("http://a", 0), ("http://a", 0)]) == {0, 1}

    def test_an_unprovisioned_slot_is_a_hard_error(self):
        """A slot nobody provisioned means the fleet changed SHAPE, not that one
        engine moved — the one thing this path must never absorb silently."""
        s = _sender(["http://a", "http://b"], world_size=4)
        with pytest.raises(RuntimeError, match="outside the provisioned set"):
            s._live_consumer_ids([("http://a", 0), ("http://new", 7)])

    def test_a_restarted_engine_keeps_its_block_under_a_new_url(self):
        """The reason identity is the slot: a restart re-reserves a port, so the
        driver reports slot 1 at an address the sender's snapshot has never seen.
        That must be a normal live engine, not an error, and it must own exactly the
        consumer ids it owned before it died."""
        s = _sender(["http://a", "http://b"], world_size=4)
        assert s._live_consumer_ids([("http://a", 0), ("http://b-restarted:8101", 1)]) is None
        assert s._live_consumer_ids([("http://b-restarted:8101", 1)]) == {2, 3}

    def test_an_empty_live_set_is_a_hard_error(self):
        s = _sender(["http://a", "http://b"], world_size=4)
        with pytest.raises(RuntimeError, match="no inference server is alive"):
            s._live_consumer_ids([])


class TestDegradationIsVisible:
    """A degraded sync must announce itself somewhere an operator will actually look.

    Both this sender-level summary and the vendored engine's per-group
    ``[rdt-degraded]`` land in SkyRL's node-local infra log
    (``/tmp/skyrl-logs/infra-*.log``); neither reaches the DRIVER's stdout, so a
    degraded run reads as healthy from the log most people watch.

    This test only pins that the sender emits at all -- an in-process sink cannot
    tell you where the line ends up. Verified on GPU: 4 post-kill syncs x 4 ranks
    produced 16 sender lines and 16 engine lines in the infra log, with rank 0
    reporting 26/26 owned groups dropped and the rest 0/26 -- matching the per-rank
    produce counters exactly.
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
            await s.send(object(), [("http://a", 0)])
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


def _engine(num_consumers, world=2, rank=0):
    """An engine with just the state ``send_weights`` reads to derive the live
    count, its downstream stubbed to record what it received."""
    e = ShardedRDTTrainerWeightTransferEngine.__new__(ShardedRDTTrainerWeightTransferEngine)
    e._init_info = ShardedRDTTrainerInitInfo(rank=rank, num_consumers=num_consumers)
    e._router = RdtRouter(world, num_consumers, None, num_groups=4)
    e.source = object()  # send_weights asserts a source is present
    received: list = []
    e._send_weights_inner = received.append
    return e, received


class TestLiveCountPlumbing:
    """``send_weights(live_consumer_ids)`` -> the barrier target. The
    provisioned geometry is frozen; a degraded sync only lowers the one integer
    every owned group's barrier counts to."""

    def test_none_counts_the_whole_provisioned_fleet(self):
        e, got = _engine(num_consumers=8)
        e.send_weights(None)
        assert got == [8]

    def test_a_live_set_counts_its_distinct_members(self):
        e, got = _engine(num_consumers=8)
        e.send_weights([0, 1, 4, 5, 5])
        assert got == [4]

    def test_a_full_live_set_matches_the_provisioned_count(self):
        """Explicitly-live-everyone and None must produce the same target, so a
        healthy FT run is indistinguishable from a non-FT run."""
        e, got = _engine(num_consumers=8)
        e.send_weights(list(range(8)))
        e.send_weights(None)
        assert got == [8, 8]

    def test_which_consumers_died_does_not_matter_only_how_many(self):
        """The whole point of the barrier: no routed per-producer targets, so
        the identity of the dead consumer is irrelevant to the producers."""
        counts = []
        for live in ([0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6]):
            e, got = _engine(num_consumers=8)
            e.send_weights(live)
            counts += got
        assert counts == [4, 4, 4]
