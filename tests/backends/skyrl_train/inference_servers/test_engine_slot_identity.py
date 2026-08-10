"""Slot identity: the prerequisite for restarting an inference engine.

Part 1 of engine fault tolerance made *position* in ``server_urls`` the identity of
an engine -- ``replica_rank = index // dp`` -- which works only as long as the fleet
can lose engines but never gain them back. A restarted engine re-reserves a port
inside its own ``SERVER_PORT_STRIDE`` window and so returns on a DIFFERENT URL, at
which point "the URL is the identity" is false and every consumer's slice mapping is
up for grabs.

The fix is a slot: an ordinal assigned at provision, reused by the restart, and never
compacted. These tests pin the three places that has to hold:

  * ``build_rdt_init_payloads`` -- ``replica_rank`` comes from the slot, and
    ``num_replicas`` from the PROVISIONED count, so a single-target re-init cannot
    collapse to "deployment 0 of 1".
  * ``ServerActorPool`` / ``ServerGroup`` -- slots are stamped driver-side and
    survive a ``replace_actor``.
  * ``RemoteInferenceClient`` -- the ``(url, slot)`` views, and adopting a moved URL.

The failure mode if any of this is wrong is not an exception: it is a consumer
silently pulling another consumer's slice. So the assertions here are about exact
rank/id arithmetic rather than about types.

CPU-only. Nothing starts a server; the pool is driven with stand-in actors.

Run:
    uv run --extra dev pytest tests/backends/skyrl_train/inference_servers/test_engine_slot_identity.py
"""

import pytest

from skyrl.backends.skyrl_train.inference_servers.common import ServerInfo
from skyrl.backends.skyrl_train.inference_servers.rdt_control_protocol import (
    RDT_INIT_METHOD,
    build_rdt_init_payloads,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.server_pool import ServerActorPool
from skyrl.train.config.config import InferenceFaultToleranceConfig


def _ranks(payloads):
    """``{url: replica_rank}`` from a payload list."""
    return {url: p["kwargs"]["init_info"]["replica_rank"] for url, p in payloads}


def _replicas(payloads):
    """The single ``num_replicas`` every payload must agree on."""
    seen = {p["kwargs"]["init_info"]["num_replicas"] for _, p in payloads}
    assert len(seen) == 1, f"payloads disagree on num_replicas: {seen}"
    return seen.pop()


class TestInitPayloadSlots:
    URLS = ["http://e0", "http://e1", "http://e2", "http://e3"]

    def test_the_default_path_is_the_positional_rule(self):
        """No slots given == the pre-fault-tolerance behaviour, exactly. Every static
        fleet and both original callers land here."""
        p = build_rdt_init_payloads({"num_consumers": 8}, self.URLS, 1)
        assert _ranks(p) == {"http://e0": 0, "http://e1": 1, "http://e2": 2, "http://e3": 3}
        assert _replicas(p) == 4
        assert all(pl["method"] == RDT_INIT_METHOD for _, pl in p)

    def test_the_default_path_divides_by_dp(self):
        p = build_rdt_init_payloads({}, self.URLS, 2)
        assert _ranks(p) == {"http://e0": 0, "http://e1": 0, "http://e2": 1, "http://e3": 1}
        assert _replicas(p) == 2

    def test_explicit_slots_matching_the_positional_rule_change_nothing(self):
        """The migration safety property: passing the slots a static fleet would have
        derived must produce byte-identical payloads."""
        implicit = build_rdt_init_payloads({"num_consumers": 8}, self.URLS, 2)
        explicit = build_rdt_init_payloads({"num_consumers": 8}, self.URLS, 2, slots=[0, 0, 1, 1], num_replicas=2)
        assert explicit == implicit

    def test_one_restarted_engine_keeps_its_rank(self):
        """The case that motivates the whole mechanism. Re-initializing slot 2 alone:
        positionally it would be rank 0 of 1 replica, which is deployment 0's consumer
        block -- it would pull another engine's slices."""
        p = build_rdt_init_payloads({}, ["http://e2-new:8201"], 1, slots=[2], num_replicas=4)
        assert _ranks(p) == {"http://e2-new:8201": 2}
        assert _replicas(p) == 4

    def test_num_replicas_is_provisioned_not_derived(self):
        """A degraded subset must not shrink num_replicas: the engine sizes its
        consumer-id offset from it, so a shrunk value re-maps survivors onto each
        other."""
        p = build_rdt_init_payloads({}, ["http://e0", "http://e3"], 1, slots=[0, 3], num_replicas=4)
        assert _replicas(p) == 4
        assert _ranks(p) == {"http://e0": 0, "http://e3": 3}

    def test_a_slot_count_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="slots for"):
            build_rdt_init_payloads({}, self.URLS, 1, slots=[0, 1])

    def test_a_rank_outside_num_replicas_is_rejected(self):
        """The mistake this catches is forgetting `num_replicas=` on a partial
        re-init, which would otherwise be a silent slice collision."""
        with pytest.raises(ValueError, match="must be addressable"):
            build_rdt_init_payloads({}, ["http://e3-new"], 1, slots=[3])

    def test_an_empty_url_list_is_not_an_error(self):
        """Nothing to init is a legitimate no-op (a fleet where every engine is
        mid-restart), and must not trip the rank bound check."""
        assert build_rdt_init_payloads({}, [], 1, slots=[], num_replicas=4) == []


class _StubActor:
    """Stands in for a ``VLLMServerActor`` handle: only ``start`` is exercised."""

    def __init__(self, ip="10.0.0.1", port=8000):
        self._info = ServerInfo(ip=ip, port=port)

    class _Remote:
        def __init__(self, value):
            self._value = value

        def remote(self):
            return self._value

    @property
    def start(self):
        return self._Remote(self._info)


class TestPoolSlots:
    """``ServerActorPool`` stamps slots; actors never learn their own."""

    @staticmethod
    def _pool(slots=None, ports=(8000, 8100)):
        actors = [_StubActor(port=p) for p in ports]
        pool = ServerActorPool(actors, slots=slots)
        # `start` returns the stub's ServerInfo directly, so no ray.get is involved.
        pool._server_infos = pool._stamp([a.start.remote() for a in actors])
        return pool

    def test_slots_default_to_the_actor_index(self):
        pool = self._pool()
        assert [i.slot for i in pool.server_infos] == [0, 1]

    def test_dp_peers_share_one_slot(self):
        """Two servers of one deployment: they share a parallel config in which vLLM's
        own data_parallel_index separates them, so one replica_rank is correct."""
        pool = self._pool(slots=[3, 3])
        assert [i.slot for i in pool.server_infos] == [3, 3]

    def test_a_slot_count_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="one slot per actor"):
            ServerActorPool([_StubActor()], slots=[0, 1])

    def test_replace_actor_keeps_the_slot_and_takes_the_new_url(self):
        """The restart contract in one assertion: new address, same identity."""
        pool = self._pool(slots=[5, 6])
        fresh = _StubActor(port=8107)
        info = pool.replace_actor(1, fresh, ServerInfo(ip="10.0.0.1", port=8107))
        assert info.slot == 6
        assert info.url == "http://10.0.0.1:8107"
        assert pool.get_actors()[1] is fresh
        assert [i.slot for i in pool.server_infos] == [5, 6]
        assert pool.get_server_urls() == ["http://10.0.0.1:8000", "http://10.0.0.1:8107"]

    def test_replace_actor_overrides_a_caller_supplied_slot(self):
        """The pool owns the mapping, so a caller cannot corrupt it by passing a
        stale or invented slot on the new info."""
        pool = self._pool(slots=[5, 6])
        info = pool.replace_actor(1, _StubActor(port=8107), ServerInfo(ip="1.2.3.4", port=8107, slot=99))
        assert info.slot == 6

    def test_replace_actor_rejects_an_out_of_range_index(self):
        pool = self._pool()
        with pytest.raises(IndexError):
            pool.replace_actor(7, _StubActor(), ServerInfo(ip="1.2.3.4", port=1))

    def test_a_dead_start_ref_is_never_awaited_again(self):
        """`_start_refs` still holds the dead actor's ref. Leaving it in place would
        make the next lazy resolution ray.get a ref belonging to a killed actor."""
        actors = [_StubActor(port=8000)]
        pool = ServerActorPool(actors, slots=[0])
        pool._start_refs = ["a-ref-for-a-now-dead-actor"]
        pool._server_infos = [ServerInfo(ip="10.0.0.1", port=8000, slot=0)]
        pool.replace_actor(0, _StubActor(port=8001), ServerInfo(ip="10.0.0.1", port=8001))
        assert pool._start_refs == []
        assert pool.get_server_urls() == ["http://10.0.0.1:8001"]


def _client(urls, slots=None, dp=1, ft=True):
    return RemoteInferenceClient(
        proxy_url="http://proxy",
        server_urls=list(urls),
        server_slots=slots,
        data_parallel_size=dp,
        fault_tolerance=InferenceFaultToleranceConfig(enabled=ft) if ft else None,
    )


class TestClientSlotViews:
    URLS = ["http://e0", "http://e1", "http://e2", "http://e3"]

    def test_slots_default_to_the_positional_derivation(self):
        c = _client(self.URLS)
        assert c.server_slots == [0, 1, 2, 3]
        assert c.num_provisioned_replicas == 4

    def test_dp_collapses_pairs_onto_one_slot(self):
        c = _client(self.URLS, dp=2)
        assert c.server_slots == [0, 0, 1, 1]
        assert c.num_provisioned_replicas == 2

    def test_a_slot_count_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="one slot per server URL"):
            _client(self.URLS, slots=[0, 1])

    def test_live_url_slots_tracks_the_dead_set(self):
        c = _client(self.URLS)
        assert c.live_url_slots == [(u, i) for i, u in enumerate(self.URLS)]
        c._disabled_server_urls.add("http://e1")
        assert c.live_url_slots == [("http://e0", 0), ("http://e2", 2), ("http://e3", 3)]

    def test_num_provisioned_replicas_ignores_deaths(self):
        """It is the denominator of the consumer-id arithmetic, so it must not move
        when engines die -- that is the whole point of not compacting slots."""
        c = _client(self.URLS)
        c._disabled_server_urls.update({"http://e1", "http://e2"})
        assert c.num_provisioned_replicas == 4

    def test_slot_of_resolves_a_provisioned_url(self):
        c = _client(self.URLS, dp=2)
        assert c.slot_of("http://e2") == 1
        with pytest.raises(KeyError):
            c.slot_of("http://nope")

    def test_replace_server_url_moves_a_slot_in_place(self):
        c = _client(self.URLS)
        c._disabled_server_urls.add("http://e2")
        gen = c.membership_generation
        c.replace_server_url(2, "http://e2-new:8201")

        assert c.server_urls == ["http://e0", "http://e1", "http://e2-new:8201", "http://e3"]
        assert c.server_slots == [0, 1, 2, 3]
        assert c.num_provisioned_replicas == 4
        # The replacement is live for the control plane and the weight sync, which is
        # exactly what it needs in order to take part in the sync it is waiting for.
        assert "http://e2-new:8201" in c.active_server_urls
        assert c.disabled_server_urls == []
        assert c.membership_generation == gen + 1

    def test_replace_server_url_rejects_an_unknown_slot(self):
        c = _client(self.URLS)
        with pytest.raises(KeyError, match="slot 9 is not provisioned"):
            c.replace_server_url(9, "http://whatever")

    def test_replace_server_url_rejects_a_url_another_slot_owns(self):
        """Two slots answering on one address means requests for two consumer blocks
        land on one engine; the second block never gets its weights."""
        c = _client(self.URLS)
        with pytest.raises(ValueError, match="already registered for slot"):
            c.replace_server_url(2, "http://e0")

    def test_replacing_a_slot_with_its_own_url_is_allowed(self):
        """A restart that happens to re-win its old port is the common case."""
        c = _client(self.URLS)
        c._disabled_server_urls.add("http://e2")
        c.replace_server_url(2, "http://e2")
        assert c.active_server_urls == self.URLS
