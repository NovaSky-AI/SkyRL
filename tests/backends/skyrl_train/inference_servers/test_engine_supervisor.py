"""The engine supervisor: restarting a dead inference engine and rejoining it.

Part 1 shrinks the fleet and carries on. Part 2's supervisor is the policy layer that
brings the slot back, and its job is almost entirely about ORDER: the wrong order here
does not raise, it silently corrupts a training batch. Two orderings carry that risk
and both are pinned below.

  * **De-route before killing.** A dead engine still in the router keeps receiving
    requests. Killing first and de-routing later leaves a window whose width is
    however long the Ray call takes.
  * **Sync before re-admitting.** A replacement loads CHECKPOINT weights. The
    prefix-cache salt is keyed on the weight version, so a stale engine serving
    rollouts produces plausible-looking wrong data rather than an error. It must not
    enter the router until a weight sync has made it current.

The other property worth its own tests is that a WEDGED engine -- alive to Ray,
holding its GPUs, never answering ``/health`` -- is normalized to a dead one. Without
the kill, a replacement would contend for the same bundle; with it, the wedge becomes
the connection-refused case Part 1 already handles.

CPU-only. ``ServerGroup`` and the Ray actors are stand-ins: what is under test is the
supervisor's decisions, not Ray's ability to start a vLLM server.

Run:
    uv run --extra dev pytest tests/backends/skyrl_train/inference_servers/test_engine_supervisor.py
"""

from typing import List, Optional

import pytest

from skyrl.backends.skyrl_train.inference_servers.common import ServerInfo
from skyrl.backends.skyrl_train.inference_servers.engine_supervisor import (
    EngineSupervisor,
    SlotState,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.train.config.config import InferenceFaultToleranceConfig


class _FakeActor:
    """A server actor that is either healthy, wedged, or gone.

    ``wedge()`` is the interesting state: the actor is ALIVE and holds its GPUs, but
    ``/health`` never answers. That is what the supervisor has to normalize into a
    death, since a replacement cannot share the bundle with it.
    """

    def __init__(self, port: int, trace: List[str], slot: int):
        self.port = port
        self.slot = slot
        self._trace = trace
        self.healthy = True
        self.killed = False

    def wedge(self) -> None:
        self.healthy = False

    @property
    def url(self) -> str:
        return f"http://10.0.0.1:{self.port}"


class _FakeGroup:
    """Stands in for ``ServerGroup``: hands out actors and restarts them in place."""

    def __init__(self, slot: int, trace: List[str], start_port: int = 8000, fail_restarts: int = 0):
        self.slot = slot
        self._trace = trace
        self._start_port = start_port
        self._restart_count = 0
        self._fail_restarts = fail_restarts
        self._actors = [_FakeActor(start_port, trace, slot)]
        # Set by a test to make the pending start ref never resolve.
        self.hang_restart = False
        # Set by a test to make the replacement's start() raise.
        self.start_raises = None
        # None = the reaper cannot scope itself and no-ops (see _get_gpu_ids_for_server).
        self.gpu_ids = None

    def get_actors(self):
        return self._actors

    def _get_gpu_ids_for_server(self, server_idx: int):
        """Physical GPU ids, which the orphan reaper needs to scope its kill.

        None here means "unresolvable", which makes the reaper a deliberate no-op --
        these tests are about the supervisor's decisions, and reaping is covered
        exhaustively (and without a GPU) in test_engine_orphans.py."""
        return self.gpu_ids

    def begin_restart(self, server_idx: int):
        self._restart_count += 1
        self._trace.append(f"begin_restart(slot={self.slot})")
        if self._restart_count <= self._fail_restarts:
            raise RuntimeError("no bundle available")
        # A restart re-reserves a port inside its own stride window, so the URL moves.
        port = self._start_port + self._restart_count
        actor = _FakeActor(port, self._trace, self.slot)
        if self.start_raises is not None:
            return actor, _Pending(self.start_raises)
        ref = _Pending(None if self.hang_restart else ServerInfo(ip="10.0.0.1", port=port))
        return actor, ref

    def finish_restart(self, server_idx: int, actor, info: ServerInfo) -> ServerInfo:
        self._trace.append(f"finish_restart(slot={self.slot},{info.port})")
        self._actors[server_idx] = actor
        return ServerInfo(ip=info.ip, port=info.port, slot=self.slot)


class _Pending:
    """A stand-in ObjectRef. ``value is None`` means "still loading"."""

    def __init__(self, value):
        self.value = value


def _install_ray_stubs(monkeypatch, trace, actors_by_ref=None):
    """Replace the three ray calls the supervisor makes."""
    from skyrl.backends.skyrl_train.inference_servers import engine_supervisor as es

    def _wait(refs, timeout=0):
        ready = [r for r in refs if getattr(r, "value", None) is not None]
        return ready, [r for r in refs if r not in ready]

    def _get(ref):
        if isinstance(ref, _Pending):
            if isinstance(ref.value, Exception):
                raise ref.value
            return ref.value
        return ref

    def _kill(actor):
        trace.append(f"kill(slot={getattr(actor, 'slot', '?')})")
        if getattr(actor, "killed", False):
            raise ValueError("actor already dead")
        actor.killed = True

    monkeypatch.setattr(es.ray, "wait", _wait)
    monkeypatch.setattr(es.ray, "get", _get)
    monkeypatch.setattr(es.ray, "kill", _kill)


class _Fleet:
    """A client + supervisor pair wired to fake groups, with a recorded trace."""

    def __init__(self, monkeypatch, n=4, ft=None, fail_restarts=0):
        self.trace: List[str] = []
        self.groups = [_FakeGroup(i, self.trace, 8000 + i * 100, fail_restarts) for i in range(n)]
        urls = [g.get_actors()[0].url for g in self.groups]
        self.client = RemoteInferenceClient(
            proxy_url="http://proxy",
            server_urls=urls,
            server_slots=list(range(n)),
            data_parallel_size=1,
            fault_tolerance=ft
            or InferenceFaultToleranceConfig(enabled=True, restart_dead_engines=True, min_live_engines=1),
        )
        # Router calls and health probes are the client's; record rather than perform.
        self.routed = set(urls)

        async def _remove(url):
            self.trace.append(f"router_remove({url})")
            self.routed.discard(url)
            return True

        async def _add(url):
            self.trace.append(f"router_add({url})")
            self.routed.add(url)
            return self._router_accepts

        async def _probe(url):
            actor = self._actor_at(url)
            ok = actor is not None and actor.healthy and not actor.killed
            self.trace.append(f"probe({url})->{ok}")
            return ok

        self._router_accepts = True
        self.client._router_remove_worker = _remove
        self.client._router_add_worker = _add
        self.client._probe_health = _probe

        self.rdt_inits: List[tuple] = []

        async def _init_rdt(payload, targets=None):
            self.trace.append(f"rdt_init({targets})")
            self.rdt_inits.append((payload, tuple(targets or ())))
            if self._rdt_init_fails:
                raise RuntimeError("bake failed")
            return {}

        self._rdt_init_fails = False
        self.client.init_weight_transfer_engine_rdt = _init_rdt

        _install_ray_stubs(monkeypatch, self.trace)
        self.sup = EngineSupervisor(self.client, self.groups, data_parallel_size=1)
        self.sup.set_rdt_worker_init({"num_consumers": n})

    def _actor_at(self, url) -> Optional[_FakeActor]:
        for g in self.groups:
            for a in g.get_actors():
                if a.url == url:
                    return a
        return None

    def actor(self, slot: int) -> _FakeActor:
        return self.groups[slot].get_actors()[0]

    def states(self):
        return {s.index: s.state for s in self.sup.slots}


@pytest.mark.asyncio
class TestReconcile:
    async def test_a_whole_fleet_reconciles_to_nothing(self, monkeypatch):
        f = _Fleet(monkeypatch)
        await f.sup.reconcile()
        assert all(st is SlotState.LIVE for st in f.states().values())
        assert not any(t.startswith("begin_restart") for t in f.trace)
        assert f.client.active_server_urls == f.client.server_urls

    async def test_a_dead_engine_is_de_routed_before_it_is_killed(self, monkeypatch):
        """The ordering: a corpse still in the router keeps taking requests, so the
        router call must not wait on a Ray call."""
        f = _Fleet(monkeypatch)
        dead_url = f.actor(2).url
        f.actor(2).wedge()
        ft = f.client.fault_tolerance
        for _ in range(ft.health_failure_threshold):
            await f.sup.reconcile()

        assert f.trace.index(f"router_remove({dead_url})") < f.trace.index("kill(slot=2)"), f.trace
        assert dead_url not in f.routed

    async def test_the_supervisor_trusts_the_fleet_probe(self, monkeypatch):
        """One evidence standard, in one place.

        The supervisor used to keep its own consecutive-failure counter because the
        reactive detector condemned on a single short probe, and this layer tried to
        repair the resulting bad verdicts. The probe now distinguishes a definitive
        failure from an ambiguous timeout and retries only the latter
        (test_engine_fault_tolerance.TestProbeAccuracy), so a negative answer here is
        already well-evidenced and is acted on directly."""
        f = _Fleet(monkeypatch)
        f.actor(1).wedge()
        await f.sup.reconcile()
        assert f.states()[1] is SlotState.DEAD or "kill(slot=1)" in f.trace
        assert "probe(http://10.0.0.1:8100)->False" in f.trace

    async def test_a_healthy_engine_is_left_alone(self, monkeypatch):
        """The negative case: nothing may be condemned while it answers."""
        f = _Fleet(monkeypatch)
        await f.sup.reconcile()
        assert all(st is SlotState.LIVE for st in f.states().values())
        assert not any(t.startswith("kill(") for t in f.trace)
        assert not any(t.startswith("begin_restart") for t in f.trace)

    async def test_the_reactive_verdict_is_adopted_as_is(self, monkeypatch):
        """A death the fleet probe already found during generation must not be
        re-litigated here -- the supervisor reads the dead set and acts."""
        f = _Fleet(monkeypatch)
        dead_url = f.actor(0).url
        f.actor(0).wedge()
        f.client.mark_dead(dead_url)
        await f.sup.reconcile()
        assert f.states()[0] is SlotState.DEAD
        assert "kill(slot=0)" in f.trace

    async def test_a_wedged_engine_is_killed_so_its_bundle_is_free(self, monkeypatch):
        """The point of normalizing a wedge to a death: the actor is ALIVE and holds
        its GPUs, so a replacement in the same bundle would contend with it."""
        f = _Fleet(monkeypatch)
        victim = f.actor(3)
        victim.wedge()
        for _ in range(f.client.fault_tolerance.health_failure_threshold):
            await f.sup.reconcile()
        assert victim.killed is True

    async def test_a_restarted_engine_still_does_re_init(self, monkeypatch):
        """The other side of that call: a genuine restart has nothing initialized, so it
        MUST re-init or it cannot pull its slices."""
        f = _Fleet(
            monkeypatch,
            ft=InferenceFaultToleranceConfig(enabled=True, restart_dead_engines=True, health_failure_threshold=1),
        )
        f.actor(2).wedge()
        await f.sup.reconcile()
        await f.sup.reconcile()
        assert f.states()[2] is SlotState.PENDING_SYNC
        admitted = await f.sup.before_weight_sync()
        assert [x.index for x in admitted] == [2]
        assert len(f.rdt_inits) == 1


@pytest.mark.asyncio
class TestRestartAndRejoin:
    async def _kill_and_restart(self, f, slot=2):
        f.actor(slot).wedge()
        for _ in range(f.client.fault_tolerance.health_failure_threshold):
            await f.sup.reconcile()
        await f.sup.reconcile()  # harvest
        return f

    async def test_a_restart_keeps_the_slot_and_takes_a_new_url(self, monkeypatch):
        f = _Fleet(monkeypatch)
        old_url = f.actor(2).url
        await self._kill_and_restart(f)

        s = f.sup.slot(2)
        assert s.state is SlotState.PENDING_SYNC
        assert s.index == 2
        assert s.url != old_url
        assert f.client.server_slots == [0, 1, 2, 3]
        assert f.client.server_urls[2] == s.url

    async def test_a_replacement_is_in_the_sync_set_but_not_the_router(self, monkeypatch):
        """The core safety property. It must reach the weight sync (that is what it is
        waiting for) and must NOT reach a rollout (its weights are the checkpoint's)."""
        f = _Fleet(monkeypatch)
        await self._kill_and_restart(f)
        new_url = f.sup.slot(2).url

        assert new_url in f.client.active_server_urls, "cannot take part in the sync it needs"
        assert new_url not in f.routed, "a stale engine is reachable by rollouts"

    async def test_rejoin_requires_a_successful_sync(self, monkeypatch):
        """before_weight_sync re-inits; only after_weight_sync routes. Nothing between
        them may put a stale engine in the router."""
        f = _Fleet(monkeypatch)
        await self._kill_and_restart(f)

        admitted = await f.sup.before_weight_sync()
        assert [s.index for s in admitted] == [2]
        assert f.sup.slot(2).url not in f.routed, "routed before the sync completed"

        await f.sup.after_weight_sync(admitted)
        assert f.sup.slot(2).url in f.routed
        assert f.states()[2] is SlotState.LIVE

    async def test_the_re_init_is_stamped_with_the_slot_not_the_position(self, monkeypatch):
        """A single-target re-init derived positionally would be replica_rank 0 -- it
        would bake a plan for deployment 0's consumer ids and pull another engine's
        slices. The slot is what makes it a rejoin."""
        f = _Fleet(monkeypatch)
        await self._kill_and_restart(f, slot=2)
        await f.sup.before_weight_sync()

        assert len(f.rdt_inits) == 1
        _payload, targets = f.rdt_inits[0]
        assert targets == ((f.sup.slot(2).url, 2),)

    async def test_a_failed_re_init_goes_back_to_dead_rather_than_into_the_sync(self, monkeypatch):
        """A consumer that has not baked cannot pull. Letting it into the sync would
        stall every producer bound to its ids -- worse than staying dead."""
        f = _Fleet(monkeypatch)
        await self._kill_and_restart(f)
        f._rdt_init_fails = True

        admitted = await f.sup.before_weight_sync()
        assert admitted == []
        assert f.states()[2] is SlotState.DEAD
        assert f.sup.slot(2).url not in f.routed

    async def test_without_a_cached_worker_init_nothing_is_admitted(self, monkeypatch):
        """The payload comes from the trainer after init_weight_sync_state. Until it
        exists a replacement can run but cannot receive weights."""
        f = _Fleet(monkeypatch)
        f.sup.set_rdt_worker_init(None)
        await self._kill_and_restart(f)
        assert await f.sup.before_weight_sync() == []
        assert f.states()[2] is SlotState.PENDING_SYNC
        assert f.rdt_inits == []

    async def test_a_router_refusal_leaves_it_current_but_unrouted(self, monkeypatch):
        """Costs one step of capacity, not the run: retried at the next boundary."""
        f = _Fleet(monkeypatch)
        await self._kill_and_restart(f)
        admitted = await f.sup.before_weight_sync()
        f._router_accepts = False
        await f.sup.after_weight_sync(admitted)
        assert f.states()[2] is SlotState.PENDING_SYNC

        f._router_accepts = True
        await f.sup.after_weight_sync(admitted)
        assert f.states()[2] is SlotState.LIVE

    async def test_a_slot_that_dies_during_the_sync_is_not_re_admitted(self, monkeypatch):
        f = _Fleet(monkeypatch)
        await self._kill_and_restart(f)
        admitted = await f.sup.before_weight_sync()
        f.sup.slot(2).state = SlotState.DEAD  # died inside the sync window
        await f.sup.after_weight_sync(admitted)
        assert f.sup.slot(2).url not in f.routed


@pytest.mark.asyncio
class TestBudgetsAndFloors:
    async def test_the_restart_budget_is_bounded(self, monkeypatch):
        """A slot that keeps dying is usually a bad GPU. Exhaustion is a permanent
        dead slot, not an error -- the run can still finish degraded."""
        f = _Fleet(
            monkeypatch,
            ft=InferenceFaultToleranceConfig(
                enabled=True, restart_dead_engines=True, max_restarts_per_engine=2, health_failure_threshold=1
            ),
        )
        for _ in range(6):
            # Each replacement comes up wedged, so it dies again immediately.
            for a in f.groups[1].get_actors():
                a.healthy = False
            await f.sup.reconcile()

        assert f.sup.slot(1).restarts == 2
        assert f.sup.slot(1).state is SlotState.DEAD
        assert sum(t.startswith("begin_restart(slot=1)") for t in f.trace) == 2
        # The permanent-death notice is a transition, not a per-reconcile refrain: the
        # first GPU demo logged it four times for one slot, burying the real events.
        assert f.sup.slot(1).budget_exhausted_logged is True

    async def test_a_gpu_that_is_still_held_does_not_cost_budget(self, monkeypatch):
        """Found on hardware. The dead engine's worker kept ~62 of 79 GiB, the replacement
        needed 55.4, and all three attempts died on free memory -- condemning the slot for
        a condition external to it.

        The verdict now comes from a MEASUREMENT taken when we launched, not from the
        exception text. That matters because vLLM raises a generic "Engine core
        initialization failed" and leaves the real "Free memory on device ..." in a
        subprocess log: the string matcher this replaces was tested against the text it
        was assumed to carry, and never fired once in a real run."""
        f = _Fleet(
            monkeypatch,
            ft=InferenceFaultToleranceConfig(
                enabled=True, restart_dead_engines=True, max_restarts_per_engine=3, health_failure_threshold=1
            ),
        )
        f.groups[1].start_raises = RuntimeError("Engine core initialization failed. See root cause above.")
        # What the clean pass concludes when the bundle is still occupied.
        monkeypatch.setattr(
            type(f.sup),
            "_reap_orphans_for",
            lambda _self, slot: setattr(slot, "bundle_has_room", False),
            raising=True,
        )
        f.actor(1).wedge()
        for _ in range(6):
            await f.sup.reconcile()

        assert f.sup.slot(1).restarts == 0, "an unavailable GPU consumed the restart budget"
        assert "begin_restart(slot=1)" in f.trace, "stopped trying entirely"

        # Once the bundle is genuinely clean, the very next boundary brings it back.
        monkeypatch.setattr(
            type(f.sup),
            "_reap_orphans_for",
            lambda _self, slot: setattr(slot, "bundle_has_room", True),
            raising=True,
        )
        f.groups[1].start_raises = None
        await f.sup.reconcile()
        assert f.sup.slot(1).state is SlotState.PENDING_SYNC

    async def test_a_real_start_failure_still_costs_budget(self, monkeypatch):
        """The other side of that call: a replacement that fails for its own reasons
        must still be bounded, or a genuinely broken slot relaunches forever."""
        f = _Fleet(
            monkeypatch,
            ft=InferenceFaultToleranceConfig(
                enabled=True, restart_dead_engines=True, max_restarts_per_engine=2, health_failure_threshold=1
            ),
        )
        f.groups[1].start_raises = RuntimeError("CUDA error: device-side assert triggered")
        monkeypatch.setattr(
            type(f.sup),
            "_reap_orphans_for",
            lambda _self, slot: setattr(slot, "bundle_has_room", True),
            raising=True,
        )
        f.actor(1).wedge()
        for _ in range(6):
            await f.sup.reconcile()
        assert f.sup.slot(1).restarts == 2
        assert f.sup.slot(1).state is SlotState.DEAD

    async def test_a_restart_that_cannot_launch_costs_budget(self, monkeypatch):
        """Otherwise an unschedulable bundle spins forever."""
        f = _Fleet(
            monkeypatch,
            ft=InferenceFaultToleranceConfig(
                enabled=True, restart_dead_engines=True, max_restarts_per_engine=2, health_failure_threshold=1
            ),
            fail_restarts=99,
        )
        f.actor(0).wedge()
        for _ in range(5):
            await f.sup.reconcile()
        assert f.sup.slot(0).restarts == 2
        assert f.sup.slot(0).state is SlotState.DEAD

    async def test_before_generation_only_reconciles(self, monkeypatch):
        """The floor-wait is gone deliberately. ``min_live_engines`` is enforced once,
        in the fleet probe that produced the evidence (``_probe_and_disable`` raises
        ``EngineFleetError`` the moment a probe takes the fleet below it). A second check
        here, with its own waiting loop, was a duplicate -- and its interaction with the
        one-pass relaunch deferral had already caused one regression."""
        ft = InferenceFaultToleranceConfig(
            enabled=True, restart_dead_engines=True, min_live_engines=2, health_failure_threshold=1
        )
        f = _Fleet(monkeypatch, n=2, ft=ft, fail_restarts=99)
        f.actor(0).wedge()

        slept = []
        monkeypatch.setattr("asyncio.sleep", lambda s_: slept.append(s_))

        # Below the floor after the kill, and no restart can succeed -- yet this must
        # neither wait nor raise; generating against a small fleet is the probe's call.
        await f.sup.before_generation()

        assert slept == [], "before_generation is still waiting on recovery"
        assert "probe(http://10.0.0.1:8000)->False" in f.trace, "did not even reconcile"

    async def test_a_reconcile_inside_the_sync_does_not_enforce_the_floor(self, monkeypatch):
        """A floor violation raised at the sync boundary would fail the sync -- turning
        a recoverable capacity problem into a dead run. It belongs to generation."""
        ft = InferenceFaultToleranceConfig(
            enabled=True, restart_dead_engines=True, min_live_engines=2, health_failure_threshold=1
        )
        f = _Fleet(monkeypatch, n=2, ft=ft, fail_restarts=99)
        f.actor(0).wedge()
        assert await f.sup.before_weight_sync() == []  # no raise


@pytest.mark.asyncio
class TestDisabledIsInert:
    async def test_restart_disabled_leaves_every_hook_a_no_op(self, monkeypatch):
        """Part 1 behaviour must be reachable with FT on and restarts off."""
        ft = InferenceFaultToleranceConfig(enabled=True, restart_dead_engines=False)
        f = _Fleet(monkeypatch, ft=ft)
        assert f.sup.enabled is False
        assert f.sup.slots == []
        f.actor(0).wedge()
        await f.sup.reconcile()
        await f.sup.before_generation()
        assert await f.sup.before_weight_sync() == []
        await f.sup.after_weight_sync()
        assert f.trace == [], f.trace

    async def test_ft_disabled_entirely_is_also_inert(self, monkeypatch):
        f = _Fleet(monkeypatch, ft=InferenceFaultToleranceConfig(enabled=False, restart_dead_engines=True))
        assert f.sup.enabled is False
        await f.sup.reconcile()
        assert f.trace == []


def test_a_slot_without_an_owning_group_is_rejected(monkeypatch):
    """The supervisor can only restart engines this process provisioned; a slot with
    no group would silently never be restarted."""
    trace: List[str] = []
    groups = [_FakeGroup(0, trace), _FakeGroup(1, trace, 8100)]
    client = RemoteInferenceClient(
        proxy_url="http://proxy",
        server_urls=["http://a", "http://b", "http://c"],
        server_slots=[0, 1, 2],
        data_parallel_size=1,
        fault_tolerance=InferenceFaultToleranceConfig(enabled=True, restart_dead_engines=True),
    )
    with pytest.raises(ValueError, match="No server group owns slot 2"):
        EngineSupervisor(client, groups, data_parallel_size=1)
