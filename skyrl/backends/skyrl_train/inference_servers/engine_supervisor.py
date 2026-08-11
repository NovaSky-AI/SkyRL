"""Engine fault tolerance, Part 2: restore the fleet.

Part 1 survives an inference-engine death — the fleet shrinks, the router stops
routing to the corpse, and the RDT weight sync gathers-and-drops the groups bound to
its consumers. The slot then stays dead for the rest of the run. This module brings
it back.

**Policy lives here; mechanism stays where it was.** Restarting an actor is
``ServerGroup.begin_restart``, router membership is ``RemoteInferenceClient``'s
worker-management calls, probing is its ``_probe_health``. The supervisor only decides
*when*, and holds the one piece of state none of them can: which slot is in what
condition, and how many times it has been relaunched.

**Why a plain driver-side object and not a named Ray actor.** The membership state has
exactly one writer (this object, on the driver, inside the training loop's own
awaits), and the driver's death ends the job regardless — so an actor would add a
failure mode and a serialization boundary in exchange for nothing.

**Why explicit calls and not the callback hooks.** Fault tolerance is control flow:
``before_generation`` can refuse to proceed, ``before_weight_sync`` can add engines to
the sync, ``after_weight_sync`` can re-admit them. The existing hook points are
synchronous observers and cannot do any of that.

**The rejoin rule, which is the subtle part.** A replacement loads weights from
*checkpoint*, so it is stale the moment training has stepped, and the prefix-cache
salt is keyed on the weight version — a stale engine serving rollouts would silently
poison the batch rather than fail. So a restarted engine is deliberately:

  * IN the client's active set, because that is what the control plane and the weight
    sync fan out over, and taking part in a sync is exactly what it is waiting to do;
  * OUT of the router, so no rollout can reach it, until that sync has completed.

Since syncs run every step, the sit-out is about one step. An immediate dedicated
sync was rejected: it needs a second pause/broadcast path and interrupts in-flight
generation to save that one step.

Everything here is inert unless ``fault_tolerance.enabled`` **and**
``fault_tolerance.restart_dead_engines``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import ray
from loguru import logger
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.backends.skyrl_train.inference_servers.server_group import ServerGroup


def _reap_on_node(gpu_ids: List[int], bundle_pids: List[int], need_bytes: int = 0) -> dict:
    """Clean a replica's node: kill its bundle's processes, sweep orphans, report room.

    Runs on the node holding ``gpu_ids``. ``bundle_pids`` are the pids the DRIVER
    resolved from Ray's actor table for this slot's placement-group bundles -- the
    precise membership. The orphan sweep stays as a second pass because it catches what
    the bundle cannot: an mp-spawned ``EngineCore`` is not a Ray actor, so it appears in
    no bundle, and under the Ray executor it is the OWNER of the worker rather than its
    OS parent.

    Returns the pids killed and the post-clean free memory, so the caller can say plainly
    whether the bundle is actually usable again instead of inferring it from a later
    failure message.
    """
    from skyrl.backends.skyrl_train.inference_servers.engine_orphans import (
        gpu_uuids_for_ids,
        kill_replica_processes,
        reap_orphaned_engines,
        wait_for_gpu_room,
    )

    uuids = gpu_uuids_for_ids(gpu_ids)
    killed = list(kill_replica_processes(bundle_pids)) if bundle_pids else []
    killed += list(reap_orphaned_engines(uuids))
    # WAIT for the memory rather than sampling it: the kill is asynchronous, so an
    # immediate reading says nothing (see wait_for_gpu_room).
    free, waited = wait_for_gpu_room(uuids, need_bytes)
    return {"killed": killed, "free": free, "waited_s": waited}


class SlotState(str, Enum):
    """Condition of one deployment slot.

    ``PENDING_SYNC`` is the state that makes a rejoin safe: healthy and reachable, but
    carrying stale checkpoint weights, so it is kept out of the router until a weight
    sync has brought it current. Everything else is what it sounds like.
    """

    LIVE = "live"
    DEAD = "dead"
    RESTARTING = "restarting"
    PENDING_SYNC = "pending_sync"


@dataclass
class EngineSlot:
    """One deployment's identity and condition.

    ``index`` is the provisioned ordinal, which is simultaneously the RDT
    ``replica_rank`` and the slot the restart reuses. ``url`` is merely where the slot
    currently answers — it CHANGES across restarts, which is the entire reason
    identity is the index (see ``ServerInfo.slot``).
    """

    index: int
    group: "ServerGroup"
    server_idx: int
    url: str
    state: SlotState = SlotState.LIVE
    restarts: int = 0
    bundle_has_room: Optional[bool] = None
    """Whether this slot's GPUs had room for an engine after the last clean.

    ``None`` means unknown (no reading). Set from an actual memory measurement rather
    than inferred from a failure message: vLLM reports "Engine core initialization
    failed" and leaves the real "Free memory on device ..." in a subprocess log, so the
    exception text can never be classified -- a string matcher on it was written, tested
    against the text it was ASSUMED to carry, and never once fired in a real run."""
    budget_exhausted_logged: bool = False
    """So the permanent-death notice is logged on the transition, not every reconcile."""
    killed_this_pass: bool = False
    """Set when THIS reconcile killed the actor, so the relaunch waits one pass for the
    bundle's GPU memory to come back. See ``EngineSupervisor._retire_dead``."""
    _pending: Optional[Tuple[Any, Any, float]] = field(default=None, repr=False)
    """``(actor, start_ref, deadline)`` for an in-flight restart, or None."""

    @property
    def is_reachable(self) -> bool:
        """Whether the control plane and the weight sync should include this slot."""
        return self.state in (SlotState.LIVE, SlotState.PENDING_SYNC)


class EngineSupervisor:
    """Keeps the provisioned engine fleet as close to whole as it can.

    Call sites in the training loop, all no-ops when disabled:

    * ``before_generation()`` — reconcile, so a death during training is found
      before a batch is committed to the fleet.
    * ``before_weight_sync()`` — reconcile and RDT-re-init anything that restarted,
      so this sync includes it.
    * ``after_weight_sync()`` — the engines that took part are now current: re-admit
      them to the router.

    Part 1's reactive probe stays the fast path for deaths during generation; these
    step-boundary calls are what covers the synchronous training and sync phases the
    reactive path cannot see. No periodic poller: with a reactive probe on one side
    and step boundaries on the other, a timer would only add a background task and
    give up Part 1's "no heartbeat monitor" property.
    """

    def __init__(
        self,
        client: "RemoteInferenceClient",
        server_groups: Sequence["ServerGroup"],
        data_parallel_size: int = 1,
    ) -> None:
        self._client = client
        self._ft = client.fault_tolerance
        self._dp = max(1, int(data_parallel_size))
        # Set by `set_rdt_worker_init` once the trainer has produced it; until then a
        # restarted engine can be relaunched but not re-initialized for weight sync.
        self._rdt_worker_init: Optional[Dict[str, Any]] = None
        self._slots: List[EngineSlot] = []
        if self.enabled:
            self._slots = self._build_slots(server_groups)
            logger.info(
                f"[ft] engine supervisor managing {len(self._slots)} slot(s); "
                f"max_restarts_per_engine={self._ft.max_restarts_per_engine}, "
                f"restart_timeout_s={self._ft.restart_timeout_s}"
            )

    # ---- construction ----

    def _build_slots(self, server_groups: Sequence["ServerGroup"]) -> List[EngineSlot]:
        """Map provisioned ``(url, slot)`` onto the group that can restart each.

        A group's ``slot`` is its deployment ordinal, so this is a lookup rather than
        arithmetic — deliberately, since the alternative (re-deriving the group index
        from a position in ``server_urls``) is the exact class of mistake slots exist
        to remove.
        """
        by_slot = {g.slot: g for g in server_groups}
        slots: List[EngineSlot] = []
        for url, slot in self._client.url_slots:
            group = by_slot.get(slot)
            if group is None:
                raise ValueError(
                    f"No server group owns slot {slot} (groups: {sorted(by_slot)}). The engine "
                    "supervisor can only restart engines this process provisioned."
                )
            # dp=1 under every fault-tolerant config (validate_cfg enforces it), so
            # the server's index inside its group is 0. Computed rather than assumed
            # so the assertion is visible if that gate ever moves.
            server_idx = [u for u, s in self._client.url_slots if s == slot].index(url)
            slots.append(EngineSlot(index=slot, group=group, server_idx=server_idx, url=url))
        return slots

    # ---- views ----

    @property
    def enabled(self) -> bool:
        """True only when FT is on AND restarts are wanted."""
        return bool(self._ft is not None and self._ft.enabled and self._ft.restart_dead_engines)

    @property
    def slots(self) -> List[EngineSlot]:
        return list(self._slots)

    def slot(self, index: int) -> EngineSlot:
        for s in self._slots:
            if s.index == index:
                return s
        raise KeyError(f"no slot {index} (have {[s.index for s in self._slots]})")

    def _in_state(self, *states: SlotState) -> List[EngineSlot]:
        return [s for s in self._slots if s.state in states]

    def describe(self) -> str:
        """One-line fleet summary for the driver log."""
        counts: Dict[str, int] = {}
        for s in self._slots:
            counts[s.state.value] = counts.get(s.state.value, 0) + 1
        return ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))

    def set_rdt_worker_init(self, payload: Optional[Dict[str, Any]]) -> None:
        """Cache the worker-init dict a restarted engine needs to rejoin RDT.

        Fetched once, from policy worker 0, after ``init_weight_sync_state``. It is
        static for the run: it describes the PRODUCERS (their actor names and the
        group-major weight metadata), and producers never restart. Without it a
        replacement can be relaunched and routed but cannot receive weights, so
        ``before_weight_sync`` refuses to promote it.
        """
        self._rdt_worker_init = payload
        if payload is None:
            logger.warning("[ft] no RDT worker-init payload cached; restarted engines cannot rejoin weight sync")

    # ---- the reconcile loop ----

    async def reconcile(self) -> None:
        """Bring slot state in line with reality, and act on the difference.

        In order, because each step depends on the previous one having happened:

        1. adopt what Part 1's reactive probe already concluded (it may have marked
           engines dead while we were inside a synchronous phase);
        2. probe slots we still believe in, so a death during training — invisible to
           the reactive path, which only sees data-plane failures — is found here;
        3. de-route and kill the newly dead — a refused connection, a self-reported
           503, and silence for the whole probe budget all arrive here as one verdict,
           so exactly one code path continues;
        4. start restarts within budget;
        5. harvest the restarts that have become healthy into ``PENDING_SYNC``.

        Never raises for a dead engine — that is what it is for. Floor enforcement is
        ``before_generation``'s job, so that a reconcile inside a sync cannot fail the
        sync.
        """
        if not self.enabled:
            return
        self._adopt_client_view()
        await self._probe_reachable()
        await self._retire_dead()
        self._begin_restarts()
        await self._harvest_restarts()

    def _adopt_client_view(self) -> None:
        """Adopt the fleet detector's verdict. A plain read, deliberately.

        ``RemoteInferenceClient._probe_health`` is the single place that decides life or
        death, and it now distinguishes a definitive failure from an ambiguous timeout and
        retries only the latter (see its docstring). So there is nothing to second-guess
        here: no parallel counter, no corroborating probe, no rehabilitation path. An
        earlier version of this method had all three, because the detector condemned on
        one short probe and this layer tried to repair the resulting bad verdicts --
        compensating downstream for weak evidence instead of strengthening it.
        """
        dead_urls = set(self._client.disabled_server_urls)
        for s_ in self._slots:
            if s_.state in (SlotState.LIVE, SlotState.PENDING_SYNC) and s_.url in dead_urls:
                logger.info(f"[ft] slot {s_.index} ({s_.url}) was found dead by the fleet probe")
                s_.state = SlotState.DEAD

    async def _probe_reachable(self) -> None:
        """Probe the slots we still believe in, covering the reactive path's blind spot.

        The reactive detector can only fire when a data-plane or control-plane request
        fails, so a death during the synchronous training phase produces no trigger. This
        runs at step boundaries and finds those.

        Uses the SAME ``_probe_health`` as everything else, which is why there is no
        threshold here: the retry-on-ambiguity lives inside the probe, so one negative
        answer from it is already "N misses with an escalating bound", not a single
        impatient timeout.
        """
        candidates = self._in_state(SlotState.LIVE, SlotState.PENDING_SYNC)
        if not candidates:
            return
        results = await asyncio.gather(
            *[self._client._probe_health(s_.url) for s_ in candidates], return_exceptions=True
        )
        for s_, ok in zip(candidates, results):
            if ok is True:
                continue
            logger.warning(f"[ft] slot {s_.index} ({s_.url}) failed the health probe; marking it dead")
            s_.state = SlotState.DEAD

    async def _retire_dead(self) -> None:
        """De-route dead slots, then kill their actors so the bundle is free.

        The kill is what makes a *wedged* engine restartable: it is still ALIVE to Ray
        and still holding its GPUs, so a replacement in the same bundle would contend
        with it. Killing first turns it into the connection-refused case Part 1
        already handles.
        """
        newly_dead = [s for s in self._in_state(SlotState.DEAD) if s._pending is None and s.url]
        if not newly_dead:
            return
        # Router first: stop sending it work before spending time on Ray calls.
        await asyncio.gather(*[self._client._router_remove_worker(s.url) for s in newly_dead], return_exceptions=True)
        for s in newly_dead:
            self._client.mark_dead(s.url)
            actor = self._actor_for(s)
            if actor is not None:
                try:
                    ray.kill(actor)
                    logger.info(f"[ft] killed the actor for slot {s.index} ({s.url})")
                except Exception as e:  # noqa: BLE001
                    # Already gone is the common case and is exactly what we wanted.
                    logger.info(f"[ft] ray.kill on slot {s.index} was a no-op: {type(e).__name__}: {e}")
            # The URL is retired with the actor: nothing may address this slot until a
            # replacement reports its own.
            s.url = ""
            # Do not relaunch into this bundle in the SAME pass. `ray.kill` is not
            # synchronous in its effect and the driver reclaims device memory only once
            # a process is fully gone, so a replacement that starts allocating inside
            # that window can OOM against memory the corpse still holds.
            #
            # Deliberately not a sleep: the next reconcile is a step boundary away,
            # which is a real event rather than a guessed duration, and one step is
            # nothing against reloading a model. Slots that were already dead on arrival
            # restart immediately -- this only defers ones we just killed.
            s.killed_this_pass = True

    def _bundle_pids(self, s: EngineSlot) -> List[int]:
        """Pids of every Ray actor scheduled into this slot's placement-group bundles.

        This is the precise answer to "what belongs to this replica", and it is the piece
        that was missing: vLLM's model-holding worker is a Ray actor in OUR placement
        group (its executor inherits it), so it is addressable by bundle even though it is
        unreachable by process parentage -- its OS parent is the raylet -- and unreachable
        by name.

        Empirically this is why restarts failed: the dead engine's worker kept ~62 of 79
        GiB, so the replacement could not fit its 55.4 GiB, and all three attempts died on
        free memory. Selecting by bundle addresses that process directly instead of hoping
        something transitively reclaims it.
        """
        try:
            import ray.util.state as ray_state

            from skyrl.backends.skyrl_train.inference_servers.engine_orphans import (
                bundle_resource_keys,
                select_bundle_actors,
            )

            pg = s.group._get_placement_group()
            keys = bundle_resource_keys(pg.id.hex(), s.group._get_bundle_indices_for_server(s.server_idx))
            rows = [
                (a.pid, a.state, a.class_name, a.required_resources or {})
                for a in ray_state.list_actors(detail=True, limit=1000)
            ]
            # The bundle IS the replica -- no class allowlist. See select_bundle_actors
            # for why adding one would trade a remote hazard for a silent regression the
            # next time vLLM renames a worker class.
            picked = select_bundle_actors(rows, keys)
            if picked:
                logger.info(f"[ft] slot {s.index} bundle holds actor(s): {picked}")
            return [pid for pid, _cls in picked]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[ft] slot {s.index}: could not resolve bundle membership: {e}")
            return []

    def _engine_gpu_bytes(self, s: EngineSlot) -> int:
        """Bytes one engine will ask for on a device: ``gpu_memory_utilization x total``.

        The total comes from the device itself on the node side; here we only know the
        ratio, so this returns the ratio scaled by a nominal 80 GiB when the real total is
        unknown. The node's own reading is authoritative for the verdict -- this value only
        has to be close enough to be a useful WAIT target.
        """
        util = float(getattr(s.group._cli_args, "gpu_memory_utilization", 0.0) or 0.0)
        if util <= 0:
            return 0
        return int(util * 80 * (2**30))

    def _reap_orphans_for(self, s: EngineSlot) -> None:
        """Leave this slot's bundle clean enough to relaunch into.

        Two passes, because they cover disjoint things: the BUNDLE pass kills every Ray
        actor Ray says is in this slot's bundles (plus their descendants, which is where
        the model lives under the mp executor), and the ORPHAN pass sweeps vLLM
        subprocesses whose owner died and which therefore belong to no bundle.

        Then it reports the resulting free memory, which is the only honest way to know
        whether the relaunch can succeed. Best-effort throughout: a failure here costs a
        restart attempt, not the run.
        """
        s.bundle_has_room = None
        try:
            gpu_ids = s.group._get_gpu_ids_for_server(s.server_idx) or []
            pids = self._bundle_pids(s)
            task = ray.remote(num_cpus=0)(_reap_on_node).options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=s.group._get_placement_group(),
                    placement_group_bundle_index=s.group._get_bundle_indices_for_server(s.server_idx)[0],
                )
            )
            need = self._engine_gpu_bytes(s)
            result = ray.get(task.remote(list(gpu_ids), pids, need), timeout=180)
            killed, free = result.get("killed", []), result.get("free", {})
            waited = result.get("waited_s", 0.0)
            if killed:
                logger.warning(
                    f"[ft] slot {s.index}: cleaned {len(killed)} process(es) {killed} out of "
                    f"GPU(s) {list(gpu_ids)} before relaunching"
                )
            if free:
                worst = min(f for f, _t in free.values())
                s.bundle_has_room = worst >= need if need else None
                logger.info(
                    f"[ft] slot {s.index}: after cleaning (+{waited:.0f}s for the device to "
                    f"settle), least-free target GPU has {worst / 2**30:.1f} GiB "
                    f"(engine needs {need / 2**30:.1f} GiB) -> "
                    f"{'ROOM' if s.bundle_has_room else 'STILL OCCUPIED'}"
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[ft] slot {s.index}: could not clean the bundle: {e}")

    def _actor_for(self, s: EngineSlot) -> Optional[Any]:
        actors = s.group.get_actors()
        if s.server_idx < len(actors):
            return actors[s.server_idx]
        return None

    def _begin_restarts(self) -> None:
        """Launch replacements for dead slots that still have budget.

        Non-blocking: a replacement can take minutes to load a model, and the fleet
        has to keep serving meanwhile. ``_harvest_restarts`` installs it later.
        """
        budget = max(0, int(self._ft.max_restarts_per_engine))
        for s in self._in_state(SlotState.DEAD):
            if s._pending is not None:
                continue
            if s.killed_this_pass:
                # Killed moments ago; let its GPU memory come back first (see
                # _retire_dead). Cleared so the NEXT reconcile launches it.
                s.killed_this_pass = False
                logger.info(f"[ft] slot {s.index} restarts at the next reconcile (its bundle is still draining)")
                continue
            if s.restarts >= budget:
                # Deliberately not an error: a slot that keeps dying is usually a bad
                # GPU, and the run can finish degraded. Logged at warning because it is
                # a permanent capacity loss, not a transient one -- but ONCE, on the
                # transition. Every later reconcile re-visits this slot, and repeating
                # the line each time buried the interesting events in the run log (the
                # first GPU demo emitted it four times for one dead slot).
                if not s.budget_exhausted_logged:
                    s.budget_exhausted_logged = True
                    logger.warning(
                        f"[ft] slot {s.index} has exhausted its restart budget "
                        f"({s.restarts}/{budget}); leaving it permanently dead"
                    )
                continue
            # Reap before EVERY attempt, not once when the slot died. vLLM's EngineCore
            # is a separate process, so killing (or losing) the server actor leaves it
            # holding the device -- and it may not have appeared in nvidia-smi's snapshot
            # at the moment we retired the slot, or a kill may have failed. Reaping here
            # makes it a precondition of the relaunch rather than a one-shot best effort,
            # which matters because an occupied GPU deliberately does not consume the
            # restart budget: without this the slot would retry forever against memory
            # nothing was ever going to release.
            self._reap_orphans_for(s)
            try:
                actor, start_ref = s.group.begin_restart(s.server_idx)
            except Exception as e:  # noqa: BLE001
                s.restarts += 1
                logger.warning(f"[ft] slot {s.index} restart could not be launched: {type(e).__name__}: {e}")
                continue
            s.restarts += 1
            s.state = SlotState.RESTARTING
            s._pending = (actor, start_ref, time.monotonic() + float(self._ft.restart_timeout_s))
            logger.info(f"[ft] restarting slot {s.index} (attempt {s.restarts}/{budget})")

    async def _harvest_restarts(self) -> None:
        """Install replacements that have become healthy; time out the rest.

        Polls rather than awaits: this runs at a step boundary, so it must not block
        the loop for the minutes a model load can take. A restart that is still
        loading simply stays ``RESTARTING`` and is harvested at a later boundary.
        """
        for s in self._in_state(SlotState.RESTARTING):
            assert s._pending is not None
            actor, start_ref, deadline = s._pending
            ready, _ = ray.wait([start_ref], timeout=0)
            if not ready:
                if time.monotonic() > deadline:
                    logger.warning(
                        f"[ft] slot {s.index} restart exceeded restart_timeout_s="
                        f"{self._ft.restart_timeout_s}s; giving up on this attempt"
                    )
                    self._abandon_restart(s, actor)
                continue
            try:
                info = ray.get(start_ref)
            except Exception as e:  # noqa: BLE001
                if s.bundle_has_room is False:
                    # This attempt was never going to succeed, so it must not count
                    # against the budget -- otherwise three retries are spent on the
                    # same wall and the slot is condemned for a condition that may
                    # still clear (an orphan we have not managed to reap yet).
                    s.restarts = max(0, s.restarts - 1)
                    logger.warning(
                        f"[ft] slot {s.index} cannot restart yet: its GPU was still occupied when we "
                        f"launched, so this attempt was doomed before it started and is not charged. "
                        f"Detail: {e}"
                    )
                else:
                    logger.warning(f"[ft] slot {s.index} restart failed: {type(e).__name__}: {e}")
                self._abandon_restart(s, actor)
                continue
            stamped = s.group.finish_restart(s.server_idx, actor, info)
            s.url = stamped.url
            s._pending = None
            # In the client's active set (so it joins the next sync) but NOT in the
            # router: its weights are the checkpoint's, and serving with those would
            # poison the batch under a weight-version-keyed prefix cache.
            self._client.replace_server_url(s.index, stamped.url)
            s.state = SlotState.PENDING_SYNC
            logger.info(
                f"[ft] slot {s.index} is up at {stamped.url}; awaiting a weight sync " "before it rejoins the router"
            )

    def _abandon_restart(self, s: EngineSlot, actor: Any) -> None:
        """Drop a failed or timed-out replacement and return the slot to DEAD."""
        try:
            ray.kill(actor)
        except Exception:  # noqa: BLE001
            pass
        s._pending = None
        s.state = SlotState.DEAD

    # ---- training-loop integration ----

    async def before_generation(self) -> None:
        """Reconcile the fleet before committing a batch to it.

        Covers what the reactive detector structurally cannot see: it only fires when a
        request fails, so an engine that dies during the synchronous training phase
        produces no trigger. Reconciling here finds those and starts the restart a step
        earlier than the next failed request would.

        Deliberately does NOT enforce ``min_live_engines`` or wait for a recovering slot.
        The floor is already enforced where the evidence is -- ``_probe_and_disable``
        raises ``EngineFleetError`` the moment a probe takes the fleet below it -- and a
        second check here was a duplicate with its own waiting loop, which is complexity
        for a case the probe already handles. No-op unless engine restarts are enabled.
        """
        if not self.enabled:
            return
        await self.reconcile()

    async def before_weight_sync(self) -> List[EngineSlot]:
        """Reconcile and RDT-re-init anything that restarted, so this sync includes it.

        Returns the slots re-initialized here, which ``after_weight_sync`` re-admits.
        A slot that fails to re-init is put back to DEAD rather than allowed into the
        sync: a consumer that has not baked cannot pull, and letting it into the sync
        would stall every producer bound to it.

        Deliberately does NOT enforce ``min_live_engines`` — a floor violation raised
        here would fail the sync rather than the generation phase, converting a
        recoverable capacity problem into a dead run.
        """
        if not self.enabled:
            return []
        await self.reconcile()
        pending = self._in_state(SlotState.PENDING_SYNC)
        if not pending:
            return []
        if self._rdt_worker_init is None:
            logger.error(
                f"[ft] {len(pending)} engine(s) are waiting to rejoin but no RDT worker-init "
                "payload was cached; they cannot receive weights and stay out of the router"
            )
            return []

        admitted: List[EngineSlot] = []
        for s in pending:
            try:
                # One POST, to exactly this engine, stamped with exactly its slot --
                # which is what makes the replacement bake a plan for the consumer ids
                # the dead engine owned rather than deployment 0's.
                await self._client.init_weight_transfer_engine_rdt(self._rdt_worker_init, targets=[(s.url, s.index)])
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"[ft] slot {s.index} ({s.url}) failed RDT re-init: {type(e).__name__}: {e}. "
                    "Marking it dead again rather than letting it stall the sync."
                )
                s.state = SlotState.DEAD
                continue
            logger.info(f"[ft] slot {s.index} re-initialized for RDT; it joins this weight sync")
            admitted.append(s)
        return admitted

    async def after_weight_sync(self, admitted: Optional[Sequence[EngineSlot]] = None) -> None:
        """Re-admit engines whose weights are now current.

        This is the only place a restarted engine enters the router, and it runs after
        the sync succeeded — so no rollout can ever reach an engine carrying
        checkpoint weights.
        """
        if not self.enabled:
            return
        slots = list(admitted) if admitted is not None else self._in_state(SlotState.PENDING_SYNC)
        for s in slots:
            if s.state is not SlotState.PENDING_SYNC:
                continue  # died during the sync; leave it to the next reconcile
            if await self._client._router_add_worker(s.url):
                s.state = SlotState.LIVE
                logger.info(f"[ft] slot {s.index} ({s.url}) rejoined the fleet; {self.describe()}")
            else:
                # Reachable and current, just not routed. Retried next boundary rather
                # than failed: it costs one step of capacity, not the run.
                logger.warning(f"[ft] slot {s.index} ({s.url}) is current but the router refused it; will retry")
