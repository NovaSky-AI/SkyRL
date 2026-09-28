"""`CommitCoordinator`: bounded background persistence, ordered per trajectory.

Every journal append that is not part of a lifecycle call goes through here.
It exists to hold two promises at once:

* **Order.** Two turns of one trajectory must reach its journal in the order
  they were committed, because the second one's graph delta references nodes
  the first one created. Each trajectory's commits are chained through its
  `pending_commit` future, so a slow append delays the next one rather than
  letting it overtake.
* **A bound.** Capture must not consume unbounded memory when the disk is
  slow. The number of appends in flight is capped, and the two capture modes
  meet that cap differently, exactly as the product contract says they should:
  text refuses the work and records a gap, and TITO waits for capacity before
  it closes the response.

A failed append is not retried in a loop. Text marks a gap, keeps serving, and
the gap is written as its own record on the next append that succeeds -- so a
disk that comes back records what was lost while it was away. TITO awaits its
commit, so a failure there surfaces as a response that cannot close cleanly,
which is the one thing that stops a later turn from retokenizing missing model
output.

**What is not allowed to be lost is the graph.** An exchange's journal record
carries only the nodes that exchange *introduced*, so losing one leaves every
later record referencing ancestors the journal does not contain -- a dangling
parent that a recovery or a compile turns into an orphaned subtree, and a
truncated path in every export off it. The exchange row can go; its nodes
cannot. So the nodes of a refused or failed commit are held and attached to
the next record for that trajectory that carries a graph, which keeps the
journal internally consistent while `calls_missing` says what was lost.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from skyrl_capture.domain.graph import GraphNode
from skyrl_capture.domain.records import ActiveTrajectory, now
from skyrl_capture.persistence.active import ActiveStore
from skyrl_capture.persistence.journal import ActiveRecord, CaptureGap, ExchangeCommitted

logger = logging.getLogger(__name__)


class CommitCoordinator:
    def __init__(
        self,
        active: ActiveStore,
        *,
        capacity: int = 1024,
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        self._active = active
        self._capacity = capacity
        # Told to a reader in this process, if there is one, so a replica that
        # also serves the viewer does not show a run a sweep behind itself.
        self._on_change = on_change or (lambda trajectory_id: None)
        self._pending = 0
        self._started_at: dict[int, float] = {}
        self._ticket = 0
        self._capacity_free = asyncio.Event()
        self._capacity_free.set()
        # Gaps observed but not yet written, per trajectory. Kept so a disk
        # that recovers records what it missed rather than silently resuming.
        self._unwritten_gaps: dict[str, list[CaptureGap]] = {}
        # Graph nodes from exchanges the journal never took. Held until a later
        # record for that trajectory can carry them, because every record after
        # a lost one references its nodes.
        self._undelivered_nodes: dict[str, list[GraphNode]] = {}
        self.pending_high_water = 0
        self.commits = 0
        self.refused = 0
        self.failures = 0
        self.last_error: str | None = None

    # -- capacity ---------------------------------------------------------------
    @property
    def pending(self) -> int:
        return self._pending

    def has_capacity(self) -> bool:
        return self._pending < self._capacity

    async def reserve(self, *, timeout: float | None = None) -> bool:
        """Wait for room to commit. What TITO does before it closes a response.

        Returns False on timeout, which a caller treats as a persistence
        failure -- there is no third answer, because a TITO response that
        cannot be captured must not close cleanly.
        """
        if self.has_capacity():
            return True
        try:
            await asyncio.wait_for(self._wait_for_capacity(), timeout)
        except TimeoutError:
            return False
        return True

    async def _wait_for_capacity(self) -> None:
        while not self.has_capacity():
            self._capacity_free.clear()
            await self._capacity_free.wait()

    # -- submitting -------------------------------------------------------------
    def submit(
        self, active: ActiveTrajectory, record: ActiveRecord
    ) -> asyncio.Task[None] | None:
        """Queue one append behind this trajectory's previous one.

        Returns the task to await, or `None` when the bound is reached. A
        `None` is a refusal, and the caller is the one that decides what a
        refusal means for the response it is serving.
        """
        if not self.has_capacity():
            self.refused += 1
            # The exchange is refused, but its nodes are what every later
            # record on this trajectory will reference.
            self._hold_nodes(active.id, record)
            return None
        self._ticket += 1
        ticket = self._ticket
        self._pending += 1
        self.pending_high_water = max(self.pending_high_water, self._pending)
        self._started_at[ticket] = time.monotonic()
        previous = active.pending_commit
        task = asyncio.create_task(
            self._commit(active, record, previous, ticket),
            name=f"commit-{active.id}-{ticket}",
        )
        active.pending_commit = task
        return task

    async def submit_when_ready(
        self, active: ActiveTrajectory, record: ActiveRecord, *, timeout: float
    ) -> asyncio.Task[None] | None:
        """Wait for capacity, then queue. What TITO calls, because it cannot
        proceed without the commit and must not refuse the work.

        Returns `None` only when capacity never came, which the caller treats
        as a persistence failure -- there is no version of TITO where a
        response closes cleanly and its exchange was never written.
        """
        deadline = time.monotonic() + timeout
        while True:
            task = self.submit(active, record)
            if task is not None:
                return task
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not await self.reserve(timeout=remaining):
                return None

    async def _commit(
        self,
        active: ActiveTrajectory,
        record: ActiveRecord,
        previous: Any,
        ticket: int,
    ) -> None:
        try:
            if previous is not None:
                # Order, not success: a failed predecessor must not stop this
                # record, and its failure is already recorded as a gap.
                with contextlib.suppress(Exception):
                    await previous
            await self._flush_gaps(active)
            carried = self._carry_lost_nodes(active.id, record)
            await self._active.append(active.id, carried)
            if carried is not record:
                # They are on disk now. A record that carried nothing -- an
                # ungraphed endpoint -- leaves them waiting for one that can.
                self._undelivered_nodes.pop(active.id, None)
            self.commits += 1
            self._on_change(active.id)
            # The bytes are on disk, so the copy in memory is redundant. This
            # is what keeps a long trajectory bounded by its graph rather than
            # by everything it ever sent.
            active.shed_payloads()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.failures += 1
            self.last_error = f"{type(error).__name__}: {error}"
            logger.warning("commit failed for trajectory %s: %s", active.id, error)
            self.record_gap(active, reason=self.last_error)
            self._hold_nodes(active.id, record)
            raise
        finally:
            self._release(ticket)
            if active.pending_commit is asyncio.current_task():
                active.pending_commit = None

    def _release(self, ticket: int) -> None:
        self._pending -= 1
        self._started_at.pop(ticket, None)
        if self.has_capacity():
            self._capacity_free.set()

    # -- the graph a lost record was carrying -------------------------------------
    def _hold_nodes(self, trajectory_id: str, record: ActiveRecord) -> None:
        """Keep the nodes of a record the journal did not take."""
        if not isinstance(record, ExchangeCommitted) or record.graph is None:
            return
        held = self._undelivered_nodes.setdefault(trajectory_id, [])
        known = {node.id for node in held}
        held.extend(node for node in record.graph.nodes if node.id not in known)

    def _carry_lost_nodes(self, trajectory_id: str, record: ActiveRecord) -> ActiveRecord:
        """Attach the nodes of earlier lost records to this one.

        In graph order and ahead of this record's own nodes, because a node is
        inserted under a parent that has to be there already. An exchange with
        no graph carries nothing and references nothing, so it is left alone
        and the nodes wait for one that does.
        """
        held = self._undelivered_nodes.get(trajectory_id)
        if not held or not isinstance(record, ExchangeCommitted) or record.graph is None:
            return record
        mine = {node.id for node in record.graph.nodes}
        carried = [node for node in held if node.id not in mine]
        if not carried:
            return record
        return replace(
            record,
            graph=replace(record.graph, nodes=(*carried, *record.graph.nodes)),
        )

    # -- gaps ---------------------------------------------------------------------
    def record_gap(
        self, active: ActiveTrajectory, *, count: int = 1, reason: str | None = None
    ) -> None:
        """Mark capture as incomplete, in memory now and on disk when it can be.

        The in-memory mark is what the final record carries, so a trajectory
        whose journal never takes the gap record still finishes as incomplete.
        """
        at = now()
        active.record_gap(count, reason, at)
        self._unwritten_gaps.setdefault(active.id, []).append(CaptureGap(count, reason, at))

    async def _flush_gaps(self, active: ActiveTrajectory) -> None:
        gaps = self._unwritten_gaps.pop(active.id, None)
        if not gaps:
            return
        try:
            for gap in gaps:
                await self._active.append(active.id, gap)
        except Exception:
            # Still unwritten. Put them back rather than losing the count.
            self._unwritten_gaps.setdefault(active.id, []).extend(gaps)
            raise

    async def flush_gaps(self, active: ActiveTrajectory) -> None:
        """Best-effort: write any gaps the disk refused earlier. Used by finish."""
        with contextlib.suppress(Exception):
            await self._flush_gaps(active)

    # -- waiting --------------------------------------------------------------------
    async def settle(self, active: ActiveTrajectory, *, timeout: float = 30.0) -> bool:
        """Wait for this trajectory's queued commits. False on timeout.

        Waiting on the tail of the chain is waiting on all of it: every commit
        awaits its predecessor, so the last one cannot finish first.
        """
        pending = active.pending_commit
        if pending is None:
            return True
        try:
            await asyncio.wait_for(asyncio.shield(pending), timeout)
        except TimeoutError:
            return False
        except Exception:
            # A failed commit is a recorded gap, not a reason to hold finish.
            return True
        return True

    async def stop(self, *, timeout: float = 30.0) -> None:
        """Let queued commits drain. Called on graceful shutdown."""
        deadline = time.monotonic() + timeout
        while self._pending and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        if self._pending:
            logger.warning("shutting down with %d commit(s) still pending", self._pending)

    # -- what health reports ------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        oldest = min(self._started_at.values(), default=None)
        return {
            "pending_commits": self._pending,
            "pending_high_water": self.pending_high_water,
            "capacity": self._capacity,
            "oldest_pending_age_s": round(time.monotonic() - oldest, 3) if oldest else 0.0,
            "commits": self.commits,
            "refused": self.refused,
            "failures": self.failures,
            "unwritten_gaps": sum(len(gaps) for gaps in self._unwritten_gaps.values()),
            "undelivered_nodes": sum(len(nodes) for nodes in self._undelivered_nodes.values()),
            "last_error": self.last_error,
        }
