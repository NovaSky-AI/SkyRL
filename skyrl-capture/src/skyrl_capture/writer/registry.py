"""`TrajectoryRegistry`: the trajectories this process is hot for.

It holds aggregates for trajectories in local use and nothing else. There is
no startup replay: a process that restarts does not read a directory of a
hundred thousand journals to find the four it is about to serve. It finds each
one when a request for it arrives, and forgets it when it finishes.

Resolution is four cases, in this order:

1. the hot aggregate, if this process has it;
2. otherwise, if a committed record exists, the trajectory is finished and
   cannot take inference -- `410`, not `404`. This is checked *before* the
   journal, not after: both files exist for the moment between the commit and
   the journal's deletion, and for good if a process died in that window, and
   adopting the journal then would reopen a trajectory whose record is already
   written. Readers prefer the committed form; so does this;
3. otherwise the active journal, recovered lazily and adopted;
4. otherwise it does not exist -- `404`.

The registry assumes what the deployment guarantees: consistent-hash routing
sends create, inference, finish and metadata for one trajectory to one process.
Nothing here detects a second writer, and nothing here could -- two processes
appending to one journal would interleave records that both look valid. That
invariant lives in the routing layer, and `docs/operations.md` says so.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import Any

from skyrl_capture.domain.records import (
    ActiveTrajectory,
    TrajectoryConflict,
    TrajectoryHeader,
)
from skyrl_capture.persistence.active import ActiveStore
from skyrl_capture.persistence.committed import CommittedStore

logger = logging.getLogger(__name__)


class TrajectoryRegistry:
    def __init__(
        self,
        *,
        active: ActiveStore,
        committed: CommittedStore,
        on_evict: Callable[[str], None] | None = None,
    ) -> None:
        self._active = active
        self._committed = committed
        # What the selected proxy wants released when a trajectory leaves hot
        # memory: token capture drops its trace, text capture holds nothing.
        self._on_evict = on_evict or (lambda trajectory_id: None)
        self._hot: dict[str, ActiveTrajectory] = {}
        # Held for the duration of a lazy recovery, so two requests that arrive
        # together for a cold trajectory do not both read and adopt the journal.
        self._recovering: dict[str, asyncio.Lock] = {}
        self.recovered = 0
        self.evicted = 0

    def on_evict(self, callback: Callable[[str], None]) -> None:
        """What the selected proxy wants released when a trajectory goes cold.

        Set after construction because the thing that holds it -- token
        capture's session manager -- reads through this registry, and one of
        the two has to exist first.
        """
        self._on_evict = callback

    @property
    def active_store(self) -> ActiveStore:
        """The journals. Lifecycle commands append to them directly, because
        a create, a finish or a metadata edit is not background work: the
        caller is waiting for it to be durable."""
        return self._active

    # -- reads ----------------------------------------------------------------
    def hot(self, trajectory_id: str) -> ActiveTrajectory | None:
        """The aggregate, if it is already in memory. No I/O."""
        return self._hot.get(trajectory_id)

    def hot_ids(self) -> list[str]:
        return list(self._hot)

    def count(self) -> int:
        return len(self._hot)

    async def resolve(self, trajectory_id: str) -> ActiveTrajectory | None:
        """The aggregate for ``trajectory_id``, recovering it if it is cold."""
        found = self._hot.get(trajectory_id)
        if found is not None:
            return found
        # One lock per trajectory this process has looked for, kept until the
        # trajectory is evicted: a trajectory looked for once usually serves
        # many turns, and the lock is what keeps two of them from both
        # adopting the journal.
        lock = self._recovering.get(trajectory_id)
        if lock is None:
            lock = self._recovering[trajectory_id] = asyncio.Lock()
        async with lock:
            found = self._hot.get(trajectory_id)
            if found is not None:
                return found
            # A committed record ends the trajectory, whatever is still on
            # disk beside it. Both files exist for the moment between the
            # commit and the journal's deletion -- and for good, if a process
            # died in that window -- and adopting the journal then would reopen
            # a trajectory whose record is already written and cannot take
            # another exchange.
            if await self._committed.exists(trajectory_id):
                await self._drop_redundant_journal(trajectory_id)
                return None
            recovered = await self._active.recover(trajectory_id)
            if recovered is None:
                # Nothing to hold a lock for. Kept out of the map because an
                # unknown id is exactly what a misrouted or hostile flood of
                # requests carries, and a lock per id would be the leak.
                self._recovering.pop(trajectory_id, None)
                return None
            self._hot[trajectory_id] = recovered
            self.recovered += 1
            logger.info(
                "recovered trajectory %s from its journal: %d exchange(s), %d node(s)",
                trajectory_id,
                len(recovered.exchanges),
                len(recovered.graph),
            )
            return recovered

    async def is_committed(self, trajectory_id: str) -> bool:
        """Whether a finished record exists. What tells `410` from `404`."""
        return await self._committed.exists(trajectory_id)

    async def _drop_redundant_journal(self, trajectory_id: str) -> None:
        """Remove a journal whose record is committed.

        The cleanup the design leaves to whoever notices: a process that died
        between the commit and the delete leaves both, and the journal is the
        redundant one. Best effort -- readers prefer the committed record
        either way, so failing to remove it costs a file and nothing else.

        The relist is not best effort in the same way. That dead process may
        have died between writing the record and listing it, and `runs/` is
        authoritative: deleting the journal now would leave a record no
        listing can reach. So the header is rewritten first, and the journal
        is kept if that fails.
        """
        try:
            await self._committed.relist(trajectory_id)
        except Exception:
            logger.warning("keeping the journal for %s: it is not listed", trajectory_id)
            return
        with contextlib.suppress(Exception):
            await self._active.remove(trajectory_id)

    # -- writes ----------------------------------------------------------------
    async def create(self, header: TrajectoryHeader) -> ActiveTrajectory:
        """Persist creation and adopt the trajectory.

        Idempotent by the create-request hash, which is persisted rather than
        kept in a process-local table: a retry after a restart has to reach the
        same answer as one before it. Raises `TrajectoryConflict` when the id
        is already in use for a different body, or already finished.
        """
        existing = await self.resolve(header.id)
        if existing is not None:
            if existing.header.create_request_hash != header.create_request_hash:
                raise TrajectoryConflict(
                    f"trajectory {header.id!r} already exists with a different creation "
                    "body; a trajectory id names one trial"
                )
            return existing
        if await self._committed.exists(header.id):
            raise TrajectoryConflict(
                f"trajectory {header.id!r} has already finished; its record is committed "
                "and it cannot be recreated"
            )
        await self._active.create(header)
        active = ActiveTrajectory.create(header)
        self._hot[header.id] = active
        return active

    def evict(self, trajectory_id: str) -> None:
        """Drop the aggregate and whatever the proxy held for it.

        Called after finish has committed. A finished trajectory is read from
        its committed record, so keeping it here would only be a second copy
        that has to be kept in step with the file.
        """
        if self._hot.pop(trajectory_id, None) is not None:
            self.evicted += 1
        self._recovering.pop(trajectory_id, None)
        self._on_evict(trajectory_id)

    # -- what health reports ----------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "hot_trajectories": len(self._hot),
            "recovered": self.recovered,
            "evicted": self.evicted,
        }
