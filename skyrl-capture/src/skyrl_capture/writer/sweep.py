"""Finish the trajectories whose caller asked and then went away.

`finish` is a barrier. It refuses to compile a record while a turn is still in
flight, because a record missing a turn that happened is worse than no record,
and answers `503` so the caller can retry. In tokens mode a healthy caller
never reaches that barrier -- an exchange is durable before its response is
allowed to close, so a harness holding all of its replies has nothing
outstanding.

A caller that *abandons* a call is the exception, and it is ordinary: an agent
that times out stops waiting for a generation the engine is still running. Its
finish is refused, correctly. But the retry the `503` invites never comes,
because the caller has moved on -- an RL generator retries on a *fresh*
trajectory, since the graph is append-only and reusing the id would interleave
two rollouts into one record. Nobody is left who knows the old id.

So the journal keeps its `FinishRequested`, the turn drains a moment later,
and the trajectory sits finishable for ever. That is what this sweep is for,
and why it is a sweep rather than a timeout: it completes work that was
already asked for, rather than expiring work that was not. The distinction
matters -- there is deliberately no trajectory TTL anywhere in this package.

What it will not do is decide that a turn is lost. A journal with a turn still
in flight is skipped and looked at again next pass; only the trajectory that
could have been finished by its caller, had the caller waited, is finished
here. `finish` is idempotent by request hash, so re-requesting one that landed
between the scan and the call is harmless.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from skyrl_capture.persistence.journal import FinishRequested, TrajectoryCreated

logger = logging.getLogger(__name__)


class FinishSweeper:
    """Complete finishes nobody is coming back for."""

    def __init__(
        self,
        *,
        commands: Any,
        active: Any,
        registry: Any,
        interval: float = 60.0,
    ) -> None:
        self._commands = commands
        self._active = active
        self._registry = registry
        self._interval = interval
        self.swept = 0
        self.skipped_in_flight = 0
        self.failures = 0

    async def run_forever(self) -> None:
        # A pass before the first sleep would race the journals a replacement
        # process is still adopting, and there is no hurry: nothing here is
        # more urgent than the interval it was configured with.
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.sweep()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("finish sweep failed; trying again next interval")

    async def sweep(self) -> int:
        """One pass. Returns how many trajectories it finished."""
        finished = 0
        for trajectory_id in await asyncio.to_thread(self._active.journal_ids):
            requested = await self._recorded_finish(trajectory_id)
            if requested is None:
                continue
            hot = self._registry.hot(trajectory_id)
            if hot is not None and hot.in_flight:
                # Still generating. Its caller may yet come back, and if not
                # the next pass will find it settled.
                self.skipped_in_flight += 1
                continue
            try:
                # The recorded request, replayed -- not a new one. A finish
                # is identified by a hash over its labels, annotations and
                # command result, and a second finish naming a *different*
                # outcome is a conflict by design. The sweep is not a second
                # caller: it is completing the outcome already in the journal,
                # so it has to ask for that one.
                await self._commands.finish(
                    trajectory_id,
                    labels=list(requested.update.labels) if requested.update else None,
                    annotations=dict(requested.update.annotations) if requested.update else None,
                    command_result=requested.command_result,
                    export_format="graph",
                )
            except Exception as error:
                # Including a finish that raced a caller who did come back.
                self.failures += 1
                logger.info("sweep could not finish %s: %s", trajectory_id, error)
                continue
            finished += 1
            self.swept += 1
            logger.info("sweep finished %s, whose caller did not return", trajectory_id)
        return finished

    async def _recorded_finish(self, trajectory_id: str) -> FinishRequested | None:
        """The finish this journal holds, if it holds one and never completed.

        Read from the journal rather than from memory: the trajectory may
        belong to a process that has since died, which is the case the sweep
        most needs to cover -- and the recorded request is what the replay has
        to name, so reading it is not optional.
        """
        try:
            records = await self._active.records(trajectory_id)
        except Exception:
            return None
        if not records or not isinstance(records[0], TrajectoryCreated):
            return None
        for record in reversed(records):
            if isinstance(record, FinishRequested):
                return record
        return None

    def stats(self) -> dict[str, Any]:
        return {
            "finish_sweeps": self.swept,
            "finish_sweep_skipped_in_flight": self.skipped_in_flight,
            "finish_sweep_failures": self.failures,
        }
