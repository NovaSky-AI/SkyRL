"""Finishing the trajectories whose caller asked and then went away.

The case is ordinary and was found by running 256 agentic rollouts: an agent
times out, abandons a generation the engine is still running, and its `finish`
is refused with a retryable `503` because a record missing that turn would be
worse than no record. The retry never comes -- an RL generator retries on a
*fresh* trajectory, because the graph is append-only -- so nobody is left who
knows the old id.
"""

from __future__ import annotations

import pytest
from test_aggregate import exchange, header
from test_persistence import NOW

from skyrl_capture.domain.records import metadata_update
from skyrl_capture.persistence import journal
from skyrl_capture.persistence.active import DiskActiveStore
from skyrl_capture.persistence.layout import ensure_record
from skyrl_capture.writer.sweep import FinishSweeper


class Commands:
    """Records what the sweep asked to finish."""

    def __init__(self, fail: set[str] | None = None) -> None:
        self.finished: list[str] = []
        self.fail = fail or set()

    async def finish(self, trajectory_id, **_named):
        if trajectory_id in self.fail:
            raise RuntimeError("still settling")
        self.finished.append(trajectory_id)
        return {"id": trajectory_id}


class Registry:
    def __init__(self, hot: dict | None = None) -> None:
        self._hot = hot or {}

    def hot(self, trajectory_id: str):
        return self._hot.get(trajectory_id)


class Hot:
    def __init__(self, in_flight: int) -> None:
        self.in_flight = in_flight


async def journal_for(root, identifier, *, requested: bool, exchanges: int = 1):
    active = DiskActiveStore(root)
    await active.create(header(identifier, project="p", run_id="r"))
    for index in range(exchanges):
        await active.append(
            identifier, journal.ExchangeCommitted(*exchange(f"ex_{index}", sequence=index), NOW)
        )
    if requested:
        await active.append(
            identifier,
            journal.FinishRequested(metadata_update(), "success", "h", NOW),
        )
    await active.close()
    return DiskActiveStore(root)


@pytest.fixture
def root(tmp_path):
    return ensure_record(tmp_path / "record")


async def test_a_trajectory_whose_caller_asked_and_left_is_finished(root):
    """The whole point. Its caller got a 503 and moved on to a fresh id."""
    active = await journal_for(root, "tr_asked", requested=True)
    commands = Commands()
    sweeper = FinishSweeper(commands=commands, active=active, registry=Registry())

    assert await sweeper.sweep() == 1
    assert commands.finished == ["tr_asked"]
    assert sweeper.swept == 1


async def test_a_trajectory_nobody_asked_about_is_left_alone(root):
    """There is deliberately no trajectory TTL. A caller that has not come
    back yet is not the same as one that asked and went away, and expiring the
    first would throw away a run in progress."""
    active = await journal_for(root, "tr_open", requested=False)
    commands = Commands()
    sweeper = FinishSweeper(commands=commands, active=active, registry=Registry())

    assert await sweeper.sweep() == 0
    assert commands.finished == []


async def test_a_turn_still_in_flight_is_skipped_not_forced(root):
    """The refusal the sweep exists behind is a real one: while a generation
    is running, the record would be missing the turn that is still happening.
    Waiting for the next pass costs nothing; compiling now would be wrong."""
    active = await journal_for(root, "tr_busy", requested=True)
    commands = Commands()
    sweeper = FinishSweeper(
        commands=commands, active=active, registry=Registry({"tr_busy": Hot(in_flight=1)})
    )

    assert await sweeper.sweep() == 0
    assert commands.finished == []
    assert sweeper.skipped_in_flight == 1

    # And once it drains, the next pass takes it.
    sweeper._registry = Registry({"tr_busy": Hot(in_flight=0)})  # noqa: SLF001
    assert await sweeper.sweep() == 1
    assert commands.finished == ["tr_busy"]


async def test_one_that_cannot_be_finished_does_not_stop_the_others(root):
    """A pass is a pass. A trajectory whose caller came back between the scan
    and the call raises, and the ones after it still get swept."""
    active = await journal_for(root, "tr_a", requested=True)
    await journal_for(root, "tr_b", requested=True)
    commands = Commands(fail={"tr_a"})
    sweeper = FinishSweeper(commands=commands, active=active, registry=Registry())

    assert await sweeper.sweep() == 1
    assert commands.finished == ["tr_b"]
    assert sweeper.failures == 1


async def test_a_finished_journal_is_read_from_disk_not_from_memory(root):
    """The case the sweep most needs to cover is a process that died holding
    the trajectory, so nothing about it is in this process's memory."""
    active = await journal_for(root, "tr_dead", requested=True)
    commands = Commands()
    # An empty registry: nothing is hot here, as after a restart.
    sweeper = FinishSweeper(commands=commands, active=active, registry=Registry())

    assert await sweeper.sweep() == 1
    assert commands.finished == ["tr_dead"]
