"""The boundaries that only misbehave when something else goes wrong.

Everything here is a case a passing test suite does not reach on its own: a
commit that fails while a trajectory keeps being used, a `finish` that runs out
of grace with a turn still open, two lifecycle calls interleaving on one
trajectory, a committed record rewritten under a reader that already saw it,
and an export created while the index is still being built.

They are grouped together because they share a shape. In each, the wrong
behaviour is not a crash but a quiet, plausible answer: a record that says it
is complete, a graph whose ancestors are missing, an annotation that lands and
then disappears, a dataset that omits what the scan had not reached. None of
them would fail a test that only asks whether the happy path works.
"""

from __future__ import annotations

import asyncio

import pytest
from test_aggregate import exchange, header
from test_persistence import NOW

from skyrl_capture.domain.records import metadata_update
from skyrl_capture.persistence import journal
from skyrl_capture.persistence.active import DiskActiveStore, read_journal, rebuild
from skyrl_capture.persistence.committed import DiskCommittedStore
from skyrl_capture.persistence.layout import ensure_record
from skyrl_capture.reader.records import RecordReader, TrajectoryQuery
from skyrl_capture.writer.compile import compile_record


class _Slow:
    """A journal whose appends take long enough to queue behind each other."""

    def __init__(self, real, *, delay: float) -> None:
        self.real = real
        self.delay = delay

    async def create(self, header):
        return await self.real.create(header)

    async def append(self, trajectory_id, record):
        await asyncio.sleep(self.delay)
        return await self.real.append(trajectory_id, record)

    async def recover(self, trajectory_id):
        return await self.real.recover(trajectory_id)

    async def remove(self, trajectory_id):
        return await self.real.remove(trajectory_id)


class _FailOnce:
    """A journal that refuses exactly one append, then works."""

    def __init__(self, real, *, failures: int = 1) -> None:
        self.real = real
        self.remaining = failures
        self.refused = 0

    async def create(self, header):
        return await self.real.create(header)

    async def append(self, trajectory_id, record):
        if self.remaining and isinstance(record, journal.ExchangeCommitted):
            self.remaining -= 1
            self.refused += 1
            raise OSError("the volume blinked")
        return await self.real.append(trajectory_id, record)

    async def recover(self, trajectory_id):
        return await self.real.recover(trajectory_id)

    async def remove(self, trajectory_id):
        return await self.real.remove(trajectory_id)


# -- the graph a lost record was carrying ----------------------------------------
async def test_a_lost_exchange_does_not_leave_the_journal_referencing_its_nodes(stack):
    """The failure this is really about is silent.

    An exchange's record carries only the nodes it introduced, so a lost one
    leaves every later record pointing at ancestors the journal does not
    contain. Nothing errors: the live graph is whole, the response was served,
    and the damage only appears when somebody recovers or exports -- as a path
    that starts in the middle.
    """
    created = await stack.create_trajectory()
    commits = stack.runtime.commits
    failing = _FailOnce(stack.runtime.active)
    real = commits._active  # noqa: SLF001
    commits._active = failing  # noqa: SLF001

    history = [{"role": "user", "content": "first"}]
    reply = await stack.chat(created, history)
    history.append(reply.json()["choices"][0]["message"])
    await asyncio.sleep(0.05)
    commits._active = real  # noqa: SLF001

    # The client got that reply, so its next request continues from it -- which
    # is exactly what makes the second record reference the lost nodes.
    history.append({"role": "user", "content": "second"})
    await stack.chat(created, history)
    await stack.settle()

    assert failing.refused == 1
    live = stack.aggregate(created["id"])
    replayed = stack.journal(created["id"])
    assert replayed is not None

    # One exchange was lost and said so; the graph came through whole.
    assert len(replayed.exchanges) == 1
    assert live.integrity.calls_missing == 1
    assert {node.id for node in replayed.graph.ordered()} == {
        node.id for node in live.graph.ordered()
    }
    # And every node in it has the parent it claims, which is the thing a
    # dangling reference breaks.
    for node in replayed.graph.ordered():
        assert node.parent_id is None or node.parent_id in replayed.graph.nodes


async def test_a_refused_commit_keeps_its_nodes_too(stack_builder):
    """The bound being reached loses the same thing a failure does."""
    from skyrl_capture.config import RecordConfig

    stack = await stack_builder(record=RecordConfig(commit_capacity=1, fsync="never"))
    created = await stack.create_trajectory()
    commits = stack.runtime.commits
    # A disk slow enough that the next turn arrives while the last append is
    # still running, which is the only way the bound is actually reached.
    slow = _Slow(stack.runtime.active, delay=0.25)
    real = commits._active  # noqa: SLF001
    commits._active = slow  # noqa: SLF001

    history: list[dict[str, str]] = []
    for turn in range(5):
        history.append({"role": "user", "content": f"turn {turn}"})
        reply = await stack.chat(created, history)
        history.append(reply.json()["choices"][0]["message"])
    assert commits.refused >= 1, "the bound was actually reached"

    commits._active = real  # noqa: SLF001
    await stack.settle()
    # One more turn, so the nodes the refusals held have a record to ride on.
    history.append({"role": "user", "content": "after the crunch"})
    await stack.chat(created, history)
    await stack.settle()
    replayed = stack.journal(created["id"])
    for node in replayed.graph.ordered():
        assert node.parent_id is None or node.parent_id in replayed.graph.nodes
    assert {node.id for node in replayed.graph.ordered()} == {
        node.id for node in stack.aggregate(created["id"]).graph.ordered()
    }


async def test_the_committed_record_of_a_lossy_run_still_has_whole_paths(stack):
    """What the graph being whole is *for*: an export row that starts at the
    root rather than in the middle of a conversation."""
    created = await stack.create_trajectory()
    commits = stack.runtime.commits
    failing = _FailOnce(stack.runtime.active)
    real = commits._active  # noqa: SLF001
    commits._active = failing  # noqa: SLF001

    history = [{"role": "user", "content": "first"}]
    reply = await stack.chat(created, history)
    history.append(reply.json()["choices"][0]["message"])
    await asyncio.sleep(0.05)
    commits._active = real  # noqa: SLF001
    history.append({"role": "user", "content": "second"})
    await stack.chat(created, history)
    await stack.finish(created["id"])

    record = stack.record(created["id"])
    assert record.trajectory.integrity.complete is False
    assert record.trajectory.integrity.calls_missing == 1

    rows = await stack.export_lines(trajectory=created["id"], format="text-samples")
    assert rows, "a lossy trajectory still exports"
    # The path reaches back to the first message rather than starting at the
    # exchange that survived.
    assert rows[0]["messages"][0]["message"]["content"] == "first"


# -- finish, out of grace --------------------------------------------------------
async def test_finish_out_of_grace_does_not_finalize(stack_builder):
    """Running out of grace means there is nothing correct to compile yet.

    The tempting alternative -- compile anyway, declare the difference missing
    -- cannot be made right. A turn awaiting its own commit is both an open
    turn and a pending commit, so it counts twice; the append keeps running
    past the timeout, so it may land before the journal is read; and the record
    would then hold the exchange and say it was lost. `503`, and the
    trajectory is left exactly as it was.
    """
    stack = await stack_builder(finish_grace_seconds=0.1)
    created = await stack.create_trajectory()

    turn = asyncio.create_task(
        stack.chat(created, [{"role": "user", "content": "slow"}], headers={"x-mock-delay": "1.0"})
    )
    await asyncio.sleep(0.1)
    refused = await stack.finish(created["id"])

    assert refused.status_code == 503
    assert "still in flight" in refused.json()["detail"]
    assert stack.record(created["id"]) is None, "nothing was committed"
    assert stack.aggregate(created["id"]) is not None, "and nothing was evicted"

    # The turn completes for the client, as text capture always lets it, and
    # its exchange lands in the journal rather than being counted as lost.
    assert (await turn).status_code == 200
    await stack.settle()

    # A retry now succeeds, and the record contains the turn that was in
    # flight -- which is the whole point of not having finalized without it.
    done = await stack.finish(created["id"])
    assert done.status_code == 200
    record = stack.record(created["id"])
    assert len(record.exchanges) == 1
    assert record.trajectory.integrity.complete is True
    assert record.trajectory.integrity.calls_missing == 0


async def test_a_retried_finish_does_not_apply_its_metadata_twice(stack, monkeypatch):
    """After the commit fails, the trajectory is still `finalizing` with the
    finish already recorded. A retry that re-ran steps 1-2 would merge the same
    annotations again and count a second revision for a correction nobody
    made -- and append a second `FinishRequested` for a replay to apply."""
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "hi"}])
    await stack.settle()

    async def refuse(record):
        raise OSError("the volume went away")

    monkeypatch.setattr(stack.runtime.committed, "put", refuse)
    outcome = {"labels": ["scored"], "annotations": {"reward": 1.0}}

    first = await stack.finish(created["id"], **outcome)
    assert first.status_code == 503
    assert stack.aggregate(created["id"]).revision == 1

    # Retried while the disk is still refusing, so the journal survives and
    # can be counted: steps 1-2 must not have run a second time.
    second = await stack.finish(created["id"], **outcome)
    assert second.status_code == 503
    assert stack.aggregate(created["id"]).revision == 1, "no second revision"
    requests = [
        item
        for item in stack.journal_records(created["id"])
        if isinstance(item, journal.FinishRequested)
    ]
    assert len(requests) == 1, "and one FinishRequested for a replay to apply"

    monkeypatch.undo()
    assert (await stack.finish(created["id"], **outcome)).status_code == 200

    record = stack.record(created["id"])
    assert record.trajectory.annotations == {"reward": 1.0}
    assert record.trajectory.labels == ["scored"]
    assert record.revision == 1, "one finish, one revision"


async def test_an_append_after_the_record_is_committed_is_a_failure_not_a_drop(stack_builder):
    """The half that matters for TITO: a response may not close cleanly on an
    append that went nowhere."""
    from skyrl_capture.persistence.active import JournalClosed

    stack = await stack_builder()
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "one"}])
    await stack.finish(created["id"])

    with pytest.raises(JournalClosed):
        await stack.runtime.active.append(
            created["id"], journal.CaptureGap(1, "after the fact", NOW)
        )


# -- lifecycle calls on one trajectory -------------------------------------------
async def test_an_annotation_racing_finish_is_never_lost(stack):
    """Both have awaits in the middle, so without a lock the annotation can
    land on a journal that `finish` has already read and is about to delete --
    accepted, acknowledged, and gone.

    One trajectory would only lose the race sometimes, which is the property
    that makes this kind of bug survive a test suite. Eight at once is enough
    that "sometimes" is every run.
    """
    created = [await stack.create_trajectory() for _ in range(8)]
    for trajectory in created:
        await stack.chat(trajectory, [{"role": "user", "content": "hi"}])
    await stack.settle()

    results = await asyncio.gather(
        *(
            call
            for trajectory in created
            for call in (
                stack.finish(trajectory["id"], labels=["done"]),
                stack.patch(
                    f"/v1/trajectories/{trajectory['id']}/metadata",
                    {"annotations": {"reward": 1.0}},
                ),
            )
        )
    )
    assert all(response.status_code == 200 for response in results)

    # Whichever order each pair serialized in, the reward is in the committed
    # record -- either because finish compiled after it, or because it was
    # applied to the record afterwards. An acknowledged write that is nowhere
    # is the failure this rules out.
    for trajectory in created:
        record = stack.record(trajectory["id"])
        assert record is not None, trajectory["id"]
        assert record.trajectory.annotations == {"reward": 1.0}, trajectory["id"]
        assert "done" in record.trajectory.labels, trajectory["id"]


async def test_two_identical_finishes_at_once_both_succeed(stack):
    """A retry that arrives while the first attempt is still compiling must not
    find a half-finished trajectory."""
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "hi"}])
    await stack.settle()

    first, second = await asyncio.gather(
        stack.finish(created["id"], annotations={"reward": 1.0}),
        stack.finish(created["id"], annotations={"reward": 1.0}),
    )

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert stack.record(created["id"]).revision == 1, "one edit, not two"


async def test_concurrent_metadata_rewrites_do_not_lose_each_other(stack):
    """Two rewrites of a committed record are a read-modify-write on one file."""
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "hi"}])
    await stack.finish(created["id"])

    await asyncio.gather(
        *(
            stack.patch(
                f"/v1/trajectories/{created['id']}/metadata", {"annotations": {f"k{index}": index}}
            )
            for index in range(6)
        )
    )

    record = stack.record(created["id"])
    assert record.trajectory.annotations == {f"k{index}": index for index in range(6)}
    assert record.revision == 6


# -- a reader on the other side of a shared volume --------------------------------
async def test_a_separate_reader_sees_a_committed_record_rewritten(tmp_path):
    """A standalone viewer has no writer to tell it anything. A committed record
    is written once *and rewritten* -- a reward that arrives after a trajectory
    finished replaces it -- so an index keyed on the id alone would skip that
    file for ever and show the pre-reward document indefinitely.
    """
    root = ensure_record(tmp_path / "record")
    active = DiskActiveStore(root)
    committed = DiskCommittedStore(root)
    await active.create(header("tr_rewritten", project="shared"))
    await active.append(
        "tr_rewritten", journal.ExchangeCommitted(*exchange("ex_0", sequence=0), NOW)
    )
    replayed = rebuild(read_journal(active.path_for("tr_rewritten")).records, recovered=False)
    await committed.put(
        compile_record(replayed, status="finished", finished_at=NOW, finish_request_hash="h")
    )
    await active.remove("tr_rewritten")
    await active.close()

    # A reader with no connection to the writer at all.
    reader = RecordReader(root)
    await reader.refresh()
    assert reader.list_trajectories(TrajectoryQuery()).items[0]["annotations"] == {}

    await committed.update_metadata("tr_rewritten", metadata_update(annotations={"reward": 1.0}))
    await reader.refresh()

    row = reader.list_trajectories(TrajectoryQuery()).items[0]
    assert row["annotations"] == {"reward": 1.0}, "the rewrite was picked up"
    assert row["revision"] == 1
    view = await reader.view("tr_rewritten")
    assert view.trajectory.annotations == {"reward": 1.0}, "and the cached detail too"


# -- an export created mid-scan ----------------------------------------------------
async def test_an_export_waits_for_the_directory_to_be_read_through(tmp_path):
    """A listing may be a scan behind; a dataset may not.

    A run export fixes its selection when the request is accepted. If that
    selection came from a half-built index, the artifact silently omits
    whatever the scan had not reached -- and nothing downstream can tell,
    because a short dataset looks exactly like a short run.
    """
    from test_indexing import populate

    from skyrl_capture.export.artifacts import ArtifactStore
    from skyrl_capture.export.jobs import ExportJobStore
    from skyrl_capture.export.service import ExportService
    from skyrl_capture.persistence.layout import export_artifacts_dir, export_jobs_dir

    root = ensure_record(tmp_path / "record")
    identifiers = await populate(root, 8)

    reader = RecordReader(root, batch=1)
    exports = ExportService(
        reader=reader,
        jobs=ExportJobStore(export_jobs_dir(root)),
        artifacts=ArtifactStore(export_artifacts_dir(root)),
        public_url="http://viewer",
    )
    # Nothing has been indexed: this is the state a viewer starts in.
    assert reader.list_trajectories(TrajectoryQuery()).indexing is True

    job = await exports.create(
        format="graph", project="scale", run=None, trajectory=None, options={}
    )

    assert sorted(job["selected_trajectory_ids"]) == sorted(identifiers), (
        "the whole run, not the part the scan had reached"
    )


async def test_a_committed_record_is_not_reopened_by_a_leftover_journal(stack_builder, tmp_path):
    """A process that dies between the commit and the journal's deletion leaves
    both files. Adopting the journal then would reopen a trajectory whose
    record is already written -- and every append to it would fail, because the
    record cannot take another exchange.

    Readers already prefer the committed form. This is the write path agreeing.
    """
    root = tmp_path / "traces"
    stack = await stack_builder(record_dir=root)
    created = await stack.create_trajectory(trajectory_id="tr_leftover")
    await stack.chat(created, [{"role": "user", "content": "hi"}])
    await stack.finish("tr_leftover")

    # Put the journal back, as a crash in that window would have left it.
    await stack.runtime.active.create(header("tr_leftover", project="test-project"))
    assert stack.runtime.active.path_for("tr_leftover").is_file()
    stack.runtime.active._finished.discard("tr_leftover")  # noqa: SLF001 - a new process would not know

    assert await stack.runtime.registry.resolve("tr_leftover") is None
    assert not stack.runtime.active.path_for("tr_leftover").is_file(), "and cleaned up"

    # The route stays closed, and metadata still reaches the committed record.
    assert (await stack.chat(created, [{"role": "user", "content": "again"}])).status_code == 410
    response = await stack.patch(
        "/v1/trajectories/tr_leftover/metadata", {"annotations": {"reward": 1.0}}
    )
    assert response.status_code == 200
    assert stack.record("tr_leftover").trajectory.annotations == {"reward": 1.0}


async def test_a_lifecycle_lock_is_not_kept_for_every_trajectory_ever_named(stack):
    """Held while somebody is using it, and not a moment longer: a long run
    names hundreds of thousands of trajectories."""
    created = [await stack.create_trajectory() for _ in range(5)]
    for trajectory in created:
        await stack.finish(trajectory["id"])
        await stack.patch(
            f"/v1/trajectories/{trajectory['id']}/metadata", {"annotations": {"reward": 1.0}}
        )

    assert stack.runtime.commands._locks.held() == 0  # noqa: SLF001 - the leak is the thing


def test_compression_is_safe_from_many_threads_at_once():
    """A `ZstdCompressor` carries internal state and cannot be shared between
    threads. One module-level instance was not a slow path -- under enough
    concurrency it raised `ZstdError: Src size is incorrect`, and often enough
    it segfaulted the interpreter.

    Every write reaches this from a worker thread: journal appends and
    committed records both go through `asyncio.to_thread`, and per-trajectory
    locking means two trajectories compress at the same moment. So this is the
    ordinary path, not an exotic one.

    The test asserts the fix rather than reproducing the crash -- a test that
    segfaults takes the runner with it and reports nothing.
    """
    import concurrent.futures
    import os

    from skyrl_capture.compression import compress, decompress

    payloads = [os.urandom(3000) * (index % 9 + 1) for index in range(300)]

    def round_trip(data: bytes) -> bool:
        return decompress(compress(data)) == data

    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        assert all(pool.map(round_trip, payloads))


async def test_many_trajectories_finishing_at_once_all_commit(stack):
    """The shape that found it: every finish compresses a record in its own
    worker thread, so a dozen at once is a dozen concurrent compressions."""
    created = [await stack.create_trajectory() for _ in range(12)]
    for trajectory in created:
        await stack.chat(trajectory, [{"role": "user", "content": "hi"}])
    await stack.settle()

    results = await asyncio.gather(
        *(stack.finish(trajectory["id"], annotations={"reward": 1.0}) for trajectory in created)
    )

    assert all(response.status_code == 200 for response in results)
    for trajectory in created:
        record = stack.record(trajectory["id"])
        assert record is not None, trajectory["id"]
        assert len(record.exchanges) == 1
