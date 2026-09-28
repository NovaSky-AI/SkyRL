"""The trajectory headers: what makes a listing cheap, and what keeps it honest.

A header is a finished trajectory's document written beside its record as
plain JSON. Every listing reads these and no records, which is what lets a
cold viewer over a large directory open at all. These tests hold that, and the
rules that let a listing be read from the headers alone.
"""

from __future__ import annotations

import orjson
from test_aggregate import exchange, header
from test_indexing import populate
from test_persistence import NOW

from skyrl_capture.domain.records import metadata_update
from skyrl_capture.persistence import journal
from skyrl_capture.persistence.active import DiskActiveStore, rebuild
from skyrl_capture.persistence.committed import DiskCommittedStore
from skyrl_capture.persistence.headers import header_path, read_header
from skyrl_capture.persistence.layout import ensure_record
from skyrl_capture.reader.records import RecordReader, TrajectoryQuery
from skyrl_capture.writer.compile import compile_record


async def commit_one(root, identifier, *, project, run_id):
    """One finished trajectory, written the way a capture process writes it."""
    active = DiskActiveStore(root)
    committed = DiskCommittedStore(root)
    created = journal.TrajectoryCreated(header(identifier, project=project, run_id=run_id))
    await active.create(created.header)
    row = journal.ExchangeCommitted(*exchange("ex_0", sequence=0), NOW)
    await active.append(identifier, row)
    replayed = rebuild([created, row], recovered=False)
    record = compile_record(
        replayed, status="finished", finished_at=NOW, finish_request_hash="h"
    )
    await committed.put(record)
    await active.remove(identifier)
    await active.close()
    return record


async def test_a_header_sits_beside_its_record_and_needs_no_decoder(tmp_path):
    """Flat and addressed by id, exactly like the record it describes."""
    root = ensure_record(tmp_path / "record")
    await commit_one(root, "tr_0001", project="demo", run_id="grpo-step-1")

    path = header_path(root, "tr_0001")
    assert path.is_file()
    assert path.parent == root / "committed" / path.parent.name
    assert (path.parent / "tr_0001.json.zst").is_file(), "the record, in the same shard"

    # Plain JSON: no zstd, no tokenizer, no capture process.
    raw = orjson.loads(path.read_bytes())
    assert raw["id"] == "tr_0001"
    assert raw["project"] == "demo"
    assert raw["run_id"] == "grpo-step-1"
    assert raw["status"] == "finished"

    # And it round-trips to the same document the API returns.
    document = read_header(path)
    assert document is not None
    assert (document.id, document.project, document.run_id) == (
        "tr_0001",
        "demo",
        "grpo-step-1",
    )


async def test_an_unfinished_trajectory_has_no_header(tmp_path):
    """A header describes a record. Until there is one, there is nothing to
    describe -- the journal is what holds a trajectory in flight."""
    root = ensure_record(tmp_path / "record")
    active = DiskActiveStore(root)
    await active.create(header("tr_live", project="demo", run_id="grpo-step-1"))
    await active.close()

    assert not header_path(root, "tr_live").exists()
    assert list((root / "committed").glob("*/*")) == []


async def test_grouping_by_run_is_a_query_not_a_directory(tmp_path):
    """The point of keeping the layout flat.

    Nothing on disk is arranged by project or run. The hierarchy a viewer
    shows is computed from the headers on read -- a run is the grouping its
    members imply.
    """
    root = ensure_record(tmp_path / "record")
    await commit_one(root, "tr_0001", project="demo", run_id="grpo-step-1")
    await commit_one(root, "tr_0002", project="demo", run_id="grpo-step-1")
    await commit_one(root, "tr_0003", project="demo", run_id="next_turn")
    await commit_one(root, "tr_0004", project="other", run_id="probe")

    # No project or run name appears anywhere in the directory.
    names = {path.name for path in root.rglob("*")}
    assert not {"demo", "other", "grpo-step-1", "next_turn", "probe"} & names

    reader = RecordReader(root)
    await reader.refresh()

    runs = reader.list_runs(project=None, limit=50)
    assert {(run["id"], run["project"], run["trajectory_count"]) for run in runs} == {
        ("grpo-step-1", "demo", 2),
        ("next_turn", "demo", 1),
        ("probe", "other", 1),
    }
    # And the same grouping filters down, which is what the sidebar asks for.
    assert {run["id"] for run in reader.list_runs(project="demo", limit=50)} == {
        "grpo-step-1",
        "next_turn",
    }
    page = reader.list_trajectories(TrajectoryQuery(run_id="grpo-step-1", limit=10))
    assert {row["id"] for row in page.items} == {"tr_0001", "tr_0002"}


async def test_a_listing_reads_headers_and_never_a_record(tmp_path):
    """The whole reason headers exist, asserted rather than assumed."""
    root = ensure_record(tmp_path / "record")
    await populate(root, 6)

    reader = RecordReader(root)
    opened: list[str] = []
    real = reader._committed.get_sync  # noqa: SLF001

    def watched(identifier):
        opened.append(identifier)
        return real(identifier)

    reader._committed.get_sync = watched  # noqa: SLF001
    await reader.refresh()

    page = reader.list_trajectories(TrajectoryQuery(limit=10))
    assert page.total == 6
    assert reader.list_runs(project=None, limit=10)[0]["trajectory_count"] == 6
    assert opened == [], "a listing decompressed a record"


async def test_a_later_revision_replaces_the_header_in_place(tmp_path):
    """A reward that lands after the trajectory finished."""
    root = ensure_record(tmp_path / "record")
    await commit_one(root, "tr_0001", project="demo", run_id="only")
    committed = DiskCommittedStore(root)
    path = header_path(root, "tr_0001")

    before = read_header(path)
    assert before is not None and before.labels == []

    updated = await committed.update_metadata(
        "tr_0001", metadata_update(labels=["scored"], annotations={"reward": 1.0})
    )
    assert updated.trajectory.revision > before.revision

    after = read_header(path)
    assert after is not None, "the same path, replaced in place"
    assert after.labels == ["scored"]
    assert after.annotations == {"reward": 1.0}
    assert after.revision == updated.trajectory.revision

    reader = RecordReader(root)
    await reader.refresh()
    rows = reader.list_trajectories(TrajectoryQuery(limit=10)).items
    assert [(row["id"], row["labels"]) for row in rows] == [("tr_0001", ["scored"])]


async def test_a_committed_record_with_no_header_is_listed_before_its_journal_goes(tmp_path):
    """The window `put` cannot close: a crash between the record and the header.

    The headers are authoritative, so deleting the journal then would leave a
    record no listing can reach. Whoever finds that journal writes the header
    first.
    """
    from skyrl_capture.writer.registry import TrajectoryRegistry

    root = ensure_record(tmp_path / "record")
    await commit_one(root, "tr_0001", project="demo", run_id="only")

    # Re-create the crash: the record is on disk, the header is not, and the
    # journal that could rebuild it is still there.
    header_path(root, "tr_0001").unlink()
    active = DiskActiveStore(root)
    await active.create(header("tr_0001", project="demo", run_id="only"))
    committed = DiskCommittedStore(root)

    registry = TrajectoryRegistry(active=active, committed=committed)
    assert await registry.resolve("tr_0001") is None, "finished, so not adopted"
    await active.close()

    assert header_path(root, "tr_0001").is_file(), (
        "relisted before the journal that could rebuild it was dropped"
    )


async def test_reindex_rebuilds_the_headers_from_the_records(tmp_path):
    """The deliberate way back, for a directory the write path never covered."""
    root = ensure_record(tmp_path / "record")
    identifiers = await populate(root, 5)
    committed = DiskCommittedStore(root)

    # A directory whose headers were deleted, plus one naming nothing.
    orphan = header_path(root, "tr_9999")
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(header_path(root, identifiers[0]).read_bytes())
    for identifier in identifiers:
        header_path(root, identifier).unlink()

    result = committed.headers.rebuild(committed)
    assert result["listed"] == 5
    assert result["dropped"] == 1, "a header naming a record that is not there"
    assert not orphan.exists()

    reader = RecordReader(root)
    await reader.refresh()
    assert reader.list_trajectories(TrajectoryQuery(limit=10)).total == 5


async def test_a_trajectory_whose_files_are_gone_leaves_the_listing(tmp_path):
    """Removal is the case a local writer's fast path used to get wrong.

    When a writer in this process finishes a trajectory it tells the reader
    directly, which is cheaper than waiting for a scan. That shortcut has to
    record the same thing a scan would -- a header, under its path -- or the
    row reaches the listing by a route a rebuild cannot see: kept for ever
    once its files are gone, and dropped while they are still there.
    """
    root = ensure_record(tmp_path / "record")
    await commit_one(root, "tr_0001", project="demo", run_id="only")
    await commit_one(root, "tr_0002", project="demo", run_id="only")

    reader = RecordReader(root)
    # The shortcut, not the scan: this is what `CaptureCommands` calls.
    reader.note_change("tr_0001")
    reader.note_change("tr_0002")
    await reader.apply_pending()
    # No full scan yet, so there is no settled total to give -- but both rows
    # are in the listing, which is what the shortcut is for.
    page = reader.list_trajectories(TrajectoryQuery(limit=10))
    assert sorted(row["id"] for row in page.items) == ["tr_0001", "tr_0002"]
    assert page.indexed_trajectories == 2

    header_path(root, "tr_0002").unlink()
    (root / "committed" / header_path(root, "tr_0002").parent.name / "tr_0002.json.zst").unlink()

    await reader.refresh()
    page = reader.list_trajectories(TrajectoryQuery(limit=10))
    assert [row["id"] for row in page.items] == ["tr_0001"], "the deleted one is gone"
    assert page.total == 1
    # And the one still on disk survived the rebuild that removal triggered.
    assert reader.list_runs(project=None, limit=10)[0]["trajectory_count"] == 1
