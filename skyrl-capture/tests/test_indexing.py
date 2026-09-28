"""Progressive indexing, and the moment a trajectory changes form.

A record directory can hold a hundred thousand trajectories, so the viewer
cannot wait for a complete scan before it answers. It answers immediately, says
how far it has got, and withholds the one number a pager would otherwise print
as fact.

The other half is the transition. A trajectory that finishes moves from
`active/` to `committed/`, and for an instant it is in both. A reader that
showed it twice, or lost it for a moment, would be wrong in a way a person
watching a run would notice at once.
"""

from __future__ import annotations

import asyncio
import time

from test_aggregate import exchange, header
from test_persistence import NOW

from skyrl_capture.domain.records import metadata_update
from skyrl_capture.persistence import journal
from skyrl_capture.persistence.active import DiskActiveStore, rebuild
from skyrl_capture.persistence.committed import DiskCommittedStore
from skyrl_capture.persistence.layout import ensure_record
from skyrl_capture.reader import records as records_module
from skyrl_capture.reader.records import RecordReader, TrajectoryQuery
from skyrl_capture.writer.compile import compile_record


async def populate(root, count: int) -> list[str]:
    """``count`` finished trajectories, written as a capture process would."""
    active = DiskActiveStore(root)
    committed = DiskCommittedStore(root)
    identifiers = []
    for index in range(count):
        identifier = f"tr_{index:04d}"
        identifiers.append(identifier)
        await active.create(header(identifier, project="scale", run_id="run-a"))
        await active.append(
            identifier, journal.ExchangeCommitted(*exchange(f"ex_{index}", sequence=0), NOW)
        )
        replayed = rebuild(
            [
                journal.TrajectoryCreated(header(identifier, project="scale", run_id="run-a")),
                journal.ExchangeCommitted(*exchange(f"ex_{index}", sequence=0), NOW),
            ],
            recovered=False,
        )
        await committed.put(
            compile_record(replayed, status="finished", finished_at=NOW, finish_request_hash="h")
        )
        await active.remove(identifier)
    await active.close()
    return identifiers


async def test_the_first_page_arrives_before_the_scan_finishes(tmp_path):
    root = ensure_record(tmp_path / "record")
    await populate(root, 12)

    # One trajectory per batch, and a read that takes a moment, so the scan
    # is observable rather than instantaneous -- what this is really standing
    # in for is a directory with a hundred thousand records in it.
    #
    # The read being slowed is the run header, because that is what indexing
    # reads. It used to be the committed record, and slowing that no longer
    # slows anything: a listing does not decompress records any more, which is
    # the whole reason `runs/` exists.
    reader = RecordReader(root, batch=1)
    real_read = records_module.read_header

    def slow_read(path):
        time.sleep(0.01)
        return real_read(path)

    records_module.read_header = slow_read
    try:
        first = reader.list_trajectories(TrajectoryQuery(limit=5))
        assert first.indexing is True
        assert first.total is None, "a total that is about to change is worse than none"
        assert first.items == [], "nothing read yet, and answering anyway"

        reader.start()
        seen: list[int] = []
        for _ in range(400):
            await asyncio.sleep(0.005)
            page = reader.list_trajectories(TrajectoryQuery(limit=5))
            seen.append(page.indexed_trajectories)
            if not page.indexing:
                break

        assert any(0 < count < 12 for count in seen), f"never saw a partial index: {seen}"
        final = reader.list_trajectories(TrajectoryQuery(limit=5))
        assert final.indexing is False
        assert final.total == 12, "a stable total, once there is one to give"
        assert len(final.items) == 5
        assert final.next_cursor is not None
    finally:
        records_module.read_header = real_read
        await reader.stop()


async def test_a_trajectory_in_both_places_is_listed_once_as_finished(tmp_path):
    """The window between the committed rename and the journal's deletion.

    Both files exist, and the reader prefers the committed one -- which is what
    makes finishing invisible to somebody watching the listing.
    """
    root = ensure_record(tmp_path / "record")
    active = DiskActiveStore(root)
    committed = DiskCommittedStore(root)
    await active.create(header("tr_both", project="scale"))
    await active.append(
        "tr_both", journal.ExchangeCommitted(*exchange("ex_0", sequence=0), NOW)
    )
    replayed = rebuild(
        [
            journal.TrajectoryCreated(header("tr_both", project="scale")),
            journal.ExchangeCommitted(*exchange("ex_0", sequence=0), NOW),
        ],
        recovered=False,
    )
    await committed.put(
        compile_record(replayed, status="finished", finished_at=NOW, finish_request_hash="h")
    )
    await active.close()

    reader = RecordReader(root)
    await reader.refresh()
    page = reader.list_trajectories(TrajectoryQuery(limit=50))

    assert [row["id"] for row in page.items] == ["tr_both"], "once, not twice"
    assert page.items[0]["status"] == "finished"
    view = await reader.view("tr_both")
    assert view.trajectory.status == "finished"


async def test_a_refresh_picks_up_new_journals_and_newly_finished_ones(tmp_path):
    """Manual refresh reaches both directions, or a viewer would go stale in
    exactly the way a person notices: the run that just finished still says
    it is running."""
    root = ensure_record(tmp_path / "record")
    active = DiskActiveStore(root)
    committed = DiskCommittedStore(root)
    reader = RecordReader(root)
    await reader.refresh()
    assert reader.list_trajectories(TrajectoryQuery()).items == []

    await active.create(header("tr_live", project="scale"))
    await reader.refresh()
    page = reader.list_trajectories(TrajectoryQuery())
    assert [row["status"] for row in page.items] == ["created"]

    await active.append(
        "tr_live", journal.ExchangeCommitted(*exchange("ex_0", sequence=0), NOW)
    )
    await reader.refresh()
    assert reader.list_trajectories(TrajectoryQuery()).items[0]["capture"]["exchange_count"] == 1

    replayed = rebuild(
        [
            journal.TrajectoryCreated(header("tr_live", project="scale")),
            journal.ExchangeCommitted(*exchange("ex_0", sequence=0), NOW),
        ],
        recovered=False,
    )
    await committed.put(
        compile_record(replayed, status="finished", finished_at=NOW, finish_request_hash="h")
    )
    await active.remove("tr_live")
    await reader.refresh()

    page = reader.list_trajectories(TrajectoryQuery())
    assert [row["status"] for row in page.items] == ["finished"]
    assert page.total == 1
    await active.close()


async def test_an_open_detail_follows_its_journal_as_it_grows(tmp_path):
    """A refresh has to reach an open trajectory's own page, not only the
    listings -- that is the page somebody watching a run is looking at."""
    root = ensure_record(tmp_path / "record")
    active = DiskActiveStore(root)
    reader = RecordReader(root)
    await active.create(header("tr_growing", project="scale"))
    await reader.refresh()
    assert len((await reader.view("tr_growing")).exchanges) == 0

    await active.append(
        "tr_growing", journal.ExchangeCommitted(*exchange("ex_0", sequence=0), NOW)
    )
    await reader.refresh()
    assert len((await reader.view("tr_growing")).exchanges) == 1, "the cached detail was dropped"

    await active.append("tr_growing", journal.MetadataUpdated(metadata_update(labels=["late"]), NOW))
    await reader.refresh()
    assert (await reader.view("tr_growing")).trajectory.labels == ["late"]
    await active.close()


async def test_a_page_does_not_cost_the_whole_directory(tmp_path):
    """The listing snapshot is rebuilt when the index changes and not once per
    page. At the scale this format is meant for -- a hundred thousand
    trajectories -- rebuilding every document to answer a page of fifty is a
    cost that does not show up until somebody has one."""
    root = ensure_record(tmp_path / "record")
    await populate(root, 6)
    reader = RecordReader(root)
    await reader.refresh()

    first = reader.list_trajectories(TrajectoryQuery(limit=2))
    again = reader.list_trajectories(TrajectoryQuery(limit=2))
    assert first.items == again.items
    assert reader._documents() is reader._documents()  # noqa: SLF001 - the cache is the point

    # And a change invalidates it rather than serving a stale page.
    active = DiskActiveStore(root)
    await active.create(header("tr_9999", project="scale", run_id="run-a"))
    await reader.refresh()
    assert reader.list_trajectories(TrajectoryQuery(limit=2)).total == 7
    assert "tr_9999" in {row["id"] for row in reader.list_trajectories(TrajectoryQuery()).items}
    await active.close()
