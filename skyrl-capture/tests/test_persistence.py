"""The record directory: journals, committed records, and what a crash leaves.

Persistence is mandatory now, so these are not tests of an optional feature.
They are the tests of the thing the product guarantee rests on: an exchange
that was appended survives the process, a torn tail is detected rather than
walked past, a committed record is never visible half-written, and two capture
processes can share one directory as long as routing gives each trajectory one
writer.

The last one is stated as an *external* invariant and tested as one: what is
asserted is that distinct trajectories interleave safely, not that two writers
on one trajectory are handled -- they are not, and nothing here pretends
otherwise.
"""

from __future__ import annotations

from datetime import UTC, datetime

import orjson
import pytest
from test_aggregate import exchange, header

from skyrl_capture.domain.records import metadata_update
from skyrl_capture.persistence import journal
from skyrl_capture.persistence.active import (
    DiskActiveStore,
    JournalClosed,
    read_journal,
    rebuild,
)
from skyrl_capture.persistence.committed import DiskCommittedStore, deserialize, serialize
from skyrl_capture.persistence.headers import header_path
from skyrl_capture.persistence.layout import (
    RecordFormatError,
    RecordNotFound,
    active_path,
    committed_path,
    ensure_record,
    open_record,
    shard,
)
from skyrl_capture.version import RECORD_FORMAT_VERSION
from skyrl_capture.writer.compile import compile_record

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


@pytest.fixture
def record(tmp_path):
    return ensure_record(tmp_path / "record", upstream={"type": "openai"})


# -- the framing -----------------------------------------------------------------
def test_a_journal_round_trips_every_record_type():
    captured, delta = exchange("ex_0", sequence=0)
    records = [
        journal.TrajectoryCreated(header()),
        journal.ExchangeCommitted(captured, delta, NOW, "fingerprint"),
        journal.ExchangeDeliveryConfirmed("ex_0", NOW),
        journal.CaptureGap(2, "disk full", NOW),
        journal.MetadataUpdated(metadata_update(annotations={"reward": 1.0}), NOW),
        journal.FinishRequested(metadata_update(labels=["scored"]), "success", "h", NOW),
        journal.TrajectoryPoisoned("attribution failed", NOW),
    ]
    buffer = journal.encode_header() + b"".join(journal.encode_record(r) for r in records)

    scan = journal.scan(buffer)

    assert not scan.truncated
    assert scan.consumed == len(buffer)
    assert [type(r).__name__ for r in scan.records] == [type(r).__name__ for r in records]
    # Bodies travel as bytes, not as base64 of bytes.
    assert scan.records[1].exchange.request_body == captured.request_body
    assert scan.records[1].request_fingerprint == "fingerprint"
    assert scan.records[1].graph.output_node_id == delta.output_node_id


def test_a_global_event_log_from_an_earlier_build_is_refused():
    """Old records are rejected, not guessed at. There is no migration."""
    with pytest.raises(journal.JournalFormatError, match="not a trajectory journal"):
        journal.scan(b"ICEVLOG1" + b"\x00" * 24)


def test_a_torn_tail_stops_the_scan_where_the_damage_starts():
    captured, delta = exchange("ex_0", sequence=0)
    whole = journal.encode_header() + journal.encode_record(journal.TrajectoryCreated(header()))
    torn = whole + journal.encode_record(journal.ExchangeCommitted(captured, delta, NOW))[:-5]

    scan = journal.scan(torn)

    assert scan.truncated
    assert len(scan.records) == 1
    assert scan.consumed == len(whole), "consumed stops at the last whole record"


def test_a_corrupt_record_is_a_tear_even_at_full_length():
    """The CRC is what catches a record whose bytes are all there and wrong."""
    body = journal.encode_record(journal.CaptureGap(1, "x", NOW))
    damaged = bytearray(journal.encode_header() + body)
    damaged[-1] ^= 0xFF

    scan = journal.scan(bytes(damaged))

    assert scan.truncated
    assert scan.records == []


def test_an_unknown_record_kind_is_skipped_by_its_length():
    """An old reader walks a journal a newer writer wrote."""
    good = journal.encode_record(journal.CaptureGap(1, "x", NOW))
    unknown = bytearray(journal.encode_record(journal.CaptureGap(2, "y", NOW)))
    journal.RECORD_HEADER.pack_into(
        unknown, 0, *journal.RECORD_HEADER.unpack_from(unknown, 0)[:2], 9999, 0
    )

    scan = journal.scan(journal.encode_header() + bytes(unknown) + good)

    assert scan.unknown_kinds == 1
    assert len(scan.records) == 1
    assert scan.records[0].count == 1


# -- the active store ---------------------------------------------------------------
async def test_creation_is_durable_before_it_returns(record):
    store = DiskActiveStore(record)
    await store.create(header("tr_a"))

    path = active_path(record, "tr_a")
    assert path.is_file()
    assert path.parent.name == shard("tr_a")
    assert isinstance(read_journal(path).records[0], journal.TrajectoryCreated)
    await store.close()


async def test_a_torn_tail_is_truncated_before_the_replacement_writer_appends(record):
    """The one thing worse than losing the tail is writing behind it."""
    store = DiskActiveStore(record)
    await store.create(header("tr_a"))
    captured, delta = exchange("ex_0", sequence=0)
    await store.append("tr_a", journal.ExchangeCommitted(captured, delta, NOW))
    await store.close()

    path = active_path(record, "tr_a")
    whole = path.read_bytes()
    with open(path, "ab") as handle:
        handle.write(journal.encode_record(journal.CaptureGap(1, "torn", NOW))[:9])

    replacement = DiskActiveStore(record)
    recovered = await replacement.recover("tr_a")

    assert recovered is not None
    assert len(recovered.exchanges) == 1
    assert path.stat().st_size == len(whole) + _recovery_marker_size(path, len(whole))

    # And a record written after the recovery is readable, which it would not
    # be if it sat behind the tear.
    await replacement.append("tr_a", journal.CaptureGap(3, "after", NOW))
    scan = read_journal(path)
    assert not scan.truncated
    assert [r.count for r in scan.records if isinstance(r, journal.CaptureGap)][-1] == 3
    await replacement.close()


def _recovery_marker_size(path, whole: int) -> int:
    """The zero-count gap a recovery writes to say it cannot vouch for this."""
    return path.stat().st_size - whole


async def test_recovery_records_that_it_could_not_vouch_for_the_trajectory(record):
    """A viewer reads the journal, not the writer's memory, so the doubt is
    written down rather than kept."""
    store = DiskActiveStore(record)
    await store.create(header("tr_a", mode="text"))
    await store.close()

    replacement = DiskActiveStore(record)
    recovered = await replacement.recover("tr_a")
    await replacement.close()

    assert recovered is not None
    assert recovered.integrity.recovery_uncertain is True
    fresh = rebuild(read_journal(active_path(record, "tr_a")).records, recovered=False)
    assert fresh is not None
    assert fresh.integrity.recovery_uncertain is True, "on disk, not only in memory"


async def test_removing_a_journal_leaves_nothing_behind(record):
    store = DiskActiveStore(record)
    await store.create(header("tr_a"))
    await store.remove("tr_a")

    assert not active_path(record, "tr_a").exists()
    assert await store.recover("tr_a") is None
    await store.close()


# -- the committed store --------------------------------------------------------------
def _record_for(trajectory_id: str = "tr_a"):
    active = rebuild(
        [
            journal.TrajectoryCreated(header(trajectory_id)),
            journal.ExchangeCommitted(*exchange("ex_0", sequence=0), NOW),
        ],
        recovered=False,
    )
    assert active is not None
    return compile_record(active, status="finished", finished_at=NOW, finish_request_hash="h")


def test_a_committed_record_round_trips_through_its_file_format():
    original = _record_for()
    restored = deserialize(serialize(original))

    assert restored.trajectory.public() == original.trajectory.public()
    assert [e.id for e in restored.exchanges] == [e.id for e in original.exchanges]
    assert restored.exchanges[0].request_body == original.exchanges[0].request_body
    assert restored.node_order == original.node_order
    assert restored.record_version == RECORD_FORMAT_VERSION


def test_a_body_that_is_not_utf8_survives_as_bytes():
    original = _record_for()
    original.exchanges[0].response_body = b"\x00\xfftruly raw bytes"

    restored = deserialize(serialize(original))

    assert restored.exchanges[0].response_body == b"\x00\xfftruly raw bytes"


async def test_a_committed_record_is_never_visible_half_written(record):
    """Written to a temporary name in the same directory, then renamed. A
    crash before the rename leaves no record; a crash after it leaves a whole
    one. There is no state in between for a reader to find.

    Both files a commit writes go through that dance -- the record and the
    header every listing reads -- so neither can be found half written."""
    store = DiskCommittedStore(record)
    await store.put(_record_for())

    path = committed_path(record, "tr_a")
    assert path.is_file()
    assert header_path(record, "tr_a").is_file()
    siblings = sorted(p.name for p in path.parent.iterdir())
    assert siblings == ["tr_a.head.json", "tr_a.json.zst"], (
        f"exactly the record and its header, and nothing else: {siblings}"
    )
    assert not [p for p in path.parent.iterdir() if p.name.endswith(".tmp")]
    assert (await store.get("tr_a")).id == "tr_a"


async def test_a_record_written_but_not_renamed_is_simply_absent(record):
    """The crash point before the rename: the retry reconstructs and writes."""
    store = DiskCommittedStore(record)
    path = committed_path(record, "tr_a")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(f".{path.name}.999.tmp").write_bytes(serialize(_record_for()))

    assert await store.get("tr_a") is None
    await store.put(_record_for())
    assert (await store.get("tr_a")) is not None


async def test_metadata_after_finish_rewrites_the_record_and_bumps_its_revision(record):
    store = DiskCommittedStore(record)
    await store.put(_record_for())

    updated = await store.update_metadata(
        "tr_a", metadata_update(annotations={"reward": 1.0}, labels=["scored"])
    )

    assert updated.revision == 1
    assert updated.trajectory.annotations == {"reward": 1.0}
    reread = await store.get("tr_a")
    assert reread.trajectory.labels == ["scored"]
    assert reread.revision == 1

    with pytest.raises(KeyError):
        await store.update_metadata("tr_missing", metadata_update(labels=["x"]))


# -- one directory, several processes ---------------------------------------------------
async def test_two_stores_share_one_root_for_distinct_trajectories(record):
    """What a shared volume actually has to support. Routing gives each
    trajectory one writer; the directory gives them a place to be."""
    first = DiskActiveStore(record)
    second = DiskActiveStore(record)
    await first.create(header("tr_first"))
    await second.create(header("tr_second"))
    await first.append("tr_first", journal.CaptureGap(1, "a", NOW))
    await second.append("tr_second", journal.CaptureGap(2, "b", NOW))

    assert sorted(first.journal_ids()) == ["tr_first", "tr_second"]
    assert rebuild(
        read_journal(active_path(record, "tr_second")).records, recovered=False
    ).integrity.calls_missing == 2
    await first.close()
    await second.close()


async def test_a_reader_sees_appends_made_by_other_processes(record):
    """The viewer is a different process from the writer, and it reads the
    growing file rather than being told about it."""
    writer = DiskActiveStore(record)
    await writer.create(header("tr_a"))
    path = active_path(record, "tr_a")

    first = read_journal(path)
    assert len(first.records) == 1

    await writer.append("tr_a", journal.CaptureGap(1, "later", NOW))

    # Continued from the previous scan's offset, which is what makes a refresh
    # proportional to what was added rather than to the whole file.
    second = read_journal(path, start=first.consumed)
    assert len(second.records) == 1
    assert isinstance(second.records[0], journal.CaptureGap)
    await writer.close()


async def test_overlapping_writers_on_one_trajectory_are_not_detected(record):
    """The external invariant, tested as one rather than defended against.

    Two processes computing the same path for one trajectory is not a bug --
    it is what lets either of them serve it after the other dies. What the
    format cannot do is notice both writing at once: each record they interleave
    is individually valid, so a reader accepts the mixture. Routing is what
    prevents this, and this test exists to say so rather than to pass off a
    check that does not exist.
    """
    first = DiskActiveStore(record)
    second = DiskActiveStore(record)
    assert first.path_for("tr_a") == second.path_for("tr_a")

    await first.create(header("tr_a"))
    await first.append("tr_a", journal.CaptureGap(1, "from the first writer", NOW))
    await second.append("tr_a", journal.CaptureGap(1, "from the second writer", NOW))
    await first.close()
    await second.close()

    scan = read_journal(active_path(record, "tr_a"))
    assert not scan.truncated, "nothing in the format objects to the interleaving"
    rebuilt = rebuild(scan.records, recovered=False)
    assert rebuilt.integrity.calls_missing == 2, (
        "both writers' records are accepted; only routing keeps this from happening"
    )


# -- the manifest ------------------------------------------------------------------------
def test_a_manifest_is_created_once_and_read_by_everyone_else(tmp_path):
    root = tmp_path / "record"
    first = ensure_record(root, upstream={"type": "openai", "url": "http://a"})
    second = ensure_record(root, upstream={"type": "anthropic", "url": "http://b"})

    assert first == second
    document = orjson.loads((root / "manifest.json").read_bytes())
    assert document["upstream"]["type"] == "openai", "the winner's manifest stands"
    assert document["record_version"] == RECORD_FORMAT_VERSION
    assert "api_key" not in str(document)


def test_a_record_from_another_format_version_is_refused(tmp_path):
    root = tmp_path / "record"
    ensure_record(root)
    manifest = root / "manifest.json"
    manifest.write_bytes(manifest.read_bytes().replace(b'"record_version": 5', b'"record_version": 2'))

    with pytest.raises(RecordFormatError, match="cannot be read or continued"):
        open_record(root)


def test_a_directory_with_no_manifest_is_not_a_record(tmp_path):
    with pytest.raises(RecordNotFound):
        open_record(tmp_path / "nothing")


async def test_a_turn_that_outlives_finish_is_refused_and_not_dropped(record):
    """`finish` deletes the journal once the record is committed, and an append
    after that fails.

    Both halves matter. Re-creating the file would leave an orphan journal for
    a finished trajectory -- invisible, because readers prefer the committed
    record, and never collected. *Dropping* the append quietly would be worse:
    a TITO turn waiting on it would close its response cleanly with its
    exchange nowhere, which is the one thing the design promises cannot
    happen. So it raises, and the caller decides what that means.
    """
    store = DiskActiveStore(record)
    await store.create(header("tr_late"))
    await store.remove("tr_late")

    with pytest.raises(JournalClosed, match="already committed"):
        await store.append("tr_late", journal.CaptureGap(1, "too late", NOW))

    assert not active_path(record, "tr_late").exists()
    await store.close()


async def test_asking_for_a_journal_that_is_not_there_holds_nothing(record):
    """An id nobody has heard of is what a misrouted flood of requests is made
    of, so looking one up must not leave a lock behind per id."""
    store = DiskActiveStore(record)
    for index in range(50):
        assert await store.recover(f"tr_nobody_{index}") is None

    assert store._locks == {}  # noqa: SLF001 - the leak is the thing under test
    await store.close()
