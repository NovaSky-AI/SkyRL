"""Persistence: the record directory, and the two stores that write it.

`ActiveStore` holds one append-only journal per unfinished trajectory;
`CommittedStore` holds one compiled record per finished one. Between them they
are the whole of what outlives a capture process, and the whole of what a
viewer reads. `HeaderIndex` writes each finished trajectory's document beside
its record as plain JSON, so building a listing costs a pass over small files
rather than a decompression of every record. Grouping those documents by
project and run is a query the reader answers, not a shape on disk.

There is no global log, no sequence to coordinate and no compaction -- a
trajectory's files are named by its id, and routing rather than storage is
what makes one process the writer of any of them.

A PostgreSQL implementation later implements these same product operations. It
does not expose SQL to the capture path.
"""

from skyrl_capture.persistence.active import ActiveStore, DiskActiveStore, rebuild
from skyrl_capture.persistence.committed import CommittedStore, DiskCommittedStore
from skyrl_capture.persistence.headers import DiskHeaderIndex, HeaderIndex
from skyrl_capture.persistence.layout import (
    RecordFormatError,
    RecordNotFound,
    ensure_record,
    open_record,
)

__all__ = [
    "ActiveStore",
    "CommittedStore",
    "DiskActiveStore",
    "DiskCommittedStore",
    "DiskHeaderIndex",
    "HeaderIndex",
    "RecordFormatError",
    "RecordNotFound",
    "ensure_record",
    "open_record",
    "rebuild",
]
