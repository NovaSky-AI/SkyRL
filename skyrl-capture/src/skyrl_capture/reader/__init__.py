"""Reading a record: one reader over active journals and committed records.

`RecordReader` is the whole of it. There is no live reader and no offline
reader, because there is nothing for them to disagree about: a trajectory is
mid-run in `active/` or finished in `committed/`, and the reader prefers the
committed form wherever both exist. The viewer, the CLI and the exporters all
go through it, so none of them can tell a running capture from a finished one.
"""

from skyrl_capture.reader.records import (
    RecordReader,
    TrajectoryPage,
    TrajectoryQuery,
    exchange_public,
)

__all__ = ["RecordReader", "TrajectoryPage", "TrajectoryQuery", "exchange_public"]
