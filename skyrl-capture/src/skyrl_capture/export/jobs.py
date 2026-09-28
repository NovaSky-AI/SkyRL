"""Export jobs, as files beside the record.

A bulk export used to be four events in the global log, which put a viewer
concern in the middle of the capture write path: creating one wrote to the
same log inference was writing to, and reading one meant replaying it. It is a
job record now -- one small JSON file per job under `exports/jobs`, replaced
atomically on each transition.

One viewer process owns execution, so there is no claiming protocol and no
lease. A job whose process died stays `running` and is picked up on the next
start, which is the correct behaviour for something that is deterministic and
idempotent: re-running it writes the same artifact.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from skyrl_capture.domain.records import now
from skyrl_capture.export.artifacts import safe_segment


@dataclass
class ExportJob:
    id: str
    format: str
    project: str | None
    run_id: str | None
    trajectory_id: str | None
    selected_trajectory_ids: list[str]
    options: dict[str, Any]
    status: str = "pending"
    created_at: str = ""
    updated_at: str = ""
    output_uri: str | None = None
    byte_count: int | None = None
    record_count: int | None = None
    checksum: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        stamp = now().isoformat()
        self.created_at = self.created_at or stamp
        self.updated_at = self.updated_at or stamp

    def touch(self, status: str) -> None:
        self.status = status
        self.updated_at = now().isoformat()

    def row(self) -> dict[str, Any]:
        """The shape `export_public` reads. Timestamps stay strings: a job
        record is written and read as JSON and never as a datetime."""
        return asdict(self)


@dataclass
class ExportJobStore:
    """The `exports/jobs` directory. One file per job."""

    root: Path
    _cache: dict[str, ExportJob] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser()

    def path_for(self, identifier: str) -> Path:
        return self.root / f"{safe_segment(identifier)}.json"

    async def put(self, job: ExportJob) -> None:
        self._cache[job.id] = job
        await asyncio.to_thread(self._put_sync, job)

    def _put_sync(self, job: ExportJob) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(job.id)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(orjson.dumps(job.row(), option=orjson.OPT_INDENT_2))
        os.replace(temporary, path)

    async def get(self, identifier: str) -> ExportJob | None:
        found = self._cache.get(identifier)
        if found is not None:
            return found
        loaded = await asyncio.to_thread(self._get_sync, identifier)
        if loaded is not None:
            self._cache[identifier] = loaded
        return loaded

    def _get_sync(self, identifier: str) -> ExportJob | None:
        try:
            return ExportJob(**orjson.loads(self.path_for(identifier).read_bytes()))
        except (FileNotFoundError, TypeError, ValueError):
            return None

    async def unfinished(self) -> list[ExportJob]:
        """Jobs left `pending` or `running`, oldest first.

        A `running` job is included on purpose: one viewer owns execution, so a
        job in that state is one whose process went away, and re-running a
        deterministic export is cheaper than leaving it stuck.
        """
        return await asyncio.to_thread(self._unfinished_sync)

    def _unfinished_sync(self) -> list[ExportJob]:
        if not self.root.is_dir():
            return []
        jobs: list[ExportJob] = []
        for path in sorted(self.root.iterdir()):
            if path.suffix != ".json":
                continue
            job = self._get_sync(path.stem)
            if job is not None and job.status in ("pending", "running"):
                jobs.append(job)
        jobs.sort(key=lambda job: job.created_at)
        return jobs


def parse_time(value: str | datetime | None) -> datetime | None:
    if isinstance(value, datetime) or value is None:
        return value
    return datetime.fromisoformat(value)
