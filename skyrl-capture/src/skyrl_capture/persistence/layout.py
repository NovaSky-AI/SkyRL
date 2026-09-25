"""The record directory: what is in it, and where each file goes.

    record/
    |-- manifest.json
    |-- active/
    |   `-- <shard>/tr_....capture          one journal per unfinished trajectory
    |-- committed/
    |   |-- <shard>/tr_....json.zst         one compiled record per finished one
    |   `-- <shard>/tr_....head.json        its document, for listings
    `-- exports/
        |-- jobs/
        `-- artifacts/

Every path is a deterministic function of the trajectory id, so two processes
sharing a directory never have to agree on anything: they compute the same
path for the same trajectory, and routing guarantees only one of them is
writing it. There is no allocation step, no segment range and no directory
that has to be rewritten as the run proceeds.

``manifest.json`` holds static format information and credential-free upstream
provenance, and nothing that changes while a run is in progress. Several
processes may race to create it; whoever loses reads what the winner wrote and
checks that it is compatible.

The shard prefix keeps a directory from holding a million entries. Two hex
characters of a stable hash of the id -- stable across processes and Python
versions, so `PYTHONHASHSEED` cannot move a trajectory's file.
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import orjson

from skyrl_capture.version import (
    DERIVATION_VERSION,
    JOURNAL_FORMAT_VERSION,
    RECORD_FORMAT_VERSION,
    SCHEMA_VERSION,
)

MANIFEST = "manifest.json"
ACTIVE_DIR = "active"
COMMITTED_DIR = "committed"
EXPORTS_DIR = "exports"
EXPORT_JOBS_DIR = "jobs"
EXPORT_ARTIFACTS_DIR = "artifacts"
JOURNAL_SUFFIX = ".capture"
COMMITTED_SUFFIX = ".json.zst"
#: The uncompressed document beside each record, which every listing reads.
HEADER_SUFFIX = ".head.json"


class RecordNotFound(KeyError):
    pass


class RecordFormatError(Exception):
    """A record this build cannot read."""


def shard(trajectory_id: str) -> str:
    """The two-character directory a trajectory's files live under."""
    return hashlib.blake2b(trajectory_id.encode("utf-8"), digest_size=1).hexdigest()


def active_path(root: Path, trajectory_id: str) -> Path:
    return root / ACTIVE_DIR / shard(trajectory_id) / f"{trajectory_id}{JOURNAL_SUFFIX}"


def committed_path(root: Path, trajectory_id: str) -> Path:
    return root / COMMITTED_DIR / shard(trajectory_id) / f"{trajectory_id}{COMMITTED_SUFFIX}"


def active_paths(root: Path) -> list[Path]:
    """Every active journal under ``root``, in a stable order."""
    return scan_shards(root / ACTIVE_DIR, JOURNAL_SUFFIX)


def committed_paths(root: Path) -> list[Path]:
    """Every committed record under ``root``, in a stable order."""
    return scan_shards(root / COMMITTED_DIR, COMMITTED_SUFFIX)


def scan_shards(directory: Path, suffix: str) -> list[Path]:
    """Every file under ``directory``'s shards ending in ``suffix``.

    A shard is pruned as soon as it empties, so one can be removed between
    this listing the shards and reading inside them. A shard that vanishes
    mid-scan held nothing worth reporting -- that is the only condition under
    which it could be pruned -- so it is skipped rather than raised.
    """
    try:
        shards = sorted(directory.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return []
    found: list[Path] = []
    for shard_dir in shards:
        try:
            entries = sorted(
                path for path in shard_dir.iterdir() if path.name.endswith(suffix)
            )
        except (FileNotFoundError, NotADirectoryError):
            continue
        found.extend(entries)
    return found


def trajectory_id_of(path: Path, suffix: str) -> str:
    return path.name[: -len(suffix)]


def export_jobs_dir(root: Path) -> Path:
    return root / EXPORTS_DIR / EXPORT_JOBS_DIR


def export_artifacts_dir(root: Path) -> Path:
    return root / EXPORTS_DIR / EXPORT_ARTIFACTS_DIR


# -- the manifest -------------------------------------------------------------------
def manifest_document(upstream: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "record_version": RECORD_FORMAT_VERSION,
        "journal_format_version": JOURNAL_FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "derivation_version": DERIVATION_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "upstream": upstream or {},
    }


def read_manifest(root: Path) -> dict[str, Any]:
    """The manifest of an existing record, or `RecordNotFound`."""
    manifest = root / MANIFEST
    if not manifest.is_file():
        raise RecordNotFound(f"no record at {root}: {MANIFEST} is missing")
    return orjson.loads(manifest.read_bytes())


def check_compatible(root: Path, document: dict[str, Any]) -> None:
    version = document.get("record_version")
    if version != RECORD_FORMAT_VERSION:
        raise RecordFormatError(
            f"{root} holds a record in format {version!r}, but this build reads "
            f"{RECORD_FORMAT_VERSION}. A global event log written before per-trajectory "
            "persistence cannot be read or continued; export it with the build that "
            "wrote it, or re-capture."
        )


def open_record(root: str | Path) -> Path:
    """The root of an existing, compatible record. Raises `RecordNotFound`."""
    path = Path(root).expanduser()
    check_compatible(path, read_manifest(path))
    return path


def ensure_record(root: str | Path, *, upstream: dict[str, Any] | None = None) -> Path:
    """Open the record at ``root``, creating its layout if it is new.

    Several capture processes may start against one directory at the same
    time. The manifest is written to a temporary name and linked into place,
    so exactly one of them creates it and the rest read what that one wrote
    and check that it is compatible with this build.
    """
    path = Path(root).expanduser()
    for directory in (
        path / ACTIVE_DIR,
        path / COMMITTED_DIR,
        export_jobs_dir(path),
        export_artifacts_dir(path),
    ):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = path / MANIFEST
    if not manifest.is_file():
        temporary = path / f".{MANIFEST}.{os.getpid()}.{uuid4().hex}.tmp"
        temporary.write_bytes(
            orjson.dumps(manifest_document(upstream), option=orjson.OPT_INDENT_2)
        )
        try:
            # `link` rather than `replace`: a loser must not overwrite the
            # winner's manifest, and this is the one atomic create-if-absent
            # every POSIX filesystem and RWX PVC agrees on.
            os.link(temporary, manifest)
        except FileExistsError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
    check_compatible(path, read_manifest(path))
    return path


def fsync_dir(directory: Path) -> None:
    """Make a rename or a create in ``directory`` durable.

    A file's own fsync does not promise its directory entry survives, which is
    exactly the promise an atomic rename needs.
    """
    handle = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)
