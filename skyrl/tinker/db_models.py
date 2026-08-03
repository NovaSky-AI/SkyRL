"""Database models for the Tinker API."""

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from sqlalchemy import DateTime, Index, event
from sqlalchemy.engine import url as sqlalchemy_url
from sqlmodel import JSON, Field, SQLModel

from skyrl.tinker import types
from skyrl.utils.log import logger

# Request and result payloads are dominated by long numeric arrays (tokens,
# logprobs, weights), and encoding them is a large share of the cost of both
# submitting a request and scanning the queue. orjson is several times faster
# than the stdlib on exactly that shape, so prefer it when it is installed.
#
# OPT_NON_STR_KEYS and OPT_SERIALIZE_NUMPY keep it at least as permissive as the
# stdlib for int-keyed metric dicts and numpy scalars coming out of backends.
try:
    import orjson

    _ORJSON_OPTIONS = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY

    def _json_dumps(obj) -> str:
        return orjson.dumps(obj, option=_ORJSON_OPTIONS).decode()

    _json_loads = orjson.loads
except ImportError:  # pragma: no cover - exercised only without orjson installed
    _json_dumps = json.dumps
    _json_loads = json.loads


def json_engine_kwargs() -> dict:
    """Engine kwargs installing the fastest available codec for JSON columns.

    Splat into ``create_engine``/``create_async_engine``.
    """
    return {"json_serializer": _json_dumps, "json_deserializer": _json_loads}


# Filesystem types on which SQLite locking and WAL shared memory are unreliable.
_NETWORK_FS_TYPES = frozenset(
    {
        "nfs",
        "nfs3",
        "nfs4",
        "cifs",
        "smbfs",
        "smb3",
        "9p",
        "afs",
        "ceph",
        "glusterfs",
        "lustre",
        "beegfs",
        "gpfs",
        "davfs",
    }
)


# Page cache per connection, in KiB (negative values mean KiB rather than pages).
# Request payloads are large, so the default 2 MB cache thrashes on the engine's
# scheduling scans.
_SQLITE_CACHE_KIB = 65536


def enable_sqlite_wal(engine) -> None:
    """Enable WAL mode, relaxed fsync, and busy timeout for SQLite engines.

    WAL mode allows concurrent readers with a single writer.
    synchronous=NORMAL is the recommended pairing with WAL: it skips the
    per-transaction fsync (a major writer bottleneck) while still
    guaranteeing database integrity.
    Busy timeout makes SQLite retry internally instead of immediately
    raising 'database is locked'.

    No-op for non-SQLite engines.
    """
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute(f"PRAGMA cache_size=-{_SQLITE_CACHE_KIB}")
        cursor.close()


def create_missing_indexes(engine) -> None:
    """Create any declared index that does not exist yet.

    ``SQLModel.metadata.create_all`` skips tables it already finds, so indexes
    added after a database was first created are never built. Call this after
    ``create_all`` so existing databases pick them up.
    """
    for table in SQLModel.metadata.tables.values():
        for index in table.indexes:
            index.create(bind=engine, checkfirst=True)


def _filesystem_type(path: Path) -> str | None:
    """Return the filesystem type of the mount containing ``path``, or None if unknown.

    Linux-only (reads /proc/self/mounts); returns None elsewhere.
    """
    try:
        mounts = Path("/proc/self/mounts").read_text()
    except OSError:
        return None

    best_fstype = None
    best_depth = -1
    resolved = path.resolve()
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        # /proc/mounts octal-escapes spaces in mount points
        mount_point = Path(fields[1].replace("\\040", " "))
        if mount_point != resolved and mount_point not in resolved.parents:
            continue
        depth = len(mount_point.parts)
        if depth > best_depth:
            best_depth = depth
            best_fstype = fields[2]
    return best_fstype


def prepare_sqlite_path(db_url: str) -> None:
    """Prepare the filesystem for a SQLite database URL.

    Creates missing parent directories for the database file and warns when
    the file lives on a network filesystem, where SQLite locking and WAL mode
    are unreliable and cause 'database is locked' errors under concurrency.

    No-op for non-SQLite URLs and in-memory databases.
    """
    parsed_url = sqlalchemy_url.make_url(db_url)
    if parsed_url.get_backend_name() != "sqlite":
        return
    database = parsed_url.database
    if not database or database == ":memory:" or parsed_url.query.get("mode") == "memory":
        return

    db_path = Path(database).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    fstype = _filesystem_type(db_path.parent)
    if fstype and (fstype in _NETWORK_FS_TYPES or fstype.startswith("fuse")):
        logger.warning(
            f"SQLite database {db_path} is on a '{fstype}' filesystem. SQLite locking and WAL "
            "mode are unreliable on network/FUSE filesystems and can cause 'database is locked' "
            "errors under concurrency. Point --database-url (or SKYRL_DATABASE_URL) at a "
            "node-local path (e.g. under /tmp), or use Postgres."
        )


def get_async_database_url(db_url: str) -> str:
    """Get the async database URL.

    Args:
        db_url: Optional database URL to use.

    Returns:
        Async database URL string for SQLAlchemy.

    Raises:
        ValueError: If the database scheme is not supported.
    """
    parsed_url = sqlalchemy_url.make_url(db_url)

    match parsed_url.get_backend_name():
        case "sqlite":
            async_url = parsed_url.set(drivername="sqlite+aiosqlite")
        case "postgresql":
            async_url = parsed_url.set(drivername="postgresql+asyncpg")
        case _ if "+" in parsed_url.drivername:
            # Already has an async driver specified, keep it
            async_url = parsed_url
        case backend_name:
            raise ValueError(f"Unsupported database scheme: {backend_name}")

    return async_url.render_as_string(hide_password=False)


class RequestStatus(str, Enum):
    """Status of a request."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class CheckpointStatus(str, Enum):
    """Status of a checkpoint."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


# SQLModel table definitions
class ModelDB(SQLModel, table=True):
    __tablename__ = "models"

    model_id: str = Field(primary_key=True)
    base_model: str
    lora_config: dict[str, object] = Field(sa_type=JSON)
    status: str = Field(index=True)
    request_id: int
    session_id: str = Field(foreign_key="sessions.session_id", index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))


class FutureDB(SQLModel, table=True):
    __tablename__ = "futures"
    __table_args__ = (
        # Covering index for the engine's scheduling scan, which filters on
        # status, reads in request_id order, and needs only these four columns.
        # Because every column it reads is in the index, the scan never visits
        # the table rows themselves -- important here, since those rows carry
        # request_data payloads that run to hundreds of KB each.
        #
        # This also supersedes a plain index on status, status being its leading
        # column, so no separate one is declared.
        Index("ix_futures_pending_scan", "status", "request_id", "request_type", "model_id"),
    )

    request_id: int | None = Field(default=None, primary_key=True, sa_column_kwargs={"autoincrement": True})
    request_type: types.RequestType
    model_id: str | None = Field(default=None, index=True)
    request_data: dict = Field(sa_type=JSON)  # this is of type types.{request_type}Input
    result_data: dict | None = Field(default=None, sa_type=JSON)  # this is of type types.{request_type}Output
    status: RequestStatus = Field(default=RequestStatus.PENDING)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))
    completed_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))


class CheckpointDB(SQLModel, table=True):
    __tablename__ = "checkpoints"

    model_id: str = Field(foreign_key="models.model_id", primary_key=True)
    checkpoint_id: str = Field(primary_key=True)
    checkpoint_type: types.CheckpointType = Field(primary_key=True)
    status: CheckpointStatus
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))
    completed_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))
    error_message: str | None = None


class SessionDB(SQLModel, table=True):
    __tablename__ = "sessions"

    session_id: str = Field(primary_key=True)
    tags: list[str] = Field(default_factory=list, sa_type=JSON)
    user_metadata: dict = Field(default_factory=dict, sa_type=JSON)
    sdk_version: str
    status: str = Field(default="active", index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))
    last_heartbeat_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True), index=True)
    heartbeat_count: int = 0


class SamplingSessionDB(SQLModel, table=True):
    __tablename__ = "sampling_sessions"

    sampling_session_id: str = Field(primary_key=True)
    session_id: str = Field(foreign_key="sessions.session_id", index=True)
    sampling_session_seq_id: int
    base_model: str | None = None
    model_path: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))


class EngineStateDB(SQLModel, table=True):
    """Engine→API handoff for the inference engine the backend stands up.

    Singleton row (``singleton_id=1``). Written by the backend when a new
    inference client is built (or torn down) and read by the API's
    forwarding client to resolve the vLLM proxy URL.
    """

    __tablename__ = "engine_state"

    singleton_id: int = Field(default=1, primary_key=True)

    # Proxy URL of the engine-managed vLLM. None when no vLLM has been
    # stood up yet (no create_model, FFT path, or last delete tore down).
    inference_proxy_url: str | None = None

    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))
