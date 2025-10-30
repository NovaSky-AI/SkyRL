"""Database models for the Tinker API."""

import os
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from sqlmodel import SQLModel, Field, JSON
from sqlalchemy import DateTime
from sqlalchemy.engine import url as sqlalchemy_url

from tx.tinker import types


def get_database_url(db_url: str | None = None) -> str:
    """Get the database URL from environment variable or parameter.

    Args:
        db_url: Optional database URL to use. If None, uses environment variable
                or defaults to SQLite.

    Returns:
        Database URL string for SQLAlchemy.

    Examples:
        SQLite: sqlite:///path/to/tinker.db
        PostgreSQL: postgresql://user:password@localhost:5432/tinker
        PostgreSQL (async): postgresql+asyncpg://user:password@localhost:5432/tinker
    """
    if db_url:
        return db_url

    return os.environ.get("TX_DATABASE_URL", f'sqlite:///{Path(__file__).parent / "tinker.db"}')


def get_async_database_url(db_url: str | None = None) -> str:
    """Get the async database URL.

    Args:
        db_url: Optional database URL to use.

    Returns:
        Async database URL string for SQLAlchemy.

    Raises:
        ValueError: If the database scheme is not supported.
    """
    parsed_url = sqlalchemy_url.make_url(get_database_url(db_url))

    match parsed_url.get_backend_name():
        case "sqlite":
            return parsed_url.set(drivername="sqlite+aiosqlite").render_as_string(hide_password=False)
        case "postgresql":
            return parsed_url.set(drivername="postgresql+asyncpg").render_as_string(hide_password=False)
        case _ if "+" in parsed_url.drivername:
            # Already has an async driver specified, keep it
            return parsed_url.render_as_string(hide_password=False)
        case backend_name:
            raise ValueError(f"Unsupported database scheme: {backend_name}")


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
    lora_config: types.LoraConfig = Field(sa_type=JSON)
    status: str
    request_id: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))


class FutureDB(SQLModel, table=True):
    __tablename__ = "futures"

    request_id: int | None = Field(default=None, primary_key=True, sa_column_kwargs={"autoincrement": True})
    request_type: types.RequestType
    model_id: str | None = Field(default=None, index=True)
    request_data: dict = Field(sa_type=JSON)  # this is of type types.{request_type}Input
    result_data: dict | None = Field(default=None, sa_type=JSON)  # this is of type types.{request_type}Output
    status: RequestStatus = Field(default=RequestStatus.PENDING, index=True)
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
