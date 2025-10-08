"""Database models for the Tinker API."""
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from sqlmodel import SQLModel, Field, JSON

# SQLite database path
DB_PATH = Path(__file__).parent / "tinker.db"


class RequestType(str, Enum):
    """Types of requests that can be processed."""
    CREATE_MODEL = "create_model"
    FORWARD_BACKWARD = "forward_backward"
    OPTIM_STEP = "optim_step"
    SAVE_WEIGHTS_FOR_SAMPLER = "save_weights_for_sampler"
    DOWNLOAD_CHECKPOINT = "download_checkpoint"


class RequestStatus(str, Enum):
    """Status of a request."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


# SQLModel table definitions
class ModelDB(SQLModel, table=True):
    __tablename__ = "models"

    model_id: str = Field(primary_key=True)
    base_model: str
    lora_config: dict | None = Field(default=None, sa_type=JSON)
    status: str
    request_id: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FutureDB(SQLModel, table=True):
    __tablename__ = "futures"

    request_id: int | None = Field(default=None, primary_key=True, sa_column_kwargs={"autoincrement": True})
    request_type: RequestType
    model_id: str | None = Field(default=None, index=True)
    request_data: dict = Field(sa_type=JSON)
    result_data: dict | None = Field(default=None, sa_type=JSON)
    status: RequestStatus = Field(default=RequestStatus.PENDING, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
