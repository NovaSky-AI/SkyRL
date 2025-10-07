from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Literal, Any, AsyncGenerator
from uuid import uuid4
from contextlib import asynccontextmanager
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
import asyncio
import subprocess
import logging

from tx.tinker.models import ModelDB, FutureDB, DB_PATH, RequestType, RequestStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""

    app.state.db_engine = create_async_engine(f"sqlite+aiosqlite:///{DB_PATH}", echo=False)

    async with app.state.db_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    # Start background engine with default base model
    # TODO: Make this configurable via environment variable or API parameter
    base_model = "Qwen/Qwen3-0.6B"
    background_engine = subprocess.Popen(
        ["uv", "run", "--extra", "tinker", "-m", "tx.tinker.engine", "--base-model", base_model]
    )
    logger.info(f"Started background engine with PID {background_engine.pid} for base model {base_model}")

    yield

    logger.info(f"Stopping background engine (PID {background_engine.pid})")
    background_engine.terminate()
    try:
        background_engine.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning(f"Background engine (PID {background_engine.pid}) did not terminate gracefully, killing")
        background_engine.kill()
        background_engine.wait()
    logger.info("Background engine stopped")


app = FastAPI(title="Tinker API Mock", version="0.0.1", lifespan=lifespan)


async def get_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get a database session."""
    async with AsyncSession(request.app.state.db_engine) as session:
        yield session


async def create_future(
    session: AsyncSession,
    request_type: RequestType,
    model_id: str | None,
    request_data: dict,
) -> int:
    """Create a FutureDB entry and return its auto-generated request_id."""
    future_db = FutureDB(
        request_type=request_type,
        model_id=model_id,
        request_data=request_data,
        status=RequestStatus.PENDING
    )
    session.add(future_db)
    await session.flush()  # Flush to generate auto-increment request_id
    assert future_db.request_id
    return future_db.request_id


class LoRAConfig(BaseModel):
    rank: int
    lora_alpha: int = 16
    target_modules: list[str] | None = None
    lora_dropout: float = 0.05


class CreateModelRequest(BaseModel):
    base_model: str
    lora_config: LoRAConfig | None = None
    type: str | None = None


class CreateModelResponse(BaseModel):
    model_id: str
    base_model: str
    lora_config: LoRAConfig | None = None
    status: str = "created"
    request_id: str


class ModelData(BaseModel):
    base_model: str
    lora_config: LoRAConfig | None = None
    model_name: str | None = None


class ModelInfoResponse(BaseModel):
    model_id: str
    status: str
    model_data: ModelData


class ForwardBackwardInput(BaseModel):
    model_id: str
    forward_backward_input: dict[str, Any]


class AdamParams(BaseModel):
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


class OptimStepRequest(BaseModel):
    model_id: str
    adam_params: AdamParams
    type: str | None = None


class SaveWeightsForSamplerRequest(BaseModel):
    model_id: str
    path: str | None = None
    type: str | None = None


class SaveWeightsForSamplerResponse(BaseModel):
    path: str
    type: str | None = None


class FutureResponse(BaseModel):
    future_id: str
    status: str = "pending"
    request_id: str | None = None


class TelemetryEvent(BaseModel):
    event: str
    event_id: str
    event_session_index: int
    severity: str
    timestamp: str
    properties: dict[str, Any] | None = None


class TelemetryRequest(BaseModel):
    events: list[TelemetryEvent]
    platform: str
    sdk_version: str
    session_id: str


class TelemetryResponse(BaseModel):
    status: Literal["accepted"] = "accepted"


class SupportedModel(BaseModel):
    model_name: str | None = None


class GetServerCapabilitiesResponse(BaseModel):
    supported_models: list[SupportedModel]


@app.post("/api/v1/create_model", response_model=CreateModelResponse)
async def create_model(request: CreateModelRequest, session: AsyncSession = Depends(get_session)):
    """Create a new model, optionally with a LoRA adapter."""
    model_id = f"model_{uuid4().hex[:8]}"

    request_id = await create_future(
        session=session,
        request_type=RequestType.CREATE_MODEL,
        model_id=model_id,
        request_data=request.model_dump()
    )

    model_db = ModelDB(
        model_id=model_id,
        base_model=request.base_model,
        lora_config=request.lora_config.model_dump() if request.lora_config else None,
        status="created",
        request_id=request_id
    )
    session.add(model_db)

    await session.commit()

    return CreateModelResponse(
        model_id=model_id,
        base_model=request.base_model,
        lora_config=request.lora_config,
        status="created",
        request_id=str(request_id)
    )


class GetInfoRequest(BaseModel):
    model_id: str
    type: str | None = None


@app.post("/api/v1/get_info", response_model=ModelInfoResponse)
async def get_model_info(request: GetInfoRequest, session: AsyncSession = Depends(get_session)):
    """Retrieve information about the current model."""
    statement = select(ModelDB).where(ModelDB.model_id == request.model_id)
    result = await session.exec(statement)
    model = result.first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    model_data = ModelData(
        base_model=model.base_model,
        lora_config=LoRAConfig(**model.lora_config),  # ty: ignore
        model_name=model.base_model
    )

    return ModelInfoResponse(
        model_id=model.model_id,
        status=model.status,
        model_data=model_data
    )


@app.post("/api/v1/forward_backward", response_model=FutureResponse)
async def forward_backward(request: ForwardBackwardInput, session: AsyncSession = Depends(get_session)):
    """Compute and accumulate gradients."""
    statement = select(ModelDB).where(ModelDB.model_id == request.model_id)
    result = await session.exec(statement)
    model = result.first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    request_id = await create_future(
        session=session,
        request_type=RequestType.FORWARD_BACKWARD,
        model_id=request.model_id,
        request_data=request.model_dump()
    )

    await session.commit()

    return FutureResponse(future_id=str(request_id), status="pending", request_id=str(request_id))


@app.post("/api/v1/optim_step", response_model=FutureResponse)
async def optim_step(request: OptimStepRequest, session: AsyncSession = Depends(get_session)):
    """Update model using accumulated gradients."""
    statement = select(ModelDB).where(ModelDB.model_id == request.model_id)
    result = await session.exec(statement)
    model = result.first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    request_id = await create_future(
        session=session,
        request_type=RequestType.OPTIM_STEP,
        model_id=request.model_id,
        request_data=request.model_dump()
    )

    await session.commit()

    return FutureResponse(future_id=str(request_id), status="pending", request_id=str(request_id))


@app.post("/api/v1/save_weights_for_sampler", response_model=FutureResponse)
async def save_weights_for_sampler(request: SaveWeightsForSamplerRequest, session: AsyncSession = Depends(get_session)):
    """Saves weights in a format compatible with sampling/inference servers."""
    statement = select(ModelDB).where(ModelDB.model_id == request.model_id)
    result = await session.exec(statement)
    model = result.first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    request_id = await create_future(
        session=session,
        request_type=RequestType.SAVE_WEIGHTS_FOR_SAMPLER,
        model_id=request.model_id,
        request_data=request.model_dump()
    )

    await session.commit()

    return FutureResponse(future_id=str(request_id), status="pending", request_id=str(request_id))


@app.get("/api/v1/get_server_capabilities", response_model=GetServerCapabilitiesResponse)
async def get_server_capabilities():
    """Retrieve information about supported models and server capabilities."""
    supported_models = [
        SupportedModel(model_name="Qwen/Qwen3-0.6B"),
    ]
    return GetServerCapabilitiesResponse(supported_models=supported_models)


class RetrieveFutureRequest(BaseModel):
    request_id: str


@app.post("/api/v1/retrieve_future")
async def retrieve_future(request: RetrieveFutureRequest, req: Request):
    """Retrieve the result of an async operation, waiting until it's available."""
    timeout = 300  # 5 minutes
    poll_interval = 0.1  # 100ms

    for _ in range(int(timeout / poll_interval)):
        async with AsyncSession(req.app.state.db_engine) as session:
            statement = select(FutureDB).where(FutureDB.request_id == int(request.request_id))
            result = await session.exec(statement)
            future = result.first()

            if not future:
                raise HTTPException(status_code=404, detail="Future not found")

            if future.status == RequestStatus.COMPLETED:
                return future.result_data

            if future.status == RequestStatus.FAILED:
                # Return 400 for handled errors (validation, etc.), 500 for unexpected failures
                if future.result_data and "error" in future.result_data:
                    raise HTTPException(status_code=400, detail=future.result_data["error"])
                else:
                    raise HTTPException(status_code=500, detail="Unknown error")

        await asyncio.sleep(poll_interval)

    raise HTTPException(status_code=408, detail="Timeout waiting for result")


@app.post("/api/v1/telemetry", response_model=TelemetryResponse)
async def send_telemetry(request: TelemetryRequest):
    """Accept batches of SDK telemetry events for analytics and diagnostics."""
    # Just acknowledge receipt without doing anything
    return TelemetryResponse(status="accepted")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Tinker API Mock",
        "version": "0.0.1",
        "endpoints": {
            "models": ["/api/v1/create_model", "/api/v1/get_info"],
            "training": ["/api/v1/forward_backward", "/api/v1/optim_step"],
            "futures": ["/api/v1/retrieve_future"],
            "service": ["/api/v1/get_server_capabilities"],
            "telemetry": ["/api/v1/telemetry"]
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
