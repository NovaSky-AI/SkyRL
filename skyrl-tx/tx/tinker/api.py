import fastapi
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, RedirectResponse
from pydantic import BaseModel
from typing import Literal, Any, AsyncGenerator
from uuid import uuid4
from contextlib import asynccontextmanager
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from urllib.parse import urlparse
from datetime import datetime, timedelta
import asyncio
import logging
import subprocess

from tx.tinker import types
from tx.tinker.config import EngineConfig, add_model, config_to_argv
from tx.tinker.db_models import ModelDB, FutureDB, DB_PATH, RequestStatus
from tx.utils.storage import download_file


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""

    app.state.db_engine = create_async_engine(f"sqlite+aiosqlite:///{DB_PATH}", echo=False)

    async with app.state.db_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    # Build subprocess command with engine config parameters
    cmd = ["uv", "run", "--extra", "tinker", "-m", "tx.tinker.engine"]
    cmd.extend(config_to_argv(app.state.engine_config))

    background_engine = subprocess.Popen(cmd)
    logger.info(f"Started background engine with PID {background_engine.pid}: {' '.join(cmd)}")

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
    request_type: types.RequestType,
    model_id: str | None,
    request_data: BaseModel,
) -> int:
    """Create a FutureDB entry and return its auto-generated request_id."""
    future_db = FutureDB(
        request_type=request_type,
        model_id=model_id,
        request_data=request_data.model_dump(),
        status=RequestStatus.PENDING,
    )
    session.add(future_db)
    await session.flush()  # Flush to generate auto-increment request_id
    assert future_db.request_id
    return future_db.request_id


class LoRAConfig(BaseModel):
    rank: int


class CreateModelRequest(BaseModel):
    base_model: str
    lora_config: LoRAConfig


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


class OptimStepRequest(BaseModel):
    model_id: str
    adam_params: AdamParams


class SaveWeightsForSamplerRequest(BaseModel):
    model_id: str
    path: str


class SampleRequest(BaseModel):
    # For now we require model_path, in the official SDK there can actually be
    # either model_path or base_model, the latter to sample from the base model:
    # https://github.com/thinking-machines-lab/tinker/blob/main/src/tinker/types/sample_request.py
    model_path: str
    prompt: dict[str, Any]
    sampling_params: dict[str, Any]
    num_samples: int
    prompt_logprobs: bool = False


class SaveWeightsRequest(BaseModel):
    model_id: str
    path: str
    type: Literal["save_weights"] | None = None


class LoadWeightsRequest(BaseModel):
    model_id: str
    path: str
    type: Literal["load_weights"] | None = None


class FutureResponse(BaseModel):
    future_id: str
    status: str = "pending"
    request_id: str


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
    model_name: str


class GetServerCapabilitiesResponse(BaseModel):
    supported_models: list[SupportedModel]


@app.post("/api/v1/create_model", response_model=CreateModelResponse)
async def create_model(request: CreateModelRequest, session: AsyncSession = Depends(get_session)):
    """Create a new model, optionally with a LoRA adapter."""
    model_id = f"model_{uuid4().hex[:8]}"

    # alpha = 32 seems to be the tinker default (see https://thinkingmachines.ai/blog/lora/)
    lora_config = types.LoraConfig(rank=request.lora_config.rank, alpha=32.0)
    request_id = await create_future(
        session=session,
        request_type=types.RequestType.CREATE_MODEL,
        model_id=model_id,
        request_data=types.CreateModelInput(lora_config=lora_config),
    )

    model_db = ModelDB(
        model_id=model_id,
        base_model=request.base_model,
        lora_config=lora_config.model_dump(),
        status="created",
        request_id=request_id,
    )
    session.add(model_db)

    await session.commit()

    return CreateModelResponse(
        model_id=model_id,
        base_model=request.base_model,
        lora_config=request.lora_config,
        status="created",
        request_id=str(request_id),
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

    lora_config = types.LoraConfig.model_validate(model.lora_config)
    model_data = ModelData(
        base_model=model.base_model, lora_config=LoRAConfig(rank=lora_config.rank), model_name=model.base_model
    )

    return ModelInfoResponse(model_id=model.model_id, status=model.status, model_data=model_data)


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
        request_type=types.RequestType.FORWARD_BACKWARD,
        model_id=request.model_id,
        request_data=types.ForwardBackwardInput(forward_backward_input=request.forward_backward_input),
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
        request_type=types.RequestType.OPTIM_STEP,
        model_id=request.model_id,
        request_data=types.OptimStepInput(adam_params=types.AdamParams(lr=request.adam_params.lr)),
    )

    await session.commit()

    return FutureResponse(future_id=str(request_id), status="pending", request_id=str(request_id))


@app.post("/api/v1/load_weights", response_model=FutureResponse)
async def load_weights(request: LoadWeightsRequest, session: AsyncSession = Depends(get_session)):
    """Loads weights and training state."""
    statement = select(ModelDB).where(ModelDB.model_id == request.model_id)
    result = await session.exec(statement)
    model = result.first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    request_id = await create_future(
        session=session,
        request_type=types.RequestType.LOAD_WEIGHTS,
        model_id=request.model_id,
        request_data=types.LoadWeightsInput(path=request.path),
    )

    await session.commit()

    return FutureResponse(future_id=str(request_id), status="pending", request_id=str(request_id))


@app.post("/api/v1/save_weights", response_model=FutureResponse)
async def save_weights(request: SaveWeightsRequest, session: AsyncSession = Depends(get_session)):
    """Saves weights and training state."""
    statement = select(ModelDB).where(ModelDB.model_id == request.model_id)
    result = await session.exec(statement)
    model = result.first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    request_id = await create_future(
        session=session,
        request_type=types.RequestType.SAVE_WEIGHTS,
        model_id=request.model_id,
        request_data=types.SaveWeightsInput(path=request.path),
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
        request_type=types.RequestType.SAVE_WEIGHTS_FOR_SAMPLER,
        model_id=request.model_id,
        request_data=types.SaveWeightsForSamplerInput(path=request.path),
    )

    await session.commit()

    return FutureResponse(future_id=str(request_id), status="pending", request_id=str(request_id))


@app.post("/api/v1/asample", response_model=FutureResponse)
async def asample(request: SampleRequest, session: AsyncSession = Depends(get_session)):
    """Generates samples from the model (async version)."""
    # Extract model_id and checkpoint_id from model_path (format: tinker://model_id/checkpoint_name)
    parsed = urlparse(request.model_path)
    if parsed.scheme != "tinker" or not (model_id := parsed.netloc) or not (checkpoint_id := parsed.path.lstrip("/")):
        raise HTTPException(status_code=400, detail="model_path must be in format tinker://model_id/checkpoint_id")

    statement = select(ModelDB).where(ModelDB.model_id == model_id)
    result = await session.exec(statement)
    model = result.first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    request_id = await create_future(
        session=session,
        request_type=types.RequestType.SAMPLE,
        model_id=model_id,
        request_data=types.SampleInput(
            prompt=request.prompt,
            sampling_params=request.sampling_params,
            num_samples=request.num_samples,
            checkpoint_id=checkpoint_id,
        ),
    )

    await session.commit()

    return FutureResponse(future_id=str(request_id), status="pending", request_id=str(request_id))


@app.get("/api/v1/get_server_capabilities", response_model=GetServerCapabilitiesResponse)
async def get_server_capabilities(request: Request):
    """Retrieve information about supported models and server capabilities."""
    supported_models = [
        SupportedModel(model_name=request.app.state.engine_config.base_model),
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


async def validate_checkpoint(request: Request, unique_id: str, checkpoint_id: str, session: AsyncSession):
    """Validate that a model and checkpoint exist, returning the checkpoint path."""
    statement = select(ModelDB).where(ModelDB.model_id == unique_id)
    result = await session.exec(statement)
    model = result.first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    checkpoint_path = request.app.state.engine_config.checkpoints_base / unique_id / f"{checkpoint_id}.tar.gz"
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {unique_id}/{checkpoint_id}")

    return checkpoint_path


@app.get("/api/v1/training_runs/{unique_id}/checkpoints/{checkpoint_id}/archive")
async def get_checkpoint_archive_url(
    request: Request,
    unique_id: str = fastapi.Path(..., pattern=r"^[a-zA-Z0-9_-]+$", max_length=255),
    checkpoint_id: str = fastapi.Path(..., pattern=r"^[a-zA-Z0-9_-]+$", max_length=255),
    session: AsyncSession = Depends(get_session),
):
    """Return a 302 redirect to the download URL (SDK expects this pattern)"""
    await validate_checkpoint(request, unique_id, checkpoint_id, session)

    # Generate URL to the download endpoint and return 302 redirect
    download_url = str(request.url_for("download_checkpoint_archive", unique_id=unique_id, checkpoint_id=checkpoint_id))
    expires = datetime.utcnow() + timedelta(minutes=120)

    response = RedirectResponse(url=download_url, status_code=302)
    response.headers["Expires"] = expires.strftime("%a, %d %b %Y %H:%M:%S GMT")
    return response


@app.get("/api/v1/training_runs/{unique_id}/checkpoints/{checkpoint_id}/download")
async def download_checkpoint_archive(
    request: Request,
    unique_id: str = fastapi.Path(..., pattern=r"^[a-zA-Z0-9_-]+$", max_length=255),
    checkpoint_id: str = fastapi.Path(..., pattern=r"^[a-zA-Z0-9_-]+$", max_length=255),
    session: AsyncSession = Depends(get_session),
):
    """Actually download the checkpoint archive bytes"""
    checkpoint_path = await validate_checkpoint(request, unique_id, checkpoint_id, session)

    file_buffer = await asyncio.to_thread(download_file, checkpoint_path)

    filename = f"{unique_id}_{checkpoint_id}.tar.gz"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Length": str(file_buffer.getbuffer().nbytes),
    }

    return StreamingResponse(file_buffer, media_type="application/octet-stream", headers=headers)


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
            "telemetry": ["/api/v1/telemetry"],
            "download": [
                "/api/v1/training_runs/{unique_id}/checkpoints/{checkpoint_id}/archive",
                "/api/v1/training_runs/{unique_id}/checkpoints/{checkpoint_id}/download",
            ],
        },
    }


if __name__ == "__main__":
    import argparse
    import uvicorn

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="SkyRL tx tinker API server")
    add_model(parser, EngineConfig)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    # Create EngineConfig from parsed arguments (only EngineConfig fields)
    engine_config = EngineConfig.model_validate({k: v for k, v in vars(args).items() if k in EngineConfig.model_fields})

    # Store config in app.state so lifespan can access it
    app.state.engine_config = engine_config

    uvicorn.run(app, host=args.host, port=args.port)
