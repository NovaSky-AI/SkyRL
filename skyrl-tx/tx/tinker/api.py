import fastapi
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, RedirectResponse
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Any, AsyncGenerator, Sequence
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from contextlib import asynccontextmanager
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.exc import IntegrityError, TimeoutError as SATimeoutError
import asyncio
import subprocess
import random
import time

from tx.tinker import types
from tx.tinker.config import EngineConfig, add_model, config_to_argv
from tx.tinker.db_models import (
    CheckpointDB,
    ModelDB,
    FutureDB,
    RequestStatus,
    CheckpointStatus,
    SessionDB,
    SamplingSessionDB,
    get_async_database_url,
)
from tx.tinker.extra import ExternalInferenceClient
from tx.utils.storage import download_file
from tx.utils.log import logger

# Validation patterns for train_run_ids, model_ids and checkpoint_ids
ID_PATTERN = r"^[a-zA-Z0-9_-]+$"
ID_MAX_LENGTH = 255


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""

    db_url = get_async_database_url(app.state.engine_config.database_url)
    app.state.db_engine = create_async_engine(db_url, echo=False)

    async with app.state.db_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    # Setup external inference client if configured
    if app.state.engine_config.external_inference_url:
        app.state.external_inference_client = ExternalInferenceClient(app.state.engine_config, app.state.db_engine)
        logger.info(f"External engine configured: {app.state.engine_config.external_inference_url}")
    else:
        app.state.external_inference_client = None
        logger.info("Using internal engine for inference")

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


async def get_model(session: AsyncSession, model_id: str) -> ModelDB:
    """Fetch a model by ID, raising 404 if not found."""
    statement = select(ModelDB).where(ModelDB.model_id == model_id)
    result = await session.exec(statement)
    model = result.first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


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


async def create_checkpoint(
    session: AsyncSession,
    model_id: str,
    checkpoint_id: str,
    checkpoint_type: types.CheckpointType,
):
    """Create a pending CheckpointDB entry, relying on database constraints for validation."""
    checkpoint_db = CheckpointDB(
        model_id=model_id,
        checkpoint_id=checkpoint_id,
        checkpoint_type=checkpoint_type,
        status=CheckpointStatus.PENDING,
    )
    session.add(checkpoint_db)

    try:
        await session.flush()
    except IntegrityError:
        # Determine which constraint failed by checking if the model exists
        statement = select(ModelDB).where(ModelDB.model_id == model_id)
        result = await session.exec(statement)

        if not result.first():
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        else:
            raise HTTPException(
                status_code=409, detail=f"Checkpoint '{checkpoint_id}' already exists for model '{model_id}'"
            )


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


class Checkpoint(BaseModel):
    checkpoint_id: str
    checkpoint_type: Literal["training", "sampler"]
    time: datetime
    tinker_path: str


class TrainingRun(BaseModel):
    training_run_id: str
    base_model: str
    model_owner: str = "default"
    is_lora: bool = True
    corrupted: bool = False
    lora_rank: int | None = None
    last_request_time: datetime
    last_checkpoint: Checkpoint | None = None
    last_sampler_checkpoint: Checkpoint | None = None
    user_metadata: dict[str, str] | None = None


class ModelInputChunk(BaseModel):
    tokens: list[int]

    def to_types(self) -> types.ModelInputChunk:
        return types.ModelInputChunk(tokens=self.tokens)


class ModelInput(BaseModel):
    chunks: list[ModelInputChunk]

    def to_types(self) -> types.ModelInput:
        return types.ModelInput(chunks=[chunk.to_types() for chunk in self.chunks])


class TensorData(BaseModel):
    data: list[int] | list[float]

    def to_types(self) -> types.TensorData:
        return types.TensorData(data=self.data)


class Datum(BaseModel):
    loss_fn_inputs: dict[str, TensorData]
    model_input: ModelInput

    def to_types(self) -> types.Datum:
        inp = self.loss_fn_inputs

        if "weights" not in inp:
            weights = types.TensorData(data=[1.0] * len(inp["target_tokens"].data))
        else:
            weights = inp["weights"].to_types()

        return types.Datum(
            loss_fn_inputs=types.LossFnInputs(
                target_tokens=inp["target_tokens"].to_types(),
                weights=weights,
                advantages=inp["advantages"].to_types() if "advantages" in inp else types.TensorData(data=[]),
                logprobs=inp["logprobs"].to_types() if "logprobs" in inp else types.TensorData(data=[]),
            ),
            model_input=self.model_input.to_types(),
        )


class ForwardBackwardInput(BaseModel):
    data: list[Datum]
    loss_fn: Literal["cross_entropy", "importance_sampling", "ppo"]

    def to_types(self) -> types.ForwardBackwardInput:
        return types.ForwardBackwardInput(data=[datum.to_types() for datum in self.data], loss_fn=self.loss_fn)


class ForwardBackwardRequest(BaseModel):
    model_id: str
    forward_backward_input: ForwardBackwardInput


class ForwardRequest(BaseModel):
    model_id: str
    forward_input: ForwardBackwardInput


class AdamParams(BaseModel):
    learning_rate: float = Field(default=1e-4, ge=0.0)
    beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    beta2: float = Field(default=0.95, ge=0.0, lt=1.0)
    eps: float = Field(default=1e-12, gt=0.0)

    def to_types(self) -> types.AdamParams:
        return types.AdamParams(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2, eps=self.eps)


class OptimStepRequest(BaseModel):
    model_id: str
    adam_params: AdamParams


class SaveWeightsForSamplerRequest(BaseModel):
    model_id: str
    path: str = Field(..., pattern=ID_PATTERN, max_length=ID_MAX_LENGTH)


class SamplingParams(BaseModel):
    max_tokens: int | None = None
    seed: int | None = None
    stop: Sequence[int] | None = None
    temperature: float = 1
    top_k: int = -1
    top_p: float = 1

    def to_types(self) -> types.SamplingParams:
        if self.max_tokens is None:
            raise HTTPException(status_code=400, detail="max_tokens is currently required")

        if self.top_k != -1:
            raise HTTPException(status_code=501, detail="'top_k' parameter is not yet implemented")
        if self.top_p != 1.0:
            raise HTTPException(status_code=501, detail="'top_p' parameter is not yet implemented")

        # Generate a random seed if not provided
        seed = self.seed if self.seed is not None else random.randint(0, 2**31 - 1)

        return types.SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=seed,
            stop=self.stop,
        )


class SampleRequest(BaseModel):
    num_samples: int = 1
    prompt: ModelInput
    sampling_params: SamplingParams
    base_model: str | None = None
    model_path: str | None = None
    sampling_session_id: str | None = None
    seq_id: int | None = None
    prompt_logprobs: bool | None = None
    topk_prompt_logprobs: int = 0
    type: Literal["sample"] = "sample"

    @model_validator(mode="after")
    def validate_model_source(self):
        """Valid if:
        - sampling_session_id is provided AND seq_id is provided
        - OR exactly one of base_model or model_path is provided
        """
        if self.sampling_session_id is not None:
            if self.seq_id is None:
                raise ValueError("'seq_id' must be provided when 'sampling_session_id' is used")
            return self
        if (self.base_model is None) == (self.model_path is None):
            raise ValueError(
                "When 'sampling_session_id' is not provided, exactly one of 'base_model' or 'model_path' must be provided"
            )
        return self


class SaveWeightsRequest(BaseModel):
    model_id: str
    path: str = Field(..., pattern=ID_PATTERN, max_length=ID_MAX_LENGTH)
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


class HealthResponse(BaseModel):
    status: Literal["ok"]


class CreateSessionRequest(BaseModel):
    tags: list[str]
    user_metadata: dict[str, Any] | None = None
    sdk_version: str
    type: Literal["create_session"] = "create_session"


class CreateSessionResponse(BaseModel):
    type: Literal["create_session"] = "create_session"
    info_message: str | None = None
    warning_message: str | None = None
    error_message: str | None = None
    session_id: str


class SessionHeartbeatRequest(BaseModel):
    session_id: str
    type: Literal["session_heartbeat"] = "session_heartbeat"


class SessionHeartbeatResponse(BaseModel):
    type: Literal["session_heartbeat"] = "session_heartbeat"


class CreateSamplingSessionRequest(BaseModel):
    session_id: str
    sampling_session_seq_id: int
    base_model: str | None = None
    model_path: str | None = None
    type: Literal["create_sampling_session"] = "create_sampling_session"


class CreateSamplingSessionResponse(BaseModel):
    type: Literal["create_sampling_session"] = "create_sampling_session"
    sampling_session_id: str


class SupportedModel(BaseModel):
    model_name: str


class GetServerCapabilitiesResponse(BaseModel):
    supported_models: list[SupportedModel]


class ListCheckpointsResponse(BaseModel):
    checkpoints: list[Checkpoint]


class WeightsInfoRequest(BaseModel):
    tinker_path: str


class WeightsInfoResponse(BaseModel):
    """Minimal information for loading public checkpoints."""

    # from: https://github.com/thinking-machines-lab/tinker/blob/main/src/tinker/types/weights_info_response.py
    base_model: str
    is_lora: bool
    lora_rank: int | None = None


@app.get("/api/v1/healthz", response_model=HealthResponse)
async def healthz():
    """Checks if the API server is ready."""
    return HealthResponse(status="ok")


@app.post("/api/v1/create_session", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest, session: AsyncSession = Depends(get_session)):
    """Create a new session + persist in DB"""
    session_id = f"session_{uuid4().hex[:8]}"
    session_db = SessionDB(
        session_id=session_id,
        tags=request.tags,
        user_metadata=request.user_metadata or {},
        sdk_version=request.sdk_version,
        status="active",
    )
    session.add(session_db)
    await session.commit()
    return CreateSessionResponse(session_id=session_id)


@app.post("/api/v1/session_heartbeat", response_model=SessionHeartbeatResponse)
async def session_heartbeat(request: SessionHeartbeatRequest, session: AsyncSession = Depends(get_session)):
    """Heartbeat for an active session to keep it alive."""
    session_db = await session.get(SessionDB, request.session_id)
    if session_db is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session_db.last_heartbeat_at = datetime.now(timezone.utc)
    session_db.heartbeat_count += 1
    await session.commit()
    return SessionHeartbeatResponse()


@app.post("/api/v1/create_sampling_session", response_model=CreateSamplingSessionResponse)
async def create_sampling_session(request: CreateSamplingSessionRequest, session: AsyncSession = Depends(get_session)):
    """Create a new sampling session within an existing session."""
    session_db = await session.get(SessionDB, request.session_id)
    if session_db is None:
        raise HTTPException(status_code=404, detail="Session not found")
    # Exactly one of base_model or model_path must be provided
    if (request.base_model is None) == (request.model_path is None):
        raise HTTPException(status_code=400, detail="Exactly one of base_model or model_path must be provided")
    sampling_session_id = f"sampling_{uuid4().hex[:8]}"
    sampling_db = SamplingSessionDB(
        sampling_session_id=sampling_session_id,
        session_id=request.session_id,
        sampling_session_seq_id=request.sampling_session_seq_id,
        base_model=request.base_model,
        model_path=request.model_path,
    )
    session.add(sampling_db)
    await session.commit()
    return CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)


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
    model = await get_model(session, request.model_id)

    lora_config = types.LoraConfig.model_validate(model.lora_config)
    model_data = ModelData(
        base_model=model.base_model, lora_config=LoRAConfig(rank=lora_config.rank), model_name=model.base_model
    )

    return ModelInfoResponse(model_id=model.model_id, status=model.status, model_data=model_data)


@app.get("/api/v1/training_runs/{model_id}", response_model=TrainingRun)
async def get_training_run(model_id: str, session: AsyncSession = Depends(get_session)):
    """Get training run for session resumption."""
    model = await get_model(session, model_id)

    lora_config = types.LoraConfig.model_validate(model.lora_config)

    return TrainingRun(
        training_run_id=model.model_id,
        base_model=model.base_model,
        model_owner="default",
        is_lora=True,
        corrupted=False,
        lora_rank=lora_config.rank,
        last_request_time=model.created_at,  # TODO: Once we track modified_at timestamps, update this
        last_checkpoint=None,
        last_sampler_checkpoint=None,
        user_metadata=None,
    )


@app.post("/api/v1/forward_backward", response_model=FutureResponse)
async def forward_backward(request: ForwardBackwardRequest, session: AsyncSession = Depends(get_session)):
    """Compute and accumulate gradients."""
    await get_model(session, request.model_id)

    request_id = await create_future(
        session=session,
        request_type=types.RequestType.FORWARD_BACKWARD,
        model_id=request.model_id,
        request_data=request.forward_backward_input.to_types(),
    )

    await session.commit()

    return FutureResponse(future_id=str(request_id), status="pending", request_id=str(request_id))


@app.post("/api/v1/forward", response_model=FutureResponse)
async def forward(request: ForwardRequest, session: AsyncSession = Depends(get_session)):
    """Forward pass to obtain logprobs without accumulating gradients"""
    await get_model(session, request.model_id)

    request_id = await create_future(
        session=session,
        request_type=types.RequestType.FORWARD,
        model_id=request.model_id,
        request_data=request.forward_input.to_types(),
    )

    await session.commit()

    return FutureResponse(future_id=str(request_id), status="pending", request_id=str(request_id))


@app.post("/api/v1/optim_step", response_model=FutureResponse)
async def optim_step(request: OptimStepRequest, session: AsyncSession = Depends(get_session)):
    """Update model using accumulated gradients."""
    await get_model(session, request.model_id)

    request_id = await create_future(
        session=session,
        request_type=types.RequestType.OPTIM_STEP,
        model_id=request.model_id,
        request_data=types.OptimStepInput(adam_params=request.adam_params.to_types()),
    )

    await session.commit()

    return FutureResponse(future_id=str(request_id), status="pending", request_id=str(request_id))


@app.post("/api/v1/load_weights", response_model=FutureResponse)
async def load_weights(request: LoadWeightsRequest, req: Request, session: AsyncSession = Depends(get_session)):
    """Loads weights and training state."""
    await get_model(session, request.model_id)

    path = types.TinkerPath.parse(request.path)
    if (
        not path
        or path.kind != "weights"
        or not (source_model_id := path.primary_id)
        or not (checkpoint_id := path.secondary_id)
    ):
        raise HTTPException(
            status_code=400, detail="request.path must be in format tinker://source_model_id/weights/checkpoint_id"
        )

    await validate_checkpoint(req, source_model_id, checkpoint_id, types.CheckpointType.TRAINING, session)

    request_id = await create_future(
        session=session,
        request_type=types.RequestType.LOAD_WEIGHTS,
        model_id=request.model_id,
        request_data=types.LoadWeightsInput(source_model_id=source_model_id, checkpoint_id=checkpoint_id),
    )

    await session.commit()

    return FutureResponse(future_id=str(request_id), status="pending", request_id=str(request_id))


@app.post("/api/v1/save_weights", response_model=FutureResponse)
async def save_weights(request: SaveWeightsRequest, session: AsyncSession = Depends(get_session)):
    """Saves weights and training state."""
    # Create pending checkpoint entry (validates model exists)
    await create_checkpoint(
        session=session,
        model_id=request.model_id,
        checkpoint_id=request.path,
        checkpoint_type=types.CheckpointType.TRAINING,
    )

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
    # Create pending checkpoint entry (validates model exists)
    await create_checkpoint(
        session=session,
        model_id=request.model_id,
        checkpoint_id=request.path,
        checkpoint_type=types.CheckpointType.SAMPLER,
    )

    request_id = await create_future(
        session=session,
        request_type=types.RequestType.SAVE_WEIGHTS_FOR_SAMPLER,
        model_id=request.model_id,
        request_data=types.SaveWeightsForSamplerInput(path=request.path),
    )

    await session.commit()

    return FutureResponse(future_id=str(request_id), status="pending", request_id=str(request_id))


async def get_sampling_model(request: SampleRequest, session: AsyncSession) -> (str | None, str | None):
    """Return (base_model, model_path) for a sampling request."""
    # Resolve model/base from sampling_session_id if provided
    if request.sampling_session_id is not None:
        sampling_session = await session.get(SamplingSessionDB, request.sampling_session_id)
        if sampling_session is None:
            raise HTTPException(status_code=404, detail="Sampling session not found")
        return (sampling_session.base_model, sampling_session.model_path)
    return (request.base_model, request.model_path)


@app.post("/api/v1/asample", response_model=FutureResponse)
async def asample(request: SampleRequest, req: Request, session: AsyncSession = Depends(get_session)):
    """Generates samples from the model (async version)."""
    base_model, model_path = await get_sampling_model(request, session)

    if base_model:
        model_id = checkpoint_id = ""
    else:
        assert model_path is not None
        path = types.TinkerPath.parse(model_path)
        if (
            not path
            # Accept either tinker://model_id/checkpoint_id or tinker://model_id/sampler_weights/checkpoint_id
            or path.kind not in ("", "sampler_weights")
            or not (model_id := path.primary_id)
            or not (checkpoint_id := path.secondary_id)
        ):
            raise HTTPException(
                status_code=400,
                detail="model_path must be tinker://model_id/checkpoint_id or tinker://model_id/sampler_weights/checkpoint_id",
            )
        await get_model(session, model_id)
        # Validate that the checkpoint exists and is ready
        await validate_checkpoint(req, model_id, checkpoint_id, types.CheckpointType.SAMPLER, session)

    request_id = await create_future(
        session=session,
        request_type=(
            types.RequestType.EXTERNAL if req.app.state.external_inference_client else types.RequestType.SAMPLE
        ),
        model_id=model_id,
        request_data=types.SampleInput(
            base_model=base_model,
            prompt=request.prompt.to_types(),
            sampling_params=request.sampling_params.to_types(),
            num_samples=request.num_samples,
            checkpoint_id=checkpoint_id,
            prompt_logprobs=request.prompt_logprobs,
        ),
    )

    await session.commit()

    if req.app.state.external_inference_client:
        asyncio.create_task(
            req.app.state.external_inference_client.call_and_store_result(request_id, request, model_id, checkpoint_id)
        )

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
    deadline = time.perf_counter() + timeout

    # Start with 100ms, grow to 1s
    poll = 0.1
    max_poll = 1.0

    while time.perf_counter() < deadline:
        try:
            async with AsyncSession(req.app.state.db_engine) as session:
                # First, only query the status to avoid deserializing JSON data
                statement = select(FutureDB.status).where(FutureDB.request_id == int(request.request_id))
                result = await session.exec(statement)
                status = result.first()

                if not status:
                    raise HTTPException(status_code=404, detail="Future not found")

                # Only fetch full record if status is terminal (completed or failed)
                if status in (RequestStatus.COMPLETED, RequestStatus.FAILED):
                    statement = select(FutureDB).where(FutureDB.request_id == int(request.request_id))
                    result = await session.exec(statement)
                    future = result.first()

                    if future.status == RequestStatus.COMPLETED:
                        return future.result_data

                    if future.status == RequestStatus.FAILED:
                        # Return 400 for handled errors (validation, etc.), 500 for unexpected failures
                        if future.result_data and "error" in future.result_data:
                            raise HTTPException(status_code=400, detail=future.result_data["error"])
                        else:
                            raise HTTPException(status_code=500, detail="Unknown error")
        except SATimeoutError:
            pass

        # Exponential backoff
        await asyncio.sleep(poll)
        poll = min(poll * 1.5, max_poll)

    raise HTTPException(status_code=408, detail="Timeout waiting for result")


@app.post("/api/v1/telemetry", response_model=TelemetryResponse)
async def send_telemetry(request: TelemetryRequest):
    """Accept batches of SDK telemetry events for analytics and diagnostics."""
    # Just acknowledge receipt without doing anything
    return TelemetryResponse(status="accepted")


async def validate_checkpoint(
    request: Request, unique_id: str, checkpoint_id: str, checkpoint_type: types.CheckpointType, session: AsyncSession
):
    """Validate that a model and checkpoint exist in the database, returning the checkpoint path."""
    checkpoint_db = await session.get(CheckpointDB, (unique_id, checkpoint_id, checkpoint_type))

    if not checkpoint_db:
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {unique_id}/{checkpoint_id}")

    if checkpoint_db.status == CheckpointStatus.PENDING:
        raise HTTPException(status_code=425, detail="Checkpoint is still being created")

    if checkpoint_db.status == CheckpointStatus.FAILED:
        raise HTTPException(status_code=500, detail=f"Checkpoint creation failed: {checkpoint_db.error_message}")

    subdir = "sampler_weights" if checkpoint_type == types.CheckpointType.SAMPLER else ""
    return request.app.state.engine_config.checkpoints_base / unique_id / subdir / f"{checkpoint_id}.tar.gz"


@app.get("/api/v1/training_runs/{unique_id}/checkpoints/{checkpoint_id}/archive")
async def get_checkpoint_archive_url(
    request: Request,
    unique_id: str = fastapi.Path(..., pattern=ID_PATTERN, max_length=ID_MAX_LENGTH),
    checkpoint_id: str = fastapi.Path(..., pattern=ID_PATTERN, max_length=ID_MAX_LENGTH),
    session: AsyncSession = Depends(get_session),
):
    """Return a 302 redirect to the download URL (SDK expects this pattern)"""
    await validate_checkpoint(request, unique_id, checkpoint_id, types.CheckpointType.SAMPLER, session)

    # Generate URL to the download endpoint and return 302 redirect
    download_url = str(request.url_for("download_checkpoint_archive", unique_id=unique_id, checkpoint_id=checkpoint_id))
    expires = datetime.utcnow() + timedelta(minutes=120)

    response = RedirectResponse(url=download_url, status_code=302)
    response.headers["Expires"] = expires.strftime("%a, %d %b %Y %H:%M:%S GMT")
    return response


@app.get("/api/v1/training_runs/{unique_id}/checkpoints/{checkpoint_id}/download")
async def download_checkpoint_archive(
    request: Request,
    unique_id: str = fastapi.Path(..., pattern=ID_PATTERN, max_length=ID_MAX_LENGTH),
    checkpoint_id: str = fastapi.Path(..., pattern=ID_PATTERN, max_length=ID_MAX_LENGTH),
    session: AsyncSession = Depends(get_session),
):
    """Actually download the checkpoint archive bytes"""
    checkpoint_path = await validate_checkpoint(
        request, unique_id, checkpoint_id, types.CheckpointType.SAMPLER, session
    )

    file_buffer = await asyncio.to_thread(download_file, checkpoint_path)

    filename = f"{unique_id}_{checkpoint_id}.tar.gz"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Length": str(file_buffer.getbuffer().nbytes),
    }

    return StreamingResponse(file_buffer, media_type="application/octet-stream", headers=headers)


@app.get("/api/v1/training_runs/{unique_id}/checkpoints")
async def list_checkpoints(
    unique_id: str = fastapi.Path(..., pattern=ID_PATTERN, max_length=ID_MAX_LENGTH),
    session: AsyncSession = Depends(get_session),
):
    """List checkpoints for a model."""
    statement = (
        select(CheckpointDB)
        .where(CheckpointDB.model_id == unique_id)
        .where(CheckpointDB.status == CheckpointStatus.COMPLETED)
    )
    result = await session.exec(statement)

    checkpoints = []
    for checkpoint in result.all():
        # Construct tinker_path based on checkpoint type
        path_kind = "weights" if checkpoint.checkpoint_type == types.CheckpointType.TRAINING else "sampler_weights"
        tinker_path = f"tinker://{unique_id}/{path_kind}/{checkpoint.checkpoint_id}"

        checkpoints.append(
            Checkpoint(
                checkpoint_id=checkpoint.checkpoint_id,
                checkpoint_type=checkpoint.checkpoint_type.value,
                time=checkpoint.completed_at,
                tinker_path=tinker_path,
            )
        )

    return ListCheckpointsResponse(checkpoints=checkpoints)


@app.get("/api/v1/models/{unique_id}/checkpoints")
async def list_checkpoints_models(
    unique_id: str = fastapi.Path(..., pattern=ID_PATTERN, max_length=ID_MAX_LENGTH),
    session: AsyncSession = Depends(get_session),
):
    """Just to be compatible with tinker SDK"""
    return await list_checkpoints(unique_id=unique_id, session=session)


@app.post("/api/v1/weights_info", response_model=WeightsInfoResponse)
async def get_weights_info(request: WeightsInfoRequest, req: Request, session: AsyncSession = Depends(get_session)):
    """Get information about weights/checkpoint from a tinker path."""
    path = types.TinkerPath.parse(request.tinker_path)

    if not path or path.kind != "weights":
        raise HTTPException(
            status_code=400, detail="Invalid tinker path format. Expected: tinker://model_id/weights/checkpoint_id"
        )

    model_id = path.primary_id
    checkpoint_id = path.secondary_id

    # Get model info (this will raise 404 if model doesn't exist)
    model = await get_model(session, model_id)

    # Validate checkpoint exists and is completed
    await validate_checkpoint(req, model_id, checkpoint_id, types.CheckpointType.TRAINING, session)

    lora_config = types.LoraConfig.model_validate(model.lora_config)
    is_lora = lora_config.rank > 0

    return WeightsInfoResponse(
        base_model=model.base_model,
        is_lora=is_lora,
        lora_rank=lora_config.rank,
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Tinker API Mock",
        "version": "0.0.1",
        "endpoints": {
            "models": ["/api/v1/create_model", "/api/v1/get_info", "/api/v1/training_runs/{model_id}"],
            "training": ["/api/v1/forward_backward", "/api/v1/optim_step"],
            "futures": ["/api/v1/retrieve_future"],
            "service": ["/api/v1/get_server_capabilities"],
            "telemetry": ["/api/v1/telemetry"],
            "checkpoints": ["/api/v1/training_runs/{unique_id}/checkpoints"],
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
