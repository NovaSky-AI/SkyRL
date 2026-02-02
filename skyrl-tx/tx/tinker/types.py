# These are the types we use to represent the data internally.
# They have some commonalities with the API request and response
# types as well as the database models, but are distinct. For
# example, usually we try to avoid optional values in these types.

from __future__ import annotations

from enum import Enum
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel


class RequestType(str, Enum):
    """Types of requests that can be processed."""

    CREATE_MODEL = "create_model"
    FORWARD_BACKWARD = "forward_backward"
    FORWARD = "forward"
    OPTIM_STEP = "optim_step"
    SAVE_WEIGHTS_FOR_SAMPLER = "save_weights_for_sampler"
    SAVE_WEIGHTS = "save_weights"
    LOAD_WEIGHTS = "load_weights"
    SAMPLE = "sample"
    UNLOAD_MODEL = "unload_model"

    # External request that should not be processed by the engine
    EXTERNAL = "external"


class CheckpointType(str, Enum):
    """Type of checkpoint."""

    TRAINING = "training"
    SAMPLER = "sampler"


class TinkerPath(BaseModel):
    primary_id: str
    kind: str
    secondary_id: str

    @classmethod
    def parse(cls, url: str) -> TinkerPath | None:
        """Parse a URL string into a TinkerPath object."""
        parsed = urlparse(url)

        match (parsed.scheme, *parsed.path.split("/")):
            case ("tinker", "", secondary_id):
                return cls(primary_id=parsed.netloc, kind="", secondary_id=secondary_id)
            case ("tinker", "", kind, secondary_id):
                return cls(primary_id=parsed.netloc, kind=kind, secondary_id=secondary_id)
            case _:
                return None


class AdamParams(BaseModel):
    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float


class LoraConfig(BaseModel):
    rank: int
    alpha: float
    seed: int
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = False


class CreateModelInput(BaseModel):
    lora_config: LoraConfig


class CreateModelOutput(BaseModel):
    model_id: str
    base_model: str
    lora_config: LoraConfig


class UnloadModelInput(BaseModel):
    pass


class UnloadModelOutput(BaseModel):
    model_id: str
    status: str
    type: str = "unload_model"


class ModelInputChunk(BaseModel):
    tokens: list[int]


class ModelInput(BaseModel):
    chunks: list[ModelInputChunk]


class TensorData(BaseModel):
    data: list[int] | list[float]


class LossFnInputs(BaseModel):
    target_tokens: TensorData
    weights: TensorData
    advantages: TensorData
    logprobs: TensorData


class Datum(BaseModel):
    loss_fn_inputs: LossFnInputs
    model_input: ModelInput


class ForwardBackwardInput(BaseModel):
    data: list[Datum]
    loss_fn: Literal["cross_entropy", "importance_sampling", "ppo"]


class ForwardBackwardOutput(BaseModel):
    loss_fn_output_type: str
    loss_fn_outputs: list[dict]
    metrics: dict


class ErrorResponse(BaseModel):
    error: str
    status: str


class OptimStepInput(BaseModel):
    adam_params: AdamParams


class OptimStepOutput(BaseModel):
    pass


class SaveWeightsForSamplerInput(BaseModel):
    path: str | None = None
    sampling_session_seq_id: int | None = None
    seq_id: int | None = None
    sampling_session_id: str | None = None


class SaveWeightsForSamplerOutput(BaseModel):
    path: str | None = None
    type: str
    sampling_session_id: str | None = None


class SaveWeightsInput(BaseModel):
    path: str


class SaveWeightsOutput(BaseModel):
    path: str
    type: str


class LoadWeightsInput(BaseModel):
    source_model_id: str
    checkpoint_id: str


class LoadWeightsOutput(BaseModel):
    type: str


class SamplingParams(BaseModel):
    temperature: float
    max_tokens: int
    seed: int
    stop_tokens: list[int] | None = None
    stop_strings: list[str] | None = None
    top_k: int = -1  # -1 for no limit
    top_p: float = 1.0  # 1.0 for no filtering


class ModelMetadata(BaseModel):
    adapter_index: int
    lora_config: LoraConfig
    loaded_checkpoint_id: str | None = None


class SampleInput(BaseModel):
    base_model: str | None = None
    prompt: ModelInput
    sampling_params: SamplingParams
    num_samples: int
    checkpoint_id: str
    prompt_logprobs: bool


class GeneratedSequence(BaseModel):
    stop_reason: Literal["length", "stop"]
    tokens: list[int]
    logprobs: list[float]


class SampleOutput(BaseModel):
    sequences: list[GeneratedSequence]
    prompt_logprobs: list[float] | None = None


# Metrics tracked in the engine
class EngineMetrics(BaseModel):
    train_seq_len_jit_times: dict[int, float] = {}
    sample_seq_len_jit_times: dict[int, float] = {}


# Prepared batch data for backend processing
# These are prepared by the engine and passed to the backend


class PreparedModelPassBatch(BaseModel):
    """Prepared batch data for forward/forward_backward operations.

    Engine extracts this from requests, backend converts to JAX arrays and computes.
    """

    # Per-example data (list of lists)
    all_input_ids: list[list[int]]
    all_targets: list[list[int]]
    all_token_weights: list[list[float]]
    all_sampling_logprobs: list[list[float]]
    all_advantages: list[list[float]]

    # Per-example scalars
    all_model_ids: list[str]
    all_loss_fn_types: list[int]

    # Mapping from examples back to requests: (request_id, model_id, start_idx, end_idx)
    request_batch_slices: list[tuple[str, str, int, int]]


class PreparedSampleBatch(BaseModel):
    """Prepared batch data for sample operations.

    Engine extracts this from requests, backend converts to JAX arrays and computes.
    """

    # Per-sample data
    all_prompts: list[list[int]]
    all_sampling_params: list[SamplingParams]
    all_model_ids: list[str]
    all_checkpoint_ids: list[str]
    all_checkpoint_paths: list[str]

    # Whether any request needs prompt logprobs
    needs_prompt_logprobs: bool

    # Mapping from samples back to requests: (request_id, model_id, start_idx, end_idx, prompt_logprobs_requested)
    request_batch_slices: list[tuple[str, str, int, int, bool]]


# Loss function type mappings (used for validation and backend dispatch)
LOSS_TYPES = {
    "cross_entropy": 0,
    "importance_sampling": 1,
    "ppo": 2,
}
