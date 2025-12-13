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


class LoraConfig(BaseModel):
    rank: int
    alpha: float
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = False


class CreateModelInput(BaseModel):
    lora_config: LoraConfig


class CreateModelOutput(BaseModel):
    model_id: str
    base_model: str
    lora_config: LoraConfig


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
    path: str


class SaveWeightsForSamplerOutput(BaseModel):
    path: str
    type: str


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
    stop: list[int] | None = None


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
