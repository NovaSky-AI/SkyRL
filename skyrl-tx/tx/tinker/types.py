# These are the types we use to represent the data internally.
# They have some commonalities with the API request and response
# types as well as the database models but are distinct. For
# example, usually we try to avoid optional values in these types.

from enum import Enum
from typing import Any

from pydantic import BaseModel


class RequestType(str, Enum):
    """Types of requests that can be processed."""
    CREATE_MODEL = "create_model"
    FORWARD_BACKWARD = "forward_backward"
    OPTIM_STEP = "optim_step"
    SAVE_WEIGHTS_FOR_SAMPLER = "save_weights_for_sampler"


class AdamParams(BaseModel):
    lr: float


class LoraConfig(BaseModel):
    rank: int


class CreateModelInput(BaseModel):
    lora_config: LoraConfig


class CreateModelOutput(BaseModel):
    model_id: str
    base_model: str
    lora_config: LoraConfig


class ForwardBackwardInput(BaseModel):
    forward_backward_input: dict[str, Any]


class OptimStepInput(BaseModel):
    adam_params: AdamParams


class OptimStepOutput(BaseModel):
    pass


class SaveWeightsForSamplerInput(BaseModel):
    path: str


class SaveWeightsForSamplerOutput(BaseModel):
    path: str
    type: str
