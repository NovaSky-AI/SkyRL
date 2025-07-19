"""
A minimal set of OpenAI API protocol for inference engine http server.
"""

import time
from typing import List, Optional, Hashable, Union, Dict, Any

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """OpenAI chat message format."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request model (minimal version)."""

    model: str
    messages: List[ChatMessage]

    # Common sampling parameters
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    ignore_eos: Optional[bool] = None
    skip_special_tokens: Optional[bool] = None
    include_stop_str_in_output: Optional[bool] = None
    min_tokens: Optional[int] = None
    n: Optional[int] = None  # Only n=1 is supported

    # SkyRL-specific parameters
    trajectory_id: Optional[Hashable] = None

    # Unsupported parameters that we still parse for error reporting
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    best_of: Optional[int] = None

    @field_validator("n")
    @classmethod
    def validate_n(cls, v):
        if v is not None and v != 1:
            raise ValueError("Only n=1 is supported")
        return v


class ChatCompletionResponseChoice(BaseModel):
    """OpenAI chat completion response choice."""

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    # NOTE: Not including logprobs for now.


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response (minimal version)."""

    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    model: str


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


UNSUPPORTED_FIELDS = ["tools", "tool_choice", "logprobs", "top_logprobs", "best_of"]


def check_unsupported_fields(request: ChatCompletionRequest) -> None:
    """Raise ValueError if unsupported fields are provided."""
    unsupported = []
    for field in UNSUPPORTED_FIELDS:
        if getattr(request, field) is not None:
            unsupported.append(field)
    if request.n not in (None, 1):
        unsupported.append("n")
    if unsupported:
        raise ValueError(f"Unsupported fields: {', '.join(unsupported)}")


def build_sampling_params(request: ChatCompletionRequest, backend: str) -> Dict[str, Any]:
    """Convert request sampling params to backend specific sampling params."""
    assert backend in ["vllm", "sglang"], f"Unsupported backend: {backend}"

    request_dict = request.model_dump(exclude_unset=True)

    sampling_fields = [
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "stop",
        "stop_token_ids",
        "presence_penalty",
        "frequency_penalty",
        "ignore_eos",
        "skip_special_tokens",
        "n",
    ]

    params = {field: request_dict[field] for field in sampling_fields if field in request_dict}

    max_token_key = "max_tokens" if backend == "vllm" else "max_new_tokens"
    if "max_tokens" in request_dict:
        params[max_token_key] = request_dict["max_tokens"]

    # Fields that only vllm supports
    vllm_only_sampling_fields = ["include_stop_str_in_output", "seed", "min_tokens"]
    for field in vllm_only_sampling_fields:
        if field in request_dict:
            if backend == "vllm":
                params[field] = request_dict[field]
            elif backend == "sglang":
                if request_dict[field] is not None:
                    raise ValueError(f"{field} is not supported for sglang backend")

    return params
