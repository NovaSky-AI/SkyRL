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

    model: str  # We'll ignore this
    messages: List[ChatMessage]

    # Common sampling parameters
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    ignore_eos: Optional[bool] = None
    skip_special_tokens: Optional[bool] = None
    include_stop_str_in_output: Optional[bool] = None
    min_tokens: Optional[int] = None
    best_of: Optional[int] = None
    use_beam_search: Optional[bool] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None

    # Unsupported parameters that we still parse for error reporting
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    n: Optional[int] = None

    # SkyRL-specific parameters
    trajectory_id: Optional[Hashable] = None

    @field_validator("n")
    @classmethod
    def validate_n(cls, v):
        if v is not None and v != 1:
            raise ValueError("Only n=1 is supported")
        return v

    # TODO(Charlie): we currently ignore all other parameters. Will revisit
    # once we figure out the workflow for users to use inference engine http server.
    # The curernt behavior is that we will use the sampling parameters defined in
    # ppo_base_config.yaml for all requests.


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
    model: str
    choices: List[ChatCompletionResponseChoice]


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


UNSUPPORTED_FIELDS = ["tools", "tool_choice"]


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
    params: Dict[str, Any] = {}

    def set_param(name: str, value: Any, target_name: Optional[str] = None) -> None:
        if value is not None:
            params[target_name or name] = value

    # map common params
    set_param("temperature", request.temperature)
    set_param("top_p", request.top_p)
    set_param("top_k", request.top_k)
    set_param("min_p", request.min_p)
    set_param("repetition_penalty", request.repetition_penalty)
    set_param("length_penalty", request.length_penalty)
    set_param("seed", request.seed)
    set_param("stop", request.stop)
    set_param("stop_token_ids", request.stop_token_ids)
    set_param("presence_penalty", request.presence_penalty)
    set_param("frequency_penalty", request.frequency_penalty)
    set_param("ignore_eos", request.ignore_eos)
    set_param("skip_special_tokens", request.skip_special_tokens)
    set_param("include_stop_str_in_output", request.include_stop_str_in_output)
    set_param("min_tokens", request.min_tokens)
    set_param("best_of", request.best_of)
    set_param("use_beam_search", request.use_beam_search)
    set_param("logprobs", request.logprobs)
    set_param("top_logprobs", request.top_logprobs)

    max_token_key = "max_tokens" if backend == "vllm" else "max_new_tokens"
    set_param(max_token_key, request.max_tokens)

    return params
