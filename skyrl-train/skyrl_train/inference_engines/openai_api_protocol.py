"""
A minimal set of OpenAI API protocol for inference HTTP endpoint.
"""

import time
from typing import List, Optional, Hashable, Union, Dict, Any, Literal, Type
import json
from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """OpenAI chat message format."""

    role: str
    content: str


class JsonSchemaResponseFormat(BaseModel):
    name: str
    description: Optional[str] = None
    # schema is the field in openai but that causes conflicts with pydantic so
    # instead use json_schema with an alias
    json_schema: Optional[dict[str, Any]] = Field(default=None, alias="schema")
    strict: Optional[bool] = None


class ResponseFormat(BaseModel):
    # type must be "json_schema", "json_object", or "text"
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None


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
    response_format: Optional[ResponseFormat] = None

    # SkyRL-specific parameters
    trajectory_id: Optional[Hashable] = None

    # Unsupported parameters that we still parse for error reporting
    stream: bool = False
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

    @field_validator("stream")
    @classmethod
    def validate_stream(cls, v):
        if v:
            raise ValueError("Streaming is not supported")
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


########################################################
# Building sampling params
# TODO(Charlie): support structural_tag, and see if the
# extra fields from vLLM/SGLang like `guided_regex`/`regex`
# are needed in ChatCompletionRequest.
########################################################


def build_response_format_vllm(request: ChatCompletionRequest) -> Dict[str, Any]:
    """
    Build response format for vllm backend.
    Code adapted from https://github.com/vllm-project/vllm/blob/v0.9.2/vllm/entrypoints/openai/protocol.py#L483
    """

    guided_json_object = None
    guided_json = None
    if request.response_format is not None:
        from vllm.sampling_params import GuidedDecodingParams

        if request.response_format.type == "json_object":
            guided_json_object = True
        elif request.response_format.type == "json_schema":
            json_schema = request.response_format.json_schema
            assert json_schema is not None
            guided_json = json_schema.json_schema

        guided_decoding = GuidedDecodingParams.from_optional(
            json=guided_json,
            json_object=guided_json_object,
        )
        return {"guided_decoding": guided_decoding}
    else:
        return {}


def build_response_format_sglang(request: ChatCompletionRequest) -> Dict[str, Any]:
    """
    Build response format for sglang backend.
    Code adapted from https://github.com/sgl-project/sglang/blob/v0.4.8.post1/python/sglang/srt/entrypoints/openai/serving_chat.py#L314
    """

    def _convert_json_schema_to_str(json_schema: Union[dict, str, Type[BaseModel]]) -> str:
        """Convert a JSON schema to a string.
        Parameters
        ----------
        json_schema
            The JSON schema.
        Returns
        -------
        str
            The JSON schema converted to a string.
        Raises
        ------
        ValueError
            If the schema is not a dictionary, a string or a Pydantic class.
        """
        if isinstance(json_schema, dict):
            schema_str = json.dumps(json_schema)
        elif isinstance(json_schema, str):
            schema_str = json_schema
        elif issubclass(json_schema, BaseModel):
            schema_str = json.dumps(json_schema.model_json_schema())
        else:
            raise ValueError(
                f"Cannot parse schema {json_schema}. The schema must be either "
                + "a Pydantic class, a dictionary or a string that contains the JSON "
                + "schema specification"
            )
        return schema_str

    result = {}
    if request.response_format and request.response_format.type == "json_schema":
        result["json_schema"] = _convert_json_schema_to_str(request.response_format.json_schema.json_schema)
    elif request.response_format and request.response_format.type == "json_object":
        result["json_schema"] = '{"type": "object"}'
    return result


# TODO(Charlie): consolidate sampling params building logics across the repo.
def build_sampling_params(request: ChatCompletionRequest, backend: str) -> Dict[str, Any]:
    """Convert request sampling params to backend specific sampling params."""
    assert backend in ["vllm", "sglang"], f"Unsupported backend: {backend}"

    request_dict = request.model_dump(exclude_unset=True)

    # 1. Shared fields between vllm and sglang
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

    # 2. Same field but different name
    max_token_key = "max_tokens" if backend == "vllm" else "max_new_tokens"
    if "max_tokens" in request_dict:
        params[max_token_key] = request_dict["max_tokens"]
    include_stop_str_key = "include_stop_str_in_output" if backend == "vllm" else "no_stop_trim"
    if include_stop_str_key in request_dict:
        params[include_stop_str_key] = request_dict[include_stop_str_key]

    # 3. Fields that only vllm supports
    vllm_only_sampling_fields = ["seed", "min_tokens"]
    for field in vllm_only_sampling_fields:
        if field in request_dict:
            if backend == "vllm":
                params[field] = request_dict[field]
            elif backend == "sglang":
                if request_dict[field] is not None:
                    raise ValueError(f"{field} is not supported for sglang backend")

    # 4. Response format
    if backend == "vllm":
        params.update(build_response_format_vllm(request))
    elif backend == "sglang":
        params.update(build_response_format_sglang(request))

    return params
