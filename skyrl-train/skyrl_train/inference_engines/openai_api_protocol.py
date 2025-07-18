import time
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """OpenAI chat message format."""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request model (minimal version)."""
    model: str  # We'll ignore this
    messages: List[ChatMessage]
    
    # Standard OpenAI sampling parameters
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    
    # Extended parameters
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = 1.0
    min_p: Optional[float] = 0.0
    
    # User field
    user: Optional[str] = None
    
    # Unsupported fields that should cause errors
    stream: Optional[bool] = False
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None
    response_format: Optional[Any] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    
    @field_validator("messages")
    @classmethod
    def validate_messages_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("Messages array cannot be empty")
        return v
    
    @field_validator("stream")
    @classmethod
    def validate_stream_not_supported(cls, v):
        if v is True:
            raise ValueError("Streaming is not supported")
        return v
    
    @field_validator("tools")
    @classmethod
    def validate_tools_not_supported(cls, v):
        if v is not None:
            raise ValueError("Tools are not supported")
        return v
    
    @field_validator("tool_choice")
    @classmethod
    def validate_tool_choice_not_supported(cls, v):
        if v is not None and v != "none":
            raise ValueError("Tool choice is not supported")
        return v
    
    @field_validator("response_format")
    @classmethod
    def validate_response_format_not_supported(cls, v):
        if v is not None:
            raise ValueError("Response format constraints are not supported")
        return v
    
    @field_validator("n")
    @classmethod
    def validate_n_equals_one(cls, v):
        if v != 1:
            raise ValueError("Only n=1 is supported")
        return v


class ChatCompletionResponseChoice(BaseModel):
    """OpenAI chat completion response choice."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    """Usage statistics (minimal version)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response (minimal version)."""
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class ErrorResponse(BaseModel):
    """Error response format."""
    error: Dict[str, Any]
