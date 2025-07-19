"""
A minimal set of OpenAI API protocol for inference engine http server.
"""

import time
from typing import List, Optional, Hashable

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """OpenAI chat message format."""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request model (minimal version)."""
    model: str  # We'll ignore this
    messages: List[ChatMessage]

    # SkyRL-specific parameters
    trajectory_id: Optional[Hashable] = None

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
