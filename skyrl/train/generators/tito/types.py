"""Shared types for TITO proxy rendering and trace commits."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from skyrl.train.generators.base import TrajectoryID

Message = Dict[str, Any]
ToolSpec = Dict[str, Any]
RoutedExpertToken = Tuple[Tuple[int, ...], ...]
RoutedExperts = Tuple[RoutedExpertToken, ...]


@dataclass(frozen=True)
class PendingTurn:
    """Read-only trace preparation result for one chat completion request."""

    trace_revision: int
    messages: Tuple[Message, ...]
    tools: Optional[Tuple[ToolSpec, ...]]
    tools_hash: str
    bridge_transition_id: Optional[int]


@dataclass(frozen=True)
class ModelTurnResult:
    """Exact renderer and inference result committed to a trace."""

    prompt_token_ids: Tuple[int, ...]
    prompt_message_indices: Tuple[int, ...]
    reused_prefix_length: int
    completion_ids: Tuple[int, ...]
    completion_logprobs: Tuple[float, ...]
    assistant_message: Message
    stop_reason: str
    routed_experts: Optional[RoutedExperts] = None
    model: str = ""
    sampling_params_json: str = "{}"


@dataclass(frozen=True)
class CommitResult:
    """Identifiers created by a successful atomic trace commit."""

    turn_id: int
    assistant_node_id: int


@dataclass(frozen=True)
class TransitionRecord:
    """Constant-size reference to one successful inference call."""

    transition_id: int
    tools_hash: str
    assistant_node_id: int
    stop_reason: str
    model: str
    sampling_params_json: str


@dataclass(frozen=True)
class TraceOutcome:
    """One completed trial attempt plus its trajectory-level outcome."""

    trace: Any
    trajectory_id: TrajectoryID
    reward: float
    stop_reason: str
    generation_time: Optional[float]
    num_turns: int
