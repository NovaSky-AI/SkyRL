"""Shared types for TITO proxy rendering and trace commits."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from skyrl.train.generators.base import TrajectoryID

Message = Dict[str, Any]
ToolSpec = Dict[str, Any]
RoutedExpertToken = Tuple[Tuple[int, ...], ...]
RoutedExperts = Tuple[RoutedExpertToken, ...]


@dataclass(frozen=True)
class BridgeAnchor:
    """Exact prior model turn that a renderer may safely extend."""

    node_id: int
    matched_message_count: int
    previous_prompt_ids: Tuple[int, ...]
    previous_completion_ids: Tuple[int, ...]


@dataclass(frozen=True)
class PendingTurn:
    """Read-only trace preparation result for one chat completion request."""

    trace_revision: int
    request_key: str
    messages: Tuple[Message, ...]
    tools: Optional[Tuple[ToolSpec, ...]]
    tools_hash: str
    matched_node_ids: Tuple[int, ...]
    new_messages: Tuple[Message, ...]
    bridge_anchor: Optional[BridgeAnchor]


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


@dataclass(frozen=True)
class CommitResult:
    """Identifiers created by a successful atomic trace commit."""

    turn_id: int
    assistant_node_id: int


@dataclass(frozen=True)
class CommittedTurn:
    """Internal exact record for one successful inference call."""

    turn_id: int
    request_key: str
    tools_hash: str
    prompt_leaf_id: Optional[int]
    assistant_node_id: int
    prompt_token_ids: Tuple[int, ...]
    completion_ids: Tuple[int, ...]
    completion_logprobs: Tuple[float, ...]
    assistant_message: Message
    stop_reason: str
    routed_experts: Optional[RoutedExperts] = None


@dataclass(frozen=True)
class TraceOutcome:
    """One completed trial attempt plus its trajectory-level outcome."""

    trace: Any
    trajectory_id: TrajectoryID
    reward: float
    stop_reason: str
    generation_time: Optional[float]
    num_turns: int
