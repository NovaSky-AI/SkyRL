"""token capture value types.

These mirror the reference implementation's contract (see
docs/design/token-capture-parity.md) because the fields are load-bearing, not
incidental: ``prompt_message_indices`` is what makes one-node-per-message
possible in token space, and ``reused_prefix_length`` is what separates the
prefix the proxy already committed from the tokens this turn adds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]
ToolSpec = dict[str, Any]
# Per token, per layer, the routed expert IDs. MoE training needs these aligned
# to the full prompt+completion sequence or not at all.
RoutedExperts = tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class RenderedPrompt:
    """Prompt token IDs plus per-token message attribution.

    ``message_indices`` describes only the tokens this turn *added* -- that is,
    ``token_ids[reused_prefix_length:]``. ``message_indices[i]`` is the index of
    the message that ``token_ids[reused_prefix_length + i]`` belongs to, or
    ``-1`` for chat-template scaffold tokens that belong to no message.

    A full render sets ``reused_prefix_length`` to 0, so there the indices cover
    everything and the two readings coincide. A bridged turn carries indices for
    the tail alone, which is what keeps a long conversation's per-turn cost
    proportional to the turn rather than to the context.
    """

    token_ids: tuple[int, ...]
    message_indices: tuple[int, ...]
    reused_prefix_length: int = 0

    def __post_init__(self) -> None:
        if self.reused_prefix_length < 0 or self.reused_prefix_length > len(self.token_ids):
            raise ValueError("reused_prefix_length is outside the prompt token range")
        if len(self.message_indices) != len(self.token_ids) - self.reused_prefix_length:
            raise ValueError(
                "message_indices must cover exactly the tokens after reused_prefix_length"
            )


@dataclass(frozen=True)
class PendingTurn:
    """Read-only preparation result for one chat request."""

    revision: int
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] | None
    tools_hash: str
    bridge_transition_id: int | None
    matched_node_ids: tuple[str, ...]


@dataclass(frozen=True)
class ModelTurnResult:
    """Exact renderer and inference result for one turn."""

    prompt_token_ids: tuple[int, ...]
    # Covers ``prompt_token_ids[reused_prefix_length:]`` only -- see RenderedPrompt.
    prompt_message_indices: tuple[int, ...]
    reused_prefix_length: int
    completion_ids: tuple[int, ...]
    completion_logprobs: tuple[float, ...]
    assistant_message: Message
    stop_reason: str
    routed_experts: RoutedExperts | None = None
    model: str = ""
    sampling_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    """One successful inference call, anchored on its assistant node."""

    transition_id: int
    assistant_node_id: str
    tools_hash: str
    stop_reason: str
    model: str
    sampling_params: dict[str, Any]
    prompt_token_count: int
    completion_token_count: int


@dataclass(frozen=True)
class CommitResult:
    transition_id: int
    assistant_node_id: str
    created_node_ids: tuple[str, ...]
    reused_node_ids: tuple[str, ...]
    input_leaf_node_id: str | None
    parent_output_node_id: str | None
    matched_message_count: int
    reused_prefix_length: int
    branched: bool


class TokenError(Exception):
    """Raised when a token capture invariant would be violated.

    These are deliberately fatal to the request rather than logged: a token capture
    exchange whose token deltas cannot be attributed exactly is worse than no
    exchange at all, because it would enter training as if it were exact.
    """


class TokenUpstreamError(TokenError):
    """The engine did not answer in a way this turn can be built from.

    Lives here rather than in ``tokens.engine`` because the request and response
    shapes belong to the upstream kind -- see ``upstream.base`` -- and this is
    the leaf both sides can raise from without an import cycle.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status
