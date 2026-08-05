"""Token-in/token-out proxy support for external agent frameworks."""

from .composer import build_trace_generator_output
from .trace import BranchView, Trace, TransitionView
from .types import (
    BridgeAnchor,
    CommitResult,
    ModelTurnResult,
    PendingTurn,
    TraceOutcome,
    TransitionRecord,
)

__all__ = [
    "BranchView",
    "BridgeAnchor",
    "CommitResult",
    "ModelTurnResult",
    "PendingTurn",
    "Trace",
    "TraceOutcome",
    "TransitionRecord",
    "TransitionView",
    "build_trace_generator_output",
]
