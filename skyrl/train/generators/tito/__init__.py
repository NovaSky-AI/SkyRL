"""Token-in/token-out proxy support for external agent frameworks."""

from .composer import build_trace_generator_output
from .trace import Trace
from .types import (
    BridgeAnchor,
    CommitResult,
    CommittedTurn,
    ModelTurnResult,
    PendingTurn,
    TraceOutcome,
)

__all__ = [
    "BridgeAnchor",
    "CommitResult",
    "CommittedTurn",
    "ModelTurnResult",
    "PendingTurn",
    "Trace",
    "TraceOutcome",
    "build_trace_generator_output",
]
