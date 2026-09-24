"""One trajectory: its graph, its lifecycle, and the document it is read as."""

from __future__ import annotations

import asyncio
import dataclasses
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from skycap.graph import MessageGraph

Status = Literal["open", "finished", "failed", "abandoned"]


def new_trajectory_id() -> str:
    return f"tr_{secrets.token_hex(8)}"


@dataclass(slots=True)
class Failure:
    """A call that produced no node: an upstream error, an unreadable reply."""

    t: float
    status: int | None
    error: str
    input_leaf: int | None = None


@dataclass(eq=False)
class Trajectory:
    id: str
    meta: dict[str, Any] = field(default_factory=dict)
    #: How this server captured it: mode, and for tokens the tokenizer and logprobs mode.
    capture: dict[str, Any] = field(default_factory=dict)
    status: Status = "open"
    annotations: dict[str, Any] = field(default_factory=dict)
    graph: MessageGraph = field(default_factory=MessageGraph)
    failures: list[Failure] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    last_active: float = field(default_factory=time.monotonic)
    #: Handlers serving this trajectory right now; sealing cancels them.
    inflight: set[asyncio.Task[Any]] = field(default_factory=set)
    #: Set once the server has released and written it. Sealed is not ended:
    #: a trajectory can fail mid-run and still be waiting for its ``finish``.
    ended: bool = False

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    def touch(self) -> None:
        self.last_active = time.monotonic()

    def seal(self, status: Status, annotations: dict[str, Any] | None = None) -> None:
        """Close the trajectory. In-flight calls are cancelled and never committed."""
        if not self.is_open:
            return
        self.status = status
        self.annotations.update(annotations or {})
        self.finished_at = time.time()
        current = asyncio.current_task() if _loop_running() else None
        for task in list(self.inflight):
            if task is not current:
                task.cancel()
        self.inflight.clear()

    def document(self) -> dict[str, Any]:
        """Everything but the token arrays, as JSON-ready data."""
        return {
            "id": self.id,
            "status": self.status,
            "meta": self.meta,
            "capture": self.capture,
            "annotations": self.annotations,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "tools": self.graph.tools,
            "failures": [dataclasses.asdict(f) for f in self.failures],
            "nodes": [
                {
                    "id": n.id,
                    "parent": n.parent,
                    "depth": n.depth,
                    "role": n.role,
                    "author": n.author,
                    "message": n.message,
                    "match_hash": n.match_hash,
                    "delta_hash": n.delta_hash,
                    "created_at": n.created_at,
                    "calls": [dataclasses.asdict(c) for c in n.calls],
                    # History matching this message continues from that sibling instead.
                    "shadowed_by": self.graph.shadowed_by(n.id),
                }
                for n in self.graph
            ],
        }


def _loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True
