"""Rehydrating a token trace from a trajectory's hot aggregate.

The in-memory trace is a cache, not the record. A trajectory whose trace was
evicted under the budget -- or that a replacement process recovered from its
journal -- must still find its committed token deltas, or prefix reuse would
silently stop working and every turn would start a new branch.

Loading is exact: node token arrays come from the graph, and the transitions
are rebuilt from the exchanges that produced each assistant node. Both are on
the aggregate, and the aggregate is either hot or recovered from the journal
that made those nodes durable -- so a trace rebuilt after a restart is the
trace the previous process had.
"""

from __future__ import annotations

from typing import Any

from skyrl_capture.domain.records import ActiveTrajectory
from skyrl_capture.tito.trace import TokenNode, TokenTrace
from skyrl_capture.tito.types import TokenError, Transition


class StoredTraces:
    """`TraceSource` over the registry: what the session manager rebuilds from.

    The one place token capture reads a trajectory's aggregate, so when that
    changes this is the class that changes and the session manager does not.
    """

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    async def load(self, trajectory_id: str) -> TokenTrace:
        active = await self._registry.resolve(trajectory_id)
        if active is None:
            raise TokenError(
                f"trajectory {trajectory_id!r} is not open on this process, so its exact "
                "token trace cannot be rebuilt"
            )
        return load_trace(active)


def load_trace(active: ActiveTrajectory) -> TokenTrace:
    trace = TokenTrace(active.id)
    for node_row in active.graph.ordered():
        payload = node_row.payload
        if payload is None:
            # Without its token delta a node cannot participate in exact
            # matching. Stop here: a partial trace would match a prefix whose
            # tokens are unknown.
            break
        tokens = payload.get("tokens") or {}
        token_ids = tuple(int(value) for value in tokens.get("token_ids") or ())
        if not token_ids:
            break
        text_segments = tokens.get("text_segments")
        if not text_segments:
            raise TokenError(f"stored TITO node {node_row.id!r} has no captured token text")
        sampled_start = tokens.get("sampled_start")
        logprobs = tuple(float(value) for value in tokens.get("logprobs") or ())
        completion_logprobs = logprobs[sampled_start:] if sampled_start is not None else ()
        routed = tokens.get("routed_experts")
        node = TokenNode(
            node_id=node_row.id,
            parent_node_id=node_row.parent_id,
            message=payload["message"],
            token_ids=token_ids,
            sampled_start=sampled_start,
            completion_logprobs=completion_logprobs,
            routed_experts=(
                tuple(tuple(tuple(int(e) for e in layer) for layer in token) for token in routed)
                if routed
                else None
            ),
            depth=node_row.depth,
            exchange_id=node_row.exchange_id,
        )
        node.text_segments = list(text_segments)
        node.turn_start_token = tokens.get("turn_start_token")
        trace.register(node)

    # Rebuild transitions from the exchanges that produced each assistant node,
    # in observed order, so bridge lookups resolve the same way they did live.
    exchange_rows: list[dict[str, Any]] = [
        row for row in active.exchange_rows() if row.get("output_node_id") is not None
    ]
    for exchange in exchange_rows:
        node_id = exchange["output_node_id"]
        if node_id not in trace._nodes:  # noqa: SLF001 - rehydration is internal
            continue
        node = trace.node(node_id)
        if node.sampled_start is None:
            continue
        metadata = exchange["source_metadata"] or {}
        tokens_meta = metadata.get("tokens") or {}
        trace.register_transition(
            Transition(
                transition_id=len(trace._transitions),  # noqa: SLF001
                assistant_node_id=node_id,
                tools_hash=tokens_meta.get("tools_hash") or metadata.get("tools_hash") or "",
                stop_reason=exchange["completion_reason"] or "stop",
                model=exchange["model"] or "",
                sampling_params=tokens_meta.get("sampling_params") or {},
                prompt_token_count=int(tokens_meta.get("prompt_token_count") or 0),
                completion_token_count=len(node.token_ids) - node.sampled_start,
            )
        )
    return trace
