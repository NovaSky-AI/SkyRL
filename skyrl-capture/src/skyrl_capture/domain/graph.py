"""One conversation graph for both capture modes.

A node is a message. One parent per node, so every root-to-leaf path is one
complete conversation and a shared prefix is stored once; a node with more
than one child is a fork, which is the whole representation of branching.
Token capture adds exact token fields to a node; text capture leaves them
absent. Nothing else differs.

Node identity is ``(parent_key, delta_hash)`` within a trajectory. In text
mode ``delta_hash`` covers the message, its tools and the model; in token mode
it also covers the exact token delta and the sampled boundary, so two
identical messages with different tokenizations stay distinct. Committing the
same delta under the same parent twice finds the existing node, which is what
makes a retried turn idempotent instead of a fork.

The graph changes only by applying a `GraphDelta`: the nodes one exchange
introduced, plus how that exchange attaches to the graph. Text capture plans a
delta by matching parsed messages against the graph (`plan_text_append`);
token capture plans one from the nodes its trace committed
(`plan_token_append`). Both produce the same type, and `apply` is idempotent
by node id, so a replayed delta converges rather than duplicating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from skyrl_capture.domain.extraction import ExtractedMessage, normalize
from skyrl_capture.domain.hashing import chain_hash, node_identity
from skyrl_capture.ids import node_id as new_node_id
from skyrl_capture.version import DERIVATION_VERSION

ROOT_PARENT_KEY = ""


@dataclass(slots=True)
class GraphNode:
    """One message in one trajectory's graph, with everything a reader shows."""

    id: str
    parent_id: str | None
    exchange_id: str
    depth: int
    role: str
    author: str
    message_hash: str
    delta_hash: str
    context_hash: str
    content_block_count: int = 0
    char_count: int = 0
    # Token capture only.
    token_count: int | None = None
    sampled_start: int | None = None
    sampled_token_count: int | None = None
    has_logprobs: bool = False
    has_routed_experts: bool = False
    tokenizer: str | None = None
    # When the model produced this node, for a model-authored one.
    output_started_at: datetime | None = None
    output_ended_at: datetime | None = None
    inserted_at: datetime | None = None
    derivation: dict[str, Any] = field(default_factory=dict)
    # The message delta -- and in token mode the exact token arrays. Inline:
    # the graph is what the exporters, the token trace and the viewer read,
    # and the log carries it as part of the exchange that committed it.
    payload: dict[str, Any] | None = None

    # -- what the payload holds, for the exporters and the viewer -------------
    # A node carries its own message delta, and in token mode the exact token
    # arrays. These read it; nothing else should reach into `payload` by key.
    @property
    def message(self) -> dict[str, Any] | None:
        return None if self.payload is None else self.payload.get("message")

    @property
    def tokens(self) -> dict[str, Any]:
        return {} if self.payload is None else (self.payload.get("tokens") or {})

    @property
    def token_ids(self) -> list[int]:
        return list(self.tokens.get("token_ids") or [])

    @property
    def sampled_mask(self) -> list[bool]:
        return list(self.tokens.get("sampled_mask") or [])

    @property
    def logprobs(self) -> list[float]:
        return list(self.tokens.get("logprobs") or [])

    @property
    def routed_experts(self) -> list[Any] | None:
        return self.tokens.get("routed_experts")

    @property
    def parent_key(self) -> str:
        return self.parent_id or ROOT_PARENT_KEY

    def public(self) -> dict[str, Any]:
        """The `/v1/trajectories/{id}/graph` node shape."""
        return {
            "node_id": self.id,
            "parent_node_id": self.parent_id,
            "exchange_id": self.exchange_id,
            "depth": self.depth,
            "role": self.role,
            "author": self.author,
            "message_hash": self.message_hash,
            "delta_hash": self.delta_hash,
            "content_block_count": self.content_block_count,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "sampled_start": self.sampled_start,
            "sampled_token_count": self.sampled_token_count,
            "has_logprobs": self.has_logprobs,
            "has_routed_experts": self.has_routed_experts,
            "tokenizer": self.tokenizer,
            "inserted_at": self.inserted_at.isoformat() if self.inserted_at else None,
            "output_started_at": self.output_started_at.isoformat() if self.output_started_at else None,
            "output_ended_at": self.output_ended_at.isoformat() if self.output_ended_at else None,
            "derivation": self.derivation,
            "has_payload": self.payload is not None,
        }


@dataclass(frozen=True, slots=True)
class GraphDelta:
    """What one exchange added to a trajectory's graph, and how it attaches.

    ``nodes`` are the nodes this exchange introduced, in commit order; a node
    the graph already held is not here. The association fields say where the
    exchange's context sat in the graph, and they reference existing nodes as
    freely as new ones.
    """

    exchange_id: str
    nodes: tuple[GraphNode, ...]
    # Where the request's context matched, and what it added.
    input_prefix_node_id: str | None
    input_node_ids: tuple[str, ...]
    input_leaf_node_id: str | None
    # The model's output, and the model output its context continued from.
    output_node_id: str | None
    parent_output_node_id: str | None
    matched_count: int
    is_duplicate_retry: bool

    @property
    def created_node_ids(self) -> tuple[str, ...]:
        return tuple(node.id for node in self.nodes)

    def association(self) -> dict[str, Any]:
        """The exchange-row columns this delta decides."""
        return {
            "input_prefix_node_id": self.input_prefix_node_id,
            "input_node_ids": list(self.input_node_ids),
            "input_leaf_node_id": self.input_leaf_node_id,
            "output_node_id": self.output_node_id,
            "parent_output_node_id": self.parent_output_node_id,
            "is_duplicate_retry": self.is_duplicate_retry,
        }


class ConversationGraph:
    """One trajectory's graph, and the four operations on it."""

    __slots__ = ("nodes", "order", "_by_delta")

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.order: list[str] = []
        # (parent_key, delta_hash) -> node id: how a repeat commit finds the
        # node it already made instead of making a second one.
        self._by_delta: dict[tuple[str, str], str] = {}

    def __len__(self) -> int:
        return len(self.nodes)

    # -- lookups -------------------------------------------------------------
    def child(self, parent_key: str, delta_hash: str) -> str | None:
        return self._by_delta.get((parent_key, delta_hash))

    def depth_of(self, node_id: str | None) -> int | None:
        node = self.nodes.get(node_id) if node_id else None
        return node.depth if node else None

    def author_of(self, node_id: str | None) -> str | None:
        node = self.nodes.get(node_id) if node_id else None
        return node.author if node else None

    def context_of(self, node_id: str | None) -> str:
        node = self.nodes.get(node_id) if node_id else None
        return node.context_hash if node else ""

    def match_prefix(self, identities: list[str]) -> list[str]:
        """Longest exact prefix already present, as node ids.

        ``identities`` are per-message delta hashes. Matching walks from the
        root, so a history rewritten from message *k* stops matching at *k*
        and the exchange branches there; nothing that matches is re-added.
        """
        matched: list[str] = []
        parent_key = ROOT_PARENT_KEY
        for identity in identities:
            node_id = self.child(parent_key, identity)
            if node_id is None:
                break
            matched.append(node_id)
            parent_key = node_id
        return matched

    # -- what a reader asks ---------------------------------------------------
    def ordered(self) -> list[GraphNode]:
        """Every node, by ``(depth, id)``: the order two builds of one graph agree on."""
        return sorted(self.nodes.values(), key=lambda node: (node.depth, node.id))

    def children_of(self) -> dict[str | None, list[str]]:
        children: dict[str | None, list[str]] = {}
        for node in self.ordered():
            children.setdefault(node.parent_id, []).append(node.id)
        return children

    def leaves(self, *, author: str | None = None) -> list[str]:
        """Nodes with no children, in depth then id order."""
        parents = {node.parent_id for node in self.nodes.values()}
        return [
            node.id
            for node in self.ordered()
            if node.id not in parents and (author is None or node.author == author)
        ]

    def branch_points(self) -> list[dict[str, Any]]:
        """Nodes with more than one child: the visible forks."""
        children = self.children_of()
        return [
            {"node_id": parent, "child_count": len(ids), "child_ids": sorted(ids)}
            for parent, ids in sorted(children.items(), key=lambda pair: pair[0] or "")
            if parent is not None and len(ids) > 1
        ]

    def path_to(self, node_id: str) -> list[GraphNode]:
        path: list[GraphNode] = []
        current: str | None = node_id
        while current is not None:
            node = self.nodes.get(current)
            if node is None:
                break
            path.append(node)
            current = node.parent_id
        path.reverse()
        return path

    # -- the one mutation -------------------------------------------------------
    def apply(self, delta: GraphDelta, *, at: datetime | None = None) -> None:
        """Insert the delta's nodes. Idempotent by node id and by identity.

        A node already present -- by id on a replay, by ``(parent_key,
        delta_hash)`` if two planners raced, which one writer per trajectory
        rules out -- is left as it is. Graph attribution is a fact about the
        moment an exchange was *first* committed.
        """
        for node in delta.nodes:
            if node.id in self.nodes:
                continue
            key = (node.parent_key, node.delta_hash)
            if key in self._by_delta:
                continue
            if node.inserted_at is None and at is not None:
                node.inserted_at = at
            self.nodes[node.id] = node
            self.order.append(node.id)
            self._by_delta[key] = node.id

    # -- planning ----------------------------------------------------------------
    def plan_text_append(
        self,
        *,
        exchange_id: str,
        request_messages: list[ExtractedMessage],
        output_message: ExtractedMessage | None,
        tools_hash: str | None,
        model: str | None,
        output_started_at: datetime | None,
        output_ended_at: datetime | None,
    ) -> GraphDelta:
        """Match the request against the graph and plan the new tail.

        The rules are deliberately strict; an edge is claimed only with exact
        evidence. A rewritten or compacted history stops matching at the last
        unchanged message and branches there; an identical retry matches every
        message and then finds its assistant node present, so it adds nothing
        and is flagged rather than forked; a genuine fork is two distinct
        messages under one parent.

        The messages arrive as an adapter extracted them -- provider-native,
        unhashed. Identity is applied here, once, the same way for every
        provider: an adapter cannot decide what makes two messages the same
        message, which is what keeps a contributed one from quietly changing
        the shape of a graph.
        """
        request_messages = [normalize(message) for message in request_messages]
        output_message = normalize(output_message) if output_message is not None else None
        identities = [node_identity(m.message_hash, tools_hash, model) for m in request_messages]
        matched = self.match_prefix(identities)
        parent_id: str | None = matched[-1] if matched else None
        depth = (self.depth_of(parent_id) or 0) + 1 if parent_id else 0
        # The exchange's context continued from a previous model output only
        # when the matched prefix leaf is itself a model node. That is the only
        # relationship an exporter may treat as a fork.
        parent_output_node_id = parent_id if parent_id and self.author_of(parent_id) == "model" else None
        context = self.context_of(parent_id)

        planned: list[GraphNode] = []
        committed_inputs: list[str] = []
        # Nodes this plan introduces are visible to the rest of the plan --
        # the assistant node hangs off the last request node -- so the plan
        # keeps its own view of what it has added.
        added: dict[tuple[str, str], str] = {}

        def existing(parent_key: str, identity: str) -> str | None:
            return self.child(parent_key, identity) or added.get((parent_key, identity))

        tail = request_messages[len(matched):]
        for offset, message in enumerate(tail):
            identity = node_identity(message.message_hash, tools_hash, model)
            parent_key = parent_id or ROOT_PARENT_KEY
            found = existing(parent_key, identity)
            if found is None:
                node = GraphNode(
                    id=new_node_id(),
                    parent_id=parent_id,
                    exchange_id=exchange_id,
                    depth=depth,
                    role=message.role,
                    author="client",
                    message_hash=message.message_hash,
                    delta_hash=identity,
                    context_hash=chain_hash(context, identity),
                    content_block_count=message.content_block_count,
                    char_count=message.char_count,
                    derivation={
                        "matched_prefix_messages": len(matched),
                        "tail_offset": offset,
                        "derivation_version": DERIVATION_VERSION,
                        **message.derivation,
                    },
                    payload=_text_payload(message, author="client"),
                )
                planned.append(node)
                added[(parent_key, identity)] = node.id
                found = node.id
            context = chain_hash(context, identity)
            committed_inputs.append(found)
            parent_id = found
            depth += 1

        input_leaf = parent_id
        output_node_id: str | None = None
        output_created = False
        if output_message is not None:
            identity = node_identity(output_message.message_hash, tools_hash, model)
            parent_key = parent_id or ROOT_PARENT_KEY
            found = existing(parent_key, identity)
            if found is None:
                node = GraphNode(
                    id=new_node_id(),
                    parent_id=parent_id,
                    exchange_id=exchange_id,
                    depth=depth,
                    role=output_message.role,
                    author="model",
                    message_hash=output_message.message_hash,
                    delta_hash=identity,
                    context_hash=chain_hash(context, identity),
                    content_block_count=output_message.content_block_count,
                    char_count=output_message.char_count,
                    output_started_at=output_started_at,
                    output_ended_at=output_ended_at,
                    derivation={
                        "matched_prefix_messages": len(matched),
                        "derivation_version": DERIVATION_VERSION,
                        **output_message.derivation,
                    },
                    payload=_text_payload(output_message, author="model"),
                )
                planned.append(node)
                found = node.id
                output_created = True
            output_node_id = found

        # A retry is a call whose entire request context already existed. If
        # its response also already existed, nothing at all was added.
        newly_committed_inputs = [node.id for node in planned if node.author == "client"]
        is_duplicate_retry = bool(request_messages) and not newly_committed_inputs and not output_created

        return GraphDelta(
            exchange_id=exchange_id,
            nodes=tuple(planned),
            input_prefix_node_id=matched[-1] if matched else None,
            input_node_ids=tuple(newly_committed_inputs),
            input_leaf_node_id=input_leaf,
            output_node_id=output_node_id,
            parent_output_node_id=parent_output_node_id,
            matched_count=len(matched),
            is_duplicate_retry=is_duplicate_retry,
        )

    def plan_token_append(
        self,
        *,
        exchange_id: str,
        tokens: dict[str, Any],
        output_ended_at: datetime | None,
    ) -> GraphDelta:
        """Plan from the node structure the token proxy committed.

        Node ids, parents, token deltas and sampled boundaries all come from
        the proxy, which owns the renderer, so the persisted graph is identical
        to the trace that served the turn. Only the context hash is chained
        here, over parents that may already be in this graph.
        """
        context_by_node: dict[str, str] = {}
        planned: list[GraphNode] = []
        for row in tokens.get("nodes") or ():
            parent_id = row["parent_node_id"]
            parent_context = context_by_node.get(parent_id or "")
            if parent_context is None:
                parent_context = self.context_of(parent_id) if parent_id else ""
            context_hash = chain_hash(parent_context, row["delta_hash"])
            context_by_node[row["id"]] = context_hash
            planned.append(
                GraphNode(
                    id=row["id"],
                    parent_id=parent_id,
                    exchange_id=exchange_id,
                    depth=row["depth"],
                    role=row["role"],
                    author=row["author"],
                    message_hash=row["message_hash"],
                    delta_hash=row["delta_hash"],
                    context_hash=context_hash,
                    content_block_count=row["content_block_count"],
                    char_count=row["char_count"],
                    token_count=row["token_count"],
                    sampled_start=row["sampled_start"],
                    sampled_token_count=row["sampled_token_count"],
                    has_logprobs=row["has_logprobs"],
                    has_routed_experts=row["has_routed_experts"],
                    tokenizer=tokens.get("tokenizer"),
                    output_ended_at=output_ended_at if row["author"] == "model" else None,
                    derivation={
                        "renderer": tokens.get("renderer"),
                        "reused_prefix_length": tokens.get("reused_prefix_length"),
                        "matched_message_count": tokens.get("matched_message_count"),
                        "bridge_transition_id": tokens.get("bridge_transition_id"),
                        "derivation_version": DERIVATION_VERSION,
                    },
                    payload=row["payload"],
                )
            )
        return GraphDelta(
            exchange_id=exchange_id,
            nodes=tuple(planned),
            input_prefix_node_id=tokens.get("parent_output_node_id"),
            input_node_ids=tuple(node.id for node in planned if node.author == "client"),
            input_leaf_node_id=tokens.get("input_leaf_node_id"),
            output_node_id=tokens.get("assistant_node_id"),
            parent_output_node_id=tokens.get("parent_output_node_id"),
            matched_count=int(tokens.get("matched_message_count") or 0),
            is_duplicate_retry=not tokens.get("nodes"),
        )


def _text_payload(message: Any, *, author: str) -> dict[str, Any]:
    """The stored body for a text node: its message delta and derivation."""
    return {
        "message": message.message,
        "role": message.role,
        "author": author,
        "message_hash": message.message_hash,
        "content_block_count": message.content_block_count,
        "char_count": message.char_count,
        "derivation": message.derivation,
    }
