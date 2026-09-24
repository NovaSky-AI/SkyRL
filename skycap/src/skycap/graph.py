"""The context graph: one trajectory's model calls as a forest of messages.

A node is one message and has one parent, so every root-to-leaf path is one
complete conversation and a shared prefix is stored once. Two children under
one parent are a fork: a resample, a subagent, a compaction, a harness edit.
None of those is special-cased; each is what the matching rule does with a
particular shape of input.

Every node carries two hashes, because two different questions get asked of it:

* ``match_hash`` -- "is this the message the request is sending?" It covers
  the message, the call's tool set and the model: what a harness can put in a
  request. Walking a request's history down the graph uses only this.
* ``delta_hash`` -- "is this the same node?" It is what ``add`` dedupes on,
  under a given parent. For a client-authored node it equals the match hash.
  For a model-authored node it also covers the author and the sampling params
  that shape the support, so the same output under a different ``top_p`` is a
  different sample, and a harness-written message never absorbs a sample that
  happens to equal it.

So author and sampling only decide whether a *new* node is created; they never
stop history from matching. A harness that replays the model's reply verbatim
matches the model node (by match hash) and continues from it: no branch. It
branches only if the replayed message differs, e.g. an edit or stripped
reasoning, and then the new node is client-authored.

Identical outputs under the same conditions are one node that records every
call that produced it. Rarely, several siblings share a match hash but differ
in delta: the same text sampled under two ``top_p`` values, a harness-written
message equal to a sample, or (token mode) the same text tokenized two ways.
History then continues from the latest model-authored one, or the latest
client-authored one if there is no model sibling. The others are
``shadowed_by`` it: still valid nodes whose paths train normally, marked so a
reader can see that later history couldn't be attributed to them.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from skycap import hashing

Author = Literal["client", "model"]


@dataclass(slots=True)
class CallInfo:
    """One successful model call, recorded on the node it produced."""

    t_start: float
    t_end: float
    model: str | None = None
    sampling: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None


@dataclass(slots=True)
class NodeTokens:
    """The exact tokens a node added to the sequence. Token-in/token-out mode only.

    For a model node, ``token_ids[:sampled_start]`` is template scaffold the
    model did not produce, and the rest is what it sampled. ``logprobs``,
    ``routed_experts`` and ``sampling_mask`` are aligned to ``token_ids``;
    ``sampling_mask`` has one support set per sampled token.

    ``text`` is what these tokens decode to, special tokens kept, and
    ``text_offsets[k]`` is the UTF-8 byte offset in ``text`` where token ``k``
    starts, so token ``k`` is ``text_bytes[text_offsets[k]:text_offsets[k + 1]]``
    (the last one ends at the end of ``text``). Offsets never decrease. A
    character that spans several tokens belongs to the token that completes it,
    and the tokens before it have empty spans, so every span decodes on its own.
    Both are filled when the trajectory is recorded, so a reader never needs the
    tokenizer.
    """

    token_ids: list[int]
    sampled_start: int | None = None
    logprobs: list[float] | None = None
    routed_experts: Any | None = None
    sampling_mask: list[list[int]] | None = None
    text: str | None = None
    text_offsets: list[int] | None = None


@dataclass(slots=True)
class Node:
    id: int
    parent: int | None
    depth: int
    role: str | None
    author: Author
    message: dict[str, Any]
    match_hash: str
    delta_hash: str
    created_at: float
    tokens: NodeTokens | None = None
    calls: list[CallInfo] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TextTurn:
    """Where one text-mode call landed in the graph."""

    matched: int
    input_leaf: int | None
    output: int
    created: tuple[int, ...]

    @property
    def output_created(self) -> bool:
        return self.output in self.created


class MessageGraph:
    """One trajectory's graph."""

    __slots__ = ("nodes", "tools", "_by_delta", "_by_match", "_children")

    def __init__(self) -> None:
        self.nodes: list[Node] = []
        #: Every tool set seen, by hash, so a call's tools are stored once.
        self.tools: dict[str, list[dict[str, Any]]] = {}
        #: (parent id, delta hash) -> node id. How ``add`` finds a node it already made.
        self._by_delta: dict[tuple[int | None, str], int] = {}
        #: (parent id, match hash) -> node id: the latest model-authored such sibling,
        #: else the latest client-authored one. How ``match`` walks a request's history.
        self._by_match: dict[tuple[int | None, str], int] = {}
        #: parent id (``None`` for roots) -> child node ids, in creation order.
        self._children: dict[int | None, list[int]] = {}

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    # -- the one mutation ------------------------------------------------------
    def add(
        self,
        parent: int | None,
        *,
        role: str | None,
        author: Author,
        message: Mapping[str, Any],
        match_hash: str,
        delta_hash: str,
        created_at: float,
        tokens: NodeTokens | None = None,
    ) -> tuple[Node, bool]:
        """The node for this delta under ``parent``, creating it if absent.

        ``match_hash`` is what later requests' history is matched against;
        ``delta_hash`` is what identifies the node, so it's what an existing node
        is found by (see the module docstring for why they differ). Returns
        ``(node, created)``. An existing node is returned unchanged.
        """
        existing = self._by_delta.get((parent, delta_hash))
        if existing is not None:
            return self.nodes[existing], False
        if parent is not None and not 0 <= parent < len(self.nodes):
            raise KeyError(f"no node {parent}")
        node = Node(
            id=len(self.nodes),
            parent=parent,
            depth=0 if parent is None else self.nodes[parent].depth + 1,
            role=role,
            author=author,
            message=dict(message),
            match_hash=match_hash,
            delta_hash=delta_hash,
            created_at=created_at,
            tokens=tokens,
        )
        self.nodes.append(node)
        self._by_delta[(parent, delta_hash)] = node.id
        chosen = self._by_match.get((parent, match_hash))
        if chosen is None or author == "model" or self.nodes[chosen].author == "client":
            self._by_match[(parent, match_hash)] = node.id
        self._children.setdefault(parent, []).append(node.id)
        return node, True

    # -- matching ----------------------------------------------------------------
    def child(self, parent: int | None, match_hash: str) -> int | None:
        return self._by_match.get((parent, match_hash))

    def match(self, match_hashes: Sequence[str]) -> list[int]:
        """The longest prefix of ``match_hashes`` already in the graph, as node ids."""
        matched: list[int] = []
        parent: int | None = None
        for match in match_hashes:
            node = self.child(parent, match)
            if node is None:
                break
            matched.append(node)
            parent = node
        return matched

    def commit_text(
        self,
        messages: Sequence[Mapping[str, Any]],
        output: Mapping[str, Any],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        model: str | None,
        call: CallInfo,
    ) -> TextTurn:
        """Place one successful text-mode call.

        1. Hash each request message with the call's tools and model.
        2. Walk the longest matching prefix.
        3. Add one client node per unmatched message, in order.
        4. Add the model's reply under the last one, or find it if it is
           already there, and record ``call`` on it.

        A failed call commits nothing; the caller records it elsewhere.
        """
        tools_key = hashing.tools_hash(tools)
        if tools_key and tools is not None:
            self.tools.setdefault(tools_key, [dict(tool) for tool in tools])
        matches = [hashing.match_hash(m, tools=tools_key, model=model) for m in messages]
        matched = self.match(matches)
        parent = matched[-1] if matched else None
        created: list[int] = []
        for message, match in zip(messages[len(matched) :], matches[len(matched) :], strict=True):
            node, new = self.add(
                parent,
                role=message.get("role"),
                author="client",
                message=message,
                match_hash=match,
                delta_hash=match,
                created_at=call.t_start,
            )
            if new:
                created.append(node.id)
            parent = node.id
        output_match = hashing.match_hash(output, tools=tools_key, model=model)
        reply, new = self.add(
            parent,
            role=output.get("role"),
            author="model",
            message=output,
            match_hash=output_match,
            delta_hash=hashing.model_delta_hash(output_match, call.sampling),
            created_at=call.t_end,
        )
        reply.calls.append(call)
        if new:
            created.append(reply.id)
        return TextTurn(matched=len(matched), input_leaf=parent, output=reply.id, created=tuple(created))

    # -- reading -------------------------------------------------------------------
    def children(self, node: int | None) -> list[int]:
        """Children of ``node`` (roots for ``None``), in creation order."""
        return list(self._children.get(node, ()))

    def roots(self) -> list[int]:
        return self.children(None)

    def leaves(self) -> list[int]:
        return [node.id for node in self.nodes if node.id not in self._children]

    def path_to(self, node: int) -> list[int]:
        path: list[int] = []
        current: int | None = node
        while current is not None:
            path.append(current)
            current = self.nodes[current].parent
        path.reverse()
        return path

    def paths(self) -> list[list[int]]:
        """Every root-to-leaf path, in leaf creation order."""
        return [self.path_to(leaf) for leaf in self.leaves()]

    def shadowed_by(self, node: int) -> int | None:
        """The sibling that history matching this node's message continues from, if not this one."""
        n = self.nodes[node]
        chosen = self._by_match[(n.parent, n.match_hash)]
        return None if chosen == node else chosen

    def branch_points(self) -> list[int]:
        return [parent for parent, kids in self._children.items() if parent is not None and len(kids) > 1]
