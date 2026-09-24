"""One token-mode turn: where it attaches, what tokens it sends, and what it commits.

Planning, in order:

1. Match the request's messages against the graph in message space.
2. If the matched prefix contains a model node, bridge from the deepest one:
   the renderer extends that call's exact prompt and completion with the new
   messages instead of re-rendering what the model sampled.
3. Otherwise, or if the renderer declines, render in full and keep only the
   prefix of matched nodes whose tokens reproduce the render exactly. The rest
   is a fork, never a node whose tokens disagree with what inference received.
4. Split the new tokens into one chunk per new message using the renderer's
   attribution. Scaffold belongs to the message that follows it; scaffold
   after the last message is the generation prompt, which opens the model node.

Committing adds those nodes and the model node (its scaffold plus the sampled
completion), each with its slice of the routed experts. The model node also
gets logprobs and the sampling mask.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from skycap import hashing
from skycap.graph import CallInfo, MessageGraph, NodeTokens
from skycap.tokens.engine import EngineOutput
from skycap.tokens.renderer import TokenRenderer


class TokenError(Exception):
    """A turn whose tokens can't be attributed exactly. Never committed."""


@dataclass(slots=True)
class Plan:
    prompt_ids: list[int]
    #: The node the new tail hangs from, and how many tokens precede the tail.
    parent: int | None
    prefix_len: int
    #: Index in the request's messages of the first new message.
    tail_start: int
    chunks: list[list[int]]
    scaffold: list[int]
    bridged: bool


def match_hashes(messages: Sequence[Mapping[str, Any]], tools: str, model: str | None) -> list[str]:
    return [hashing.match_hash(m, tools=tools, model=model) for m in messages]


def plan(
    graph: MessageGraph,
    renderer: TokenRenderer,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    matches: Sequence[str],
) -> Plan:
    if not messages:
        raise TokenError("a request needs at least one message")
    matched = graph.match(matches)
    bridged = _bridge(graph, renderer, messages, tools, matched)
    if bridged is not None:
        return bridged
    rendered = renderer.render(messages, tools)
    prompt = rendered.token_ids
    offset, parent, start = 0, None, 0
    for depth, node_id in enumerate(matched):
        tokens = graph.nodes[node_id].tokens
        if tokens is None or prompt[offset : offset + len(tokens.token_ids)] != tokens.token_ids:
            break
        offset += len(tokens.token_ids)
        parent, start = node_id, depth + 1
    indices = [i - start if i >= 0 else -1 for i in rendered.tail_indices[offset:]]
    chunks, scaffold = attribute(prompt[offset:], indices, len(messages) - start)
    return Plan(prompt, parent, offset, start, chunks, scaffold, bridged=False)


def _bridge(
    graph: MessageGraph,
    renderer: TokenRenderer,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    matched: list[int],
) -> Plan | None:
    """Extend the deepest matched model call, if there is one and the renderer agrees."""
    for depth in range(len(matched) - 1, -1, -1):
        node = graph.nodes[matched[depth]]
        if node.author != "model" or node.tokens is None or node.tokens.sampled_start is None:
            continue
        new_messages = messages[depth + 1 :]
        if not new_messages:
            return None
        ss = node.tokens.sampled_start
        previous_prompt = [t for i in matched[:depth] for t in _tokens(graph, i)] + node.tokens.token_ids[:ss]
        previous_completion = node.tokens.token_ids[ss:]
        rendered = renderer.bridge(previous_prompt, previous_completion, new_messages, tools)
        if rendered is None:
            return None
        chunks, scaffold = attribute(rendered.token_ids[rendered.reused :], rendered.tail_indices, len(new_messages))
        return Plan(rendered.token_ids, node.id, rendered.reused, depth + 1, chunks, scaffold, bridged=True)
    return None


def _tokens(graph: MessageGraph, node_id: int) -> list[int]:
    tokens = graph.nodes[node_id].tokens
    if tokens is None:
        raise TokenError(f"node {node_id} on a token path has no tokens")
    return tokens.token_ids


def attribute(token_ids: Sequence[int], indices: Sequence[int], count: int) -> tuple[list[list[int]], list[int]]:
    """Split tokens into ``count`` per-message chunks and the trailing generation prompt.

    ``indices`` are relative to the new messages; ``-1`` is scaffold, which is
    attributed to the message after it. A walk from the end makes "the message
    after it" cheap to know.
    """
    if len(token_ids) != len(indices):
        raise TokenError("token attribution arrays differ in length")
    owners: list[int | None] = [None] * len(token_ids)
    following: int | None = None
    for position in range(len(token_ids) - 1, -1, -1):
        index = indices[position]
        if index >= count:
            raise TokenError(f"renderer attributed a token to message {index} of {count}")
        if index >= 0:
            following = index
        owners[position] = following
    chunks: list[list[int]] = [[] for _ in range(count)]
    scaffold: list[int] = []
    last = 0
    for token, owner in zip(token_ids, owners, strict=True):
        if owner is None:
            scaffold.append(token)
            continue
        if owner < last:
            raise TokenError("renderer attribution is not in message order")
        last = owner
        chunks[owner].append(token)
    return chunks, scaffold


def commit(
    graph: MessageGraph,
    turn: Plan,
    *,
    messages: Sequence[Mapping[str, Any]],
    matches: Sequence[str],
    reply: Mapping[str, Any],
    reply_match: str,
    output: EngineOutput,
    call: CallInfo,
) -> int:
    """Add the turn's nodes. Returns the model node's id."""
    if not output.completion_ids:
        raise TokenError("cannot commit an empty completion")
    total = len(turn.prompt_ids) + len(output.completion_ids)
    routed = _Routing(output.routed_experts, output.routed_start, total)
    routed.replace_placeholder(graph, turn.parent, turn.prefix_len)

    parent, position = turn.parent, turn.prefix_len
    for offset, chunk in enumerate(turn.chunks):
        index = turn.tail_start + offset
        node, new = graph.add(
            parent,
            role=messages[index].get("role"),
            author="client",
            message=messages[index],
            match_hash=matches[index],
            delta_hash=hashing.client_token_delta_hash(matches[index], chunk),
            created_at=call.t_start,
            tokens=NodeTokens(token_ids=list(chunk), routed_experts=routed.slice(position, len(chunk))),
        )
        parent, position = node.id, position + len(chunk)

    tokens = [*turn.scaffold, *output.completion_ids]
    sampled_start = len(turn.scaffold)
    reply_node, new = graph.add(
        parent,
        role=reply.get("role"),
        author="model",
        message=reply,
        match_hash=reply_match,
        delta_hash=hashing.model_token_delta_hash(reply_match, call.sampling, tokens, sampled_start),
        created_at=call.t_end,
        tokens=NodeTokens(
            token_ids=tokens,
            sampled_start=sampled_start,
            logprobs=[0.0] * sampled_start + list(output.logprobs),
            routed_experts=routed.slice(position, len(tokens)),
            sampling_mask=output.sampling_mask,
        ),
    )
    reply_node.calls.append(call)
    return reply_node.id


class _Routing:
    """This turn's routed experts, sliced per node by sequence position.

    The engine never runs a forward pass on the last sampled token, so the
    array is one row short. The node that ends the sequence gets a copy of the
    previous row as a placeholder, and the next turn that forwards that token
    replaces it with the real row.
    """

    def __init__(self, array: np.ndarray | None, start: int, total: int) -> None:
        self.array, self.start, self.total = array, start, total
        if array is not None and array.ndim != 3:
            raise TokenError(f"routed experts must be [tokens, layers, k], got shape {array.shape}")
        if array is not None and start + array.shape[0] not in (total - 1, total):
            raise TokenError(f"routed experts cover {start}+{array.shape[0]} of {total} positions")

    def slice(self, position: int, length: int) -> np.ndarray | None:
        if self.array is None or length == 0:
            return None
        begin, end = position - self.start, position - self.start + length
        if begin < 0:
            return None
        rows = self.array[begin:end]
        if rows.shape[0] == length - 1 and end == self.total - self.start:
            rows = np.concatenate([rows, rows[-1:] if len(rows) else self.array[-1:]], axis=0)
        return rows.copy() if rows.shape[0] == length else None

    def replace_placeholder(self, graph: MessageGraph, parent: int | None, prefix_len: int) -> None:
        if self.array is None or parent is None or prefix_len == 0:
            return
        node = graph.nodes[parent]
        tokens = node.tokens
        if node.author != "model" or tokens is None or tokens.routed_experts is None:
            return
        row = prefix_len - 1 - self.start
        if 0 <= row < self.array.shape[0] and tokens.routed_experts.shape[1:] == self.array.shape[1:]:
            tokens.routed_experts = np.concatenate(
                [tokens.routed_experts[:-1], self.array[row : row + 1].astype(tokens.routed_experts.dtype)]
            )
