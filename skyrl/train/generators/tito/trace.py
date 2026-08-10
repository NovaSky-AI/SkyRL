"""Exact message-graph bookkeeping for TITO model calls."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .types import (
    BridgeAnchor,
    CommitResult,
    Message,
    ModelTurnResult,
    PendingTurn,
    RoutedExperts,
    TransitionRecord,
)


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"TITO messages and tools must contain JSON values, got {type(value).__name__}")


def _canonical_message(message: Mapping[str, Any]) -> Message:
    normalized = _normalize_json(message)
    if not isinstance(normalized, dict):
        raise TypeError("Message must be a mapping")
    role = normalized.get("role")
    if not isinstance(role, str) or not role:
        raise ValueError("Message must contain a non-empty string `role`")
    return normalized


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MessageNode:
    """One canonical message and its exact token delta."""

    node_id: int
    parent_id: Optional[int]
    message: Message
    message_hash: str
    token_ids: Tuple[int, ...]
    sampled_mask: Tuple[bool, ...]
    logprobs: Tuple[float, ...]
    sampled_start: Optional[int]
    routed_experts: Optional[RoutedExperts] = None


class TransitionView:
    """Lazy exact view of one successful inference call."""

    __slots__ = ("_record", "_trace")

    def __init__(self, trace: "Trace", record: TransitionRecord) -> None:
        self._trace = trace
        self._record = record

    @property
    def transition_id(self) -> int:
        return self._record.transition_id

    @property
    def request_key(self) -> str:
        return self._record.request_key

    @property
    def tools_hash(self) -> str:
        return self._record.tools_hash

    @property
    def assistant_node_id(self) -> int:
        return self._record.assistant_node_id

    @property
    def stop_reason(self) -> str:
        return self._record.stop_reason

    @property
    def model(self) -> str:
        return self._record.model

    @property
    def sampling_params(self) -> Dict[str, Any]:
        return json.loads(self._record.sampling_params_json)

    @property
    def tools(self) -> Optional[Tuple[Dict[str, Any], ...]]:
        return self._trace._tools_by_hash.get(self.tools_hash)

    @property
    def node_ids(self) -> Tuple[int, ...]:
        return self._trace._path_to_node(self.assistant_node_id)

    @property
    def messages(self) -> Tuple[Message, ...]:
        return tuple(self._trace._nodes[node_id].message for node_id in self.node_ids[:-1])

    @property
    def assistant_message(self) -> Message:
        return self._trace._nodes[self.assistant_node_id].message

    @property
    def prompt_token_ids(self) -> Tuple[int, ...]:
        node = self._trace._nodes[self.assistant_node_id]
        if node.sampled_start is None:
            raise RuntimeError("Transition assistant node has no sampled boundary")
        # Assistant nodes store generation scaffold before the sampled suffix.
        tokens = list(self._trace._tokens_for_nodes(self.node_ids[:-1]))
        tokens.extend(node.token_ids[: node.sampled_start])
        return tuple(tokens)

    @property
    def completion_ids(self) -> Tuple[int, ...]:
        node = self._trace._nodes[self.assistant_node_id]
        if node.sampled_start is None:
            raise RuntimeError("Transition assistant node has no sampled boundary")
        return node.token_ids[node.sampled_start :]

    @property
    def completion_logprobs(self) -> Tuple[float, ...]:
        node = self._trace._nodes[self.assistant_node_id]
        if node.sampled_start is None:
            raise RuntimeError("Transition assistant node has no sampled boundary")
        return node.logprobs[node.sampled_start :]

    @property
    def full_token_ids(self) -> Tuple[int, ...]:
        return self.prompt_token_ids + self.completion_ids

    def is_exact_extension_of(self, previous: "TransitionView") -> bool:
        previous_ids = previous.full_token_ids
        prompt_ids = self.prompt_token_ids
        return len(previous_ids) <= len(prompt_ids) and prompt_ids[: len(previous_ids)] == previous_ids


class BranchView:
    """Lazy root-to-leaf exact token path."""

    __slots__ = ("_leaf_id", "_trace")

    def __init__(self, trace: "Trace", leaf_id: int) -> None:
        self._trace = trace
        self._leaf_id = leaf_id

    @property
    def leaf_id(self) -> int:
        return self._leaf_id

    @property
    def node_ids(self) -> Tuple[int, ...]:
        return self._trace._path_to_node(self._leaf_id)

    @property
    def nodes(self) -> Tuple[MessageNode, ...]:
        return tuple(self._trace._nodes[node_id] for node_id in self.node_ids)

    @property
    def transition_ids(self) -> Tuple[int, ...]:
        # Only sampled assistant nodes terminate Transitions.
        return tuple(
            self._trace._transition_id_by_assistant_node[node_id]
            for node_id in self.node_ids
            if node_id in self._trace._transition_id_by_assistant_node
        )

    @property
    def messages(self) -> Tuple[Message, ...]:
        return tuple(node.message for node in self.nodes)

    @property
    def token_ids(self) -> Tuple[int, ...]:
        return self._trace._tokens_for_nodes(self.node_ids)

    @property
    def sampled_mask(self) -> Tuple[bool, ...]:
        mask: List[bool] = []
        for node_id in self.node_ids:
            mask.extend(self._trace._nodes[node_id].sampled_mask)
        return tuple(mask)

    @property
    def logprobs(self) -> Tuple[float, ...]:
        values: List[float] = []
        for node_id in self.node_ids:
            values.extend(self._trace._nodes[node_id].logprobs)
        return tuple(values)

    @property
    def routed_experts(self) -> Optional[RoutedExperts]:
        routed = []
        for node_id in self.node_ids:
            node_routed = self._trace._nodes[node_id].routed_experts
            if node_routed is None:
                return None
            routed.extend(node_routed)
        return tuple(routed)


class Trace:
    """Message graph and exact committed model turns for one trial attempt."""

    def __init__(self) -> None:
        self._nodes: List[MessageNode] = []
        self._children: DefaultDict[Tuple[Optional[int], str], List[int]] = defaultdict(list)
        self._transitions: List[TransitionRecord] = []
        self._transition_id_by_assistant_node: Dict[int, int] = {}
        self._tools_by_hash: Dict[str, Optional[Tuple[Dict[str, Any], ...]]] = {}
        self._commits_by_request_key: Dict[str, CommitResult] = {}
        self._revision = 0
        self._sealed = False

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    def seal(self) -> None:
        self._sealed = True

    def transitions(self) -> Tuple[TransitionView, ...]:
        return tuple(TransitionView(self, record) for record in self._transitions)

    def transition(self, transition_id: int) -> TransitionView:
        return TransitionView(self, self._transitions[transition_id])

    def committed_turns(self) -> Tuple[TransitionView, ...]:
        """Compatibility alias for callers that predate ``transitions()``."""
        return self.transitions()

    def nodes(self) -> Tuple[MessageNode, ...]:
        return tuple(self._nodes)

    def branches(self) -> Tuple[BranchView, ...]:
        # Graph leaves define the exact training branches.
        parents = {node.parent_id for node in self._nodes if node.parent_id is not None}
        return tuple(BranchView(self, node.node_id) for node in self._nodes if node.node_id not in parents)

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "storage": {
                "nodes": len(self._nodes),
                "transitions": len(self._transitions),
                "branches": len(self.branches()),
                "stored_token_ids": sum(len(node.token_ids) for node in self._nodes),
                "materialized_transition_token_ids": sum(
                    len(transition.full_token_ids) for transition in self.transitions()
                ),
            },
            "nodes": [
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "message": node.message,
                    "token_ids": list(node.token_ids),
                    "sampled_mask": list(node.sampled_mask),
                    "logprobs": list(node.logprobs),
                    "sampled_start": node.sampled_start,
                }
                for node in self._nodes
            ],
            "transitions": [
                {
                    "transition_id": transition.transition_id,
                    "request_key": transition.request_key,
                    "assistant_node_id": transition.assistant_node_id,
                    "messages": list(transition.messages),
                    "assistant_message": transition.assistant_message,
                    "prompt_token_ids": list(transition.prompt_token_ids),
                    "completion_ids": list(transition.completion_ids),
                    "completion_logprobs": list(transition.completion_logprobs),
                    "stop_reason": transition.stop_reason,
                    "model": transition.model,
                    "sampling_params": transition.sampling_params,
                    "tools": list(transition.tools) if transition.tools is not None else None,
                }
                for transition in self.transitions()
            ],
            "branches": [
                {
                    "branch_index": branch_index,
                    "leaf_id": branch.leaf_id,
                    "node_ids": list(branch.node_ids),
                    "transition_ids": list(branch.transition_ids),
                    "messages": list(branch.messages),
                    "token_ids": list(branch.token_ids),
                    "sampled_mask": list(branch.sampled_mask),
                    "logprobs": list(branch.logprobs),
                }
                for branch_index, branch in enumerate(self.branches())
            ],
        }

    def prepare_turn(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        request_key: str,
    ) -> PendingTurn:
        """Find the longest semantic prefix and an exact renderer bridge anchor."""
        if self._sealed:
            raise RuntimeError("Cannot prepare a turn on a sealed trace")
        if not messages:
            raise ValueError("Chat completion request must contain at least one message")
        if not request_key:
            raise ValueError("request_key must be non-empty")

        canonical_messages = tuple(_canonical_message(message) for message in messages)
        canonical_tools = None
        if tools is not None:
            normalized_tools = _normalize_json(tools)
            if not isinstance(normalized_tools, list):
                raise TypeError("tools must be a sequence")
            canonical_tools = tuple(normalized_tools)
        tools_hash = _stable_hash(canonical_tools or [])

        # Match in message space before the renderer produces prompt IDs.
        matched_node_ids = self._longest_message_prefix(canonical_messages)
        bridge_anchor = self._find_bridge_anchor(matched_node_ids, tools_hash)
        matched_message_count = bridge_anchor.matched_message_count if bridge_anchor is not None else 0

        return PendingTurn(
            trace_revision=self._revision,
            request_key=request_key,
            messages=canonical_messages,
            tools=canonical_tools,
            tools_hash=tools_hash,
            matched_node_ids=matched_node_ids,
            new_messages=canonical_messages[matched_message_count:],
            bridge_anchor=bridge_anchor,
        )

    def commit(self, pending: PendingTurn, result: ModelTurnResult) -> CommitResult:
        """Atomically add one exact inference result to the message graph."""
        if self._sealed:
            raise RuntimeError("Cannot commit to a sealed trace")
        existing = self._commits_by_request_key.get(pending.request_key)
        if existing is not None:
            return existing
        if pending.trace_revision != self._revision:
            raise RuntimeError(
                f"Stale pending turn: prepared at revision {pending.trace_revision}, current revision is {self._revision}"
            )
        self._validate_result(pending, result)

        # Stage graph updates until all attribution checks pass.
        parent_id, message_chunks, assistant_scaffold = self._prepare_prompt_commit(pending, result)
        staged_nodes: List[MessageNode] = []
        routed_experts = result.routed_experts
        routed_cursor = len(self._tokens_for_nodes(self._path_to_node(parent_id))) if parent_id is not None else 0

        message_start = len(pending.messages) - len(message_chunks)
        for offset, token_ids in enumerate(message_chunks):
            message = pending.messages[message_start + offset]
            # Reuse exact prompt nodes discovered by a full render.
            reusable = self._find_exact_child(parent_id, message, token_ids)
            if reusable is not None:
                parent_id = reusable
                routed_cursor += len(token_ids)
                continue
            node_routed = (
                routed_experts[routed_cursor : routed_cursor + len(token_ids)] if routed_experts is not None else None
            )
            node = self._build_node(
                parent_id=parent_id,
                message=message,
                token_ids=token_ids,
                sampled_start=None,
                completion_logprobs=(),
                routed_experts=node_routed,
                staged_count=len(staged_nodes),
            )
            staged_nodes.append(node)
            parent_id = node.node_id
            routed_cursor += len(token_ids)

        assistant_tokens = assistant_scaffold + result.completion_ids
        assistant_sampled_start = len(assistant_scaffold)
        assistant_routed = (
            routed_experts[routed_cursor : routed_cursor + len(assistant_tokens)]
            if routed_experts is not None
            else None
        )
        # Keep one sampled assistant node per Transition.
        assistant_node = self._build_node(
            parent_id=parent_id,
            message=_canonical_message(result.assistant_message),
            token_ids=assistant_tokens,
            sampled_start=assistant_sampled_start,
            completion_logprobs=result.completion_logprobs,
            routed_experts=assistant_routed,
            staged_count=len(staged_nodes),
        )
        staged_nodes.append(assistant_node)
        assistant_node_id = assistant_node.node_id

        transition_id = len(self._transitions)
        transition = TransitionRecord(
            transition_id=transition_id,
            request_key=pending.request_key,
            tools_hash=pending.tools_hash,
            assistant_node_id=assistant_node_id,
            stop_reason=result.stop_reason,
            model=result.model,
            sampling_params_json=result.sampling_params_json,
        )

        for node in staged_nodes:
            self._nodes.append(node)
            self._children[(node.parent_id, node.message_hash)].append(node.node_id)
        self._transitions.append(transition)
        self._transition_id_by_assistant_node[assistant_node_id] = transition_id
        self._tools_by_hash.setdefault(pending.tools_hash, pending.tools)
        commit_result = CommitResult(turn_id=transition_id, assistant_node_id=assistant_node_id)
        self._commits_by_request_key[pending.request_key] = commit_result
        self._revision += 1
        return commit_result

    def _longest_message_prefix(self, messages: Sequence[Message]) -> Tuple[int, ...]:
        # Preserve token-distinct candidates with identical messages.
        candidates: List[Tuple[Optional[int], Tuple[int, ...]]] = [(None, ())]
        best: Tuple[int, ...] = ()
        for message in messages:
            message_hash = _stable_hash(message)
            next_candidates: List[Tuple[Optional[int], Tuple[int, ...]]] = []
            for parent_id, path in candidates:
                for node_id in self._children.get((parent_id, message_hash), ()):
                    if self._nodes[node_id].message == message:
                        next_path = path + (node_id,)
                        next_candidates.append((node_id, next_path))
            if not next_candidates:
                break
            candidates = next_candidates
            best = max((path for _, path in candidates), key=lambda path: path[-1])
        return best

    def _longest_exact_prefix(
        self,
        messages: Sequence[Message],
        prompt_token_ids: Sequence[int],
    ) -> Tuple[int, ...]:
        """Find the longest semantic path whose node deltas exactly prefix the prompt."""
        # Tighten semantic candidates using the rendered prompt IDs.
        candidates: List[Tuple[Optional[int], Tuple[int, ...], int]] = [(None, (), 0)]
        best: Tuple[int, ...] = ()
        for message in messages:
            message_hash = _stable_hash(message)
            next_candidates: List[Tuple[Optional[int], Tuple[int, ...], int]] = []
            for parent_id, path, offset in candidates:
                for node_id in self._children.get((parent_id, message_hash), ()):
                    node = self._nodes[node_id]
                    if node.message != message:
                        continue
                    end = offset + len(node.token_ids)
                    if tuple(prompt_token_ids[offset:end]) != node.token_ids:
                        continue
                    next_path = path + (node_id,)
                    next_candidates.append((node_id, next_path, end))
                    if len(next_path) > len(best) or (len(next_path) == len(best) and next_path[-1] > best[-1]):
                        best = next_path
            if not next_candidates:
                break
            candidates = next_candidates
        return best

    def _find_bridge_anchor(self, matched_node_ids: Sequence[int], tools_hash: str) -> Optional[BridgeAnchor]:
        # Renderer bridging starts from a completed model-call boundary.
        for message_index in range(len(matched_node_ids) - 1, -1, -1):
            node_id = matched_node_ids[message_index]
            transition_id = self._transition_id_by_assistant_node.get(node_id)
            if transition_id is None:
                continue
            transition = self._transitions[transition_id]
            if transition.tools_hash != tools_hash:
                continue
            view = TransitionView(self, transition)
            return BridgeAnchor(
                node_id=node_id,
                matched_message_count=message_index + 1,
                previous_prompt_ids=view.prompt_token_ids,
                previous_completion_ids=view.completion_ids,
            )
        return None

    def _validate_result(self, pending: PendingTurn, result: ModelTurnResult) -> None:
        if len(result.prompt_token_ids) != len(result.prompt_message_indices):
            raise ValueError("prompt token IDs and message indices must have the same length")
        if len(result.completion_ids) != len(result.completion_logprobs):
            raise ValueError("completion token IDs and logprobs must have the same length")
        if not result.completion_ids:
            raise ValueError("Cannot commit an empty completion")
        if result.routed_experts is not None and len(result.routed_experts) != (
            len(result.prompt_token_ids) + len(result.completion_ids)
        ):
            raise ValueError("Routed-expert data must align with the full prompt and completion")
        if result.reused_prefix_length < 0 or result.reused_prefix_length > len(result.prompt_token_ids):
            raise ValueError("reused_prefix_length is outside the prompt token range")
        if result.reused_prefix_length > 0:
            anchor = pending.bridge_anchor
            if anchor is None:
                raise ValueError("A reused prompt prefix requires a bridge anchor")
            expected_prefix = anchor.previous_prompt_ids + anchor.previous_completion_ids
            if result.reused_prefix_length != len(expected_prefix):
                raise ValueError("reused_prefix_length does not match the bridge anchor")
            if result.prompt_token_ids[: result.reused_prefix_length] != expected_prefix:
                raise ValueError("Rendered bridge did not preserve the exact previous prompt and completion IDs")

    def _prepare_prompt_commit(
        self,
        pending: PendingTurn,
        result: ModelTurnResult,
    ) -> Tuple[Optional[int], List[Tuple[int, ...]], Tuple[int, ...]]:
        if result.reused_prefix_length:
            if pending.bridge_anchor is None:
                raise ValueError("A reused prompt prefix requires a bridge anchor")
            prefix_node_ids = self._path_to_node(pending.bridge_anchor.node_id)
            prefix_tokens = self._tokens_for_nodes(prefix_node_ids)
            if prefix_tokens != result.prompt_token_ids[: result.reused_prefix_length]:
                raise ValueError("Trace node deltas do not match the renderer bridge prefix")
            parent_id: Optional[int] = pending.bridge_anchor.node_id
            tail_tokens = result.prompt_token_ids[result.reused_prefix_length :]
            tail_indices = result.prompt_message_indices[result.reused_prefix_length :]
            message_start = pending.bridge_anchor.matched_message_count
            message_chunks, assistant_scaffold = self._attribute_prompt_tokens(
                tail_tokens,
                tail_indices,
                message_start=message_start,
                message_count=len(pending.messages),
            )
            return parent_id, message_chunks, assistant_scaffold

        # Full renders reuse the longest candidate matching the actual prompt.
        exact_prefix = self._longest_exact_prefix(pending.messages, result.prompt_token_ids)
        parent_id = exact_prefix[-1] if exact_prefix else None
        path_len = len(self._tokens_for_nodes(exact_prefix))
        message_chunks, assistant_scaffold = self._attribute_prompt_tokens(
            result.prompt_token_ids[path_len:],
            result.prompt_message_indices[path_len:],
            message_start=len(exact_prefix),
            message_count=len(pending.messages),
        )
        return parent_id, message_chunks, assistant_scaffold

    def _attribute_prompt_tokens(
        self,
        token_ids: Sequence[int],
        message_indices: Sequence[int],
        *,
        message_start: int,
        message_count: int,
    ) -> Tuple[List[Tuple[int, ...]], Tuple[int, ...]]:
        if len(token_ids) != len(message_indices):
            raise ValueError("Token attribution arrays must have equal length")
        if message_start > message_count:
            raise ValueError("message_start cannot exceed message_count")

        owners: List[Optional[int]] = [None] * len(token_ids)
        next_owner: Optional[int] = None
        # Assign leading scaffold to the following message.
        for position in range(len(token_ids) - 1, -1, -1):
            message_index = message_indices[position]
            if message_index >= 0:
                if message_index >= message_count:
                    raise ValueError(
                        f"Renderer message index {message_index} is outside the expected range "
                        f"[{message_start}, {message_count})"
                    )
                if message_index < message_start:
                    owners[position] = next_owner
                    continue
                next_owner = message_index
                owners[position] = message_index
            elif message_index == -1:
                owners[position] = next_owner
            else:
                raise ValueError(f"Renderer message index must be -1 or non-negative, got {message_index}")

        chunks: List[List[int]] = [[] for _ in range(message_start, message_count)]
        assistant_scaffold: List[int] = []
        last_owner = message_start
        for token_id, owner in zip(token_ids, owners):
            if owner is None:
                assistant_scaffold.append(token_id)
                continue
            if owner < last_owner:
                raise ValueError("Renderer token attribution is not in message order")
            last_owner = owner
            chunks[owner - message_start].append(token_id)
        return [tuple(chunk) for chunk in chunks], tuple(assistant_scaffold)

    def _find_exact_child(
        self,
        parent_id: Optional[int],
        message: Mapping[str, Any],
        token_ids: Sequence[int],
        *,
        sampled_start: Optional[int] = None,
    ) -> Optional[int]:
        canonical = _canonical_message(message)
        message_hash = _stable_hash(canonical)
        for node_id in reversed(self._children.get((parent_id, message_hash), ())):
            node = self._nodes[node_id]
            if node.message == canonical and node.token_ids == tuple(token_ids) and node.sampled_start == sampled_start:
                return node_id
        return None

    def _build_node(
        self,
        *,
        parent_id: Optional[int],
        message: Message,
        token_ids: Sequence[int],
        sampled_start: Optional[int],
        completion_logprobs: Sequence[float],
        routed_experts: Optional[RoutedExperts],
        staged_count: int,
    ) -> MessageNode:
        canonical = _canonical_message(message)
        token_tuple = tuple(token_ids)
        if sampled_start is None:
            sampled_mask = (False,) * len(token_tuple)
            logprobs = (0.0,) * len(token_tuple)
        else:
            if sampled_start < 0 or sampled_start > len(token_tuple):
                raise ValueError("sampled_start is outside the node token range")
            sampled_length = len(token_tuple) - sampled_start
            if sampled_length != len(completion_logprobs):
                raise ValueError("Sampled token count and completion logprob count must match")
            sampled_mask = (False,) * sampled_start + (True,) * sampled_length
            logprobs = (0.0,) * sampled_start + tuple(float(value) for value in completion_logprobs)
        return MessageNode(
            node_id=len(self._nodes) + staged_count,
            parent_id=parent_id,
            message=canonical,
            message_hash=_stable_hash(canonical),
            token_ids=token_tuple,
            sampled_mask=sampled_mask,
            logprobs=logprobs,
            sampled_start=sampled_start,
            routed_experts=routed_experts,
        )

    def _path_to_node(self, node_id: int) -> Tuple[int, ...]:
        path: List[int] = []
        current: Optional[int] = node_id
        while current is not None:
            path.append(current)
            current = self._nodes[current].parent_id
        path.reverse()
        return tuple(path)

    def _tokens_for_nodes(self, node_ids: Iterable[int]) -> Tuple[int, ...]:
        tokens: List[int] = []
        for node_id in node_ids:
            tokens.extend(self._nodes[node_id].token_ids)
        return tuple(tokens)
