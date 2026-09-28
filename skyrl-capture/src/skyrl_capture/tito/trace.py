"""The token-exact message graph for one trajectory.

This is the part of the system where "close enough" is a correctness bug. The
stored token deltas are the authoritative training record, so every reuse of a
previously committed prefix is verified against the tokens that inference
actually received, and any mismatch forces a full re-render and a new branch
instead of silently mixing tokenizations.

The algorithm, and why each step exists:

1. ``prepare_turn`` matches the incoming messages against the graph in
   **message space** and finds a *bridge transition*: a completed model call
   whose assistant node terminates the matched prefix and whose ``tools_hash``
   matches. Tools participate because chat templates render tool schemas into
   the prompt, so the same messages with different tools are a different token
   sequence.
2. The renderer either bridges from that transition's exact prompt+completion
   IDs, or performs a full render.
3. ``commit`` re-checks the match in **token space**. For a bridged render the
   reused prefix must be byte-identical to the bridge's own
   prompt+completion. For a full render the longest message-space candidate
   whose node deltas exactly prefix the rendered prompt is used.
4. Prompt tokens are attributed to messages using the renderer's per-token
   indices, producing one node per message, with trailing scaffold becoming the
   assistant node's non-sampled prefix.

The trace is held in memory for the life of a trial and rebuilt from the graph
on a cache miss, so a trajectory whose trace was evicted degrades to "a new
branch" -- never to wrong tokens.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from skyrl_capture.domain.hashing import canonical_bytes, normalize_json, normalize_with_hash, stable_hash
from skyrl_capture.ids import node_id as new_node_id
from skyrl_capture.tito.compare import PrefixComparison, compare_prefix
from skyrl_capture.tito.types import (
    CommitResult,
    Message,
    ModelTurnResult,
    PendingTurn,
    RoutedExperts,
    TokenError,
    ToolSpec,
    Transition,
)

# Re-prove prefix integrity token by token on every bridged turn. Off by
# default: it is O(context) per turn and re-derives what `_validate`'s length
# check already establishes structurally. Worth turning on to answer "is the
# graph diverging from what we send?" when something has gone wrong.
#
# `report` audits without refusing: the divergence is classified and counted,
# and the turn still commits. That is for a long run where stopping is worse
# than a wrong token -- an investigation, not a deployment.
_AUDIT_SETTING = os.environ.get("TOKENS_AUDIT_PREFIX", "").lower()
AUDIT_PREFIX = _AUDIT_SETTING in {"1", "true", "yes", "report"}
# Refusing is the default, so that turning the audit on any other way -- a test
# patching `AUDIT_PREFIX`, say -- gets the strict behaviour rather than the
# lenient one. Only the explicit word `report` opts out.
AUDIT_REFUSES = _AUDIT_SETTING != "report"


logger = logging.getLogger(__name__)

ROOT_PARENT_KEY = ""


def canonical_with_hash(message: Mapping[str, Any]) -> tuple[Message, str]:
    """A message's canonical form and its hash, validated for token capture.

    The canonical form and the hash are `domain.hashing`'s, because a node's
    identity has to mean the same thing here and in text mode. What this adds
    is token mode's refusals: a value that is not JSON, or a message without a
    role, cannot be attributed exactly, and saying so here is how the turn
    fails before inference rather than after.
    """
    try:
        normalized, digest = normalize_with_hash(message)
    except TypeError as error:
        raise TokenError(f"token capture messages and tools must be JSON values: {error}") from error
    if not isinstance(normalized, dict):
        raise TokenError("message must be a mapping")
    role = normalized.get("role")
    if not isinstance(role, str) or not role:
        raise TokenError("message must contain a non-empty string role")
    return normalized, digest


def canonical_message(message: Mapping[str, Any]) -> Message:
    return canonical_with_hash(message)[0]


def canonical_tools(tools: Sequence[Mapping[str, Any]]) -> tuple[ToolSpec, ...]:
    """The tools a turn declared, canonical -- or a `TokenError`."""
    try:
        normalized = normalize_json(list(tools))
    except TypeError as error:
        raise TokenError(f"token capture messages and tools must be JSON values: {error}") from error
    if not isinstance(normalized, list):
        raise TokenError("tools must be a sequence")
    return tuple(normalized)


def delta_hash_for(message_hash: str, token_ids: Sequence[int], sampled_start: int | None) -> str:
    """Identity of a node's delta.

    Two identical messages with different tokenizations, or with a different
    sampled boundary, are different nodes. Folding tokens into the identity is
    what keeps that true (parity requirement 9).
    """
    digest = hashlib.sha256()
    digest.update(message_hash.encode())
    digest.update(b"|")
    digest.update(canonical_bytes(list(token_ids)))
    digest.update(b"|")
    digest.update(str(sampled_start).encode())
    return digest.hexdigest()


class TokenNode:
    """One message and its exact token delta."""

    __slots__ = (
        "node_id",
        "parent_node_id",
        "message",
        "message_hash",
        "delta_hash",
        "token_ids",
        "sampled_mask",
        "logprobs",
        "sampled_start",
        "routed_experts",
        "depth",
        "role",
        "author",
        "exchange_id",
        "cumulative_tokens",
        "text_segments",
        "turn_start_token",
    )

    def __init__(
        self,
        *,
        node_id: str,
        parent_node_id: str | None,
        message: Message,
        token_ids: tuple[int, ...],
        sampled_start: int | None,
        completion_logprobs: Sequence[float],
        routed_experts: RoutedExperts | None,
        depth: int,
        exchange_id: str,
        parent_cumulative_tokens: int = 0,
    ) -> None:
        canonical = canonical_message(message)
        self.node_id = node_id
        self.parent_node_id = parent_node_id
        self.message = canonical
        self.message_hash = stable_hash(canonical)
        self.token_ids = tuple(token_ids)
        self.sampled_start = sampled_start
        self.depth = depth
        self.role = str(canonical.get("role"))
        self.author = "model" if sampled_start is not None else "client"
        self.exchange_id = exchange_id
        self.routed_experts = routed_experts
        # Tokens from the root through this node. Every prefix length used to
        # cost a walk of the whole path plus a tuple copy of it, which on a
        # long context is the difference between linear and quadratic.
        self.cumulative_tokens = parent_cumulative_tokens + len(self.token_ids)

        if sampled_start is None:
            # A request/environment message introduces non-sampled tokens.
            self.sampled_mask = (False,) * len(self.token_ids)
            self.logprobs = (0.0,) * len(self.token_ids)
        else:
            if sampled_start < 0 or sampled_start > len(self.token_ids):
                raise TokenError("sampled_start is outside the node token range")
            sampled_length = len(self.token_ids) - sampled_start
            if sampled_length != len(completion_logprobs):
                raise TokenError(
                    f"sampled token count ({sampled_length}) and logprob count "
                    f"({len(completion_logprobs)}) must match"
                )
            # The assistant node holds the generation scaffold before its
            # sampled suffix, so the mask is False over the scaffold.
            self.sampled_mask = (False,) * sampled_start + (True,) * sampled_length
            self.logprobs = (0.0,) * sampled_start + tuple(float(value) for value in completion_logprobs)

        self.delta_hash = delta_hash_for(self.message_hash, self.token_ids, sampled_start)
        # Filled by `attach_text` before this node can be committed.
        self.text_segments: list[dict[str, Any]] | None = None
        self.turn_start_token: int | None = None

    def attach_text(self, decode: Any, turn_start_token: int | None) -> None:
        """Persist the exact token-to-text rendering while capturing.

        This process rendered these ids and still has the renderer loaded, so
        the decode is free here and a dependency everywhere else. A record is
        meant to be a directory you can copy to a laptop; one that can only be
        read where the model's tokenizer is installed is not that.

        The cuts are the ones a path is blocked on -- the turn marker and the
        sampled boundary -- so any block boundary downstream falls on a segment
        boundary, and a block's text is a concatenation of whole segments. IDs
        remain authoritative for training, but this representation is not
        optional: readers never load a tokenizer.
        """
        if decode is None:
            raise TokenError("TITO capture requires a renderer before committing tokens")
        cuts = {0, len(self.token_ids)}
        if turn_start_token is not None:
            cuts.update(
                index for index, token in enumerate(self.token_ids) if token == turn_start_token
            )
        if self.sampled_start is not None:
            cuts.add(self.sampled_start)
        ordered = sorted(cuts)
        self.turn_start_token = turn_start_token
        self.text_segments = [
            self._segment(decode, begin, end)
            for begin, end in zip(ordered, ordered[1:], strict=False)
        ]

    def _segment(self, decode: Any, begin: int, end: int) -> dict[str, Any]:
        """One segment's text, and where each token starts inside it.

        Decode the whole span, decode each token alone, and take prefix sums:
        token ``i`` is ``text[offsets[i]:offsets[i+1]]``, so the list is one
        longer than the token count and a reader can point at a single token
        without owning a tokenizer.

        The prefix sums are correct only while the pieces concatenate to the
        span, which is checked rather than assumed. A byte-level BPE can split
        one character's bytes across tokens -- ``' \U0001f680'`` is three
        tokens on Qwen -- and those pieces decode to U+FFFD, so their lengths
        are meaningless. `_grouped_offsets` handles that from the same pieces.

        Offsets leave here counted in UTF-16 code units, not code points. They
        exist so a reader can slice `text`, every reader that does is slicing
        a JSON string, and a JSON string is indexed in UTF-16 by the language
        that holds it. In code points an emoji -- the very character this
        function goes to such lengths over -- would shift every token after it
        by one.

        `text` is the span decode either way, so a reader always sees exactly
        what these tokens say. Refusing the turn instead, which this did
        before, threw away an exact and perfectly trainable exchange over a
        presentation detail and poisoned every turn after it.
        """
        ids = list(self.token_ids[begin:end])
        try:
            text = decode(ids)
            pieces = [decode([one]) for one in ids]
        except Exception as error:
            raise TokenError(
                f"renderer could not capture text for token range [{begin}:{end}]: {error}"
            ) from error
        if "".join(pieces) == text:
            offsets, cursor = [0], 0
            for piece in pieces:
                cursor += len(piece)
                offsets.append(cursor)
        else:
            offsets = _grouped_offsets(pieces, text)
        return {
            "start": begin,
            "end": end,
            "text": text,
            "offsets": _utf16(text, offsets),
        }

    @property
    def parent_key(self) -> str:
        return self.parent_node_id or ROOT_PARENT_KEY

    @property
    def sampled_token_count(self) -> int:
        return sum(1 for flag in self.sampled_mask if flag)

    def payload(self) -> dict[str, Any]:
        """Object-store body for this node."""
        return {
            "message": self.message,
            "role": self.role,
            "author": self.author,
            "message_hash": self.message_hash,
            "tokens": {
                "token_ids": list(self.token_ids),
                "sampled_mask": list(self.sampled_mask),
                "logprobs": list(self.logprobs),
                "sampled_start": self.sampled_start,
                "routed_experts": (
                    [[list(layer) for layer in token] for token in self.routed_experts]
                    if self.routed_experts is not None
                    else None
                ),
                # The decoded text and exact per-token character offsets. TITO
                # commits require this, so readers never need a tokenizer.
                "text_segments": self.text_segments,
                "turn_start_token": self.turn_start_token,
            },
        }

    def index_row(self) -> dict[str, Any]:
        """The node's identity and shape, as the graph records it."""
        return {
            "id": self.node_id,
            "parent_node_id": self.parent_node_id,
            "depth": self.depth,
            "role": self.role,
            "author": self.author,
            "message_hash": self.message_hash,
            "delta_hash": self.delta_hash,
            "token_count": len(self.token_ids),
            "sampled_start": self.sampled_start,
            "sampled_token_count": self.sampled_token_count,
            "has_logprobs": any(value != 0.0 for value in self.logprobs),
            "has_routed_experts": self.routed_experts is not None,
            "content_block_count": _block_count(self.message.get("content")),
            "char_count": _char_count(self.message.get("content")),
        }


def _utf16(text: str, offsets: list[int] | None) -> list[int] | None:
    """Recount code-point offsets into `text` as UTF-16 code units.

    Everything above works in code points, because that is what Python slices
    and what makes the walk readable. The wire format cannot: these offsets
    are used to slice a JSON string, and a character outside the Basic
    Multilingual Plane -- every emoji -- is one code point and two UTF-16
    code units. Converting once, here, keeps that distinction in one place
    instead of in every reader.
    """
    if offsets is None:
        return None
    prefix = [0]
    for character in text:
        prefix.append(prefix[-1] + (2 if ord(character) > 0xFFFF else 1))
    return [prefix[offset] for offset in offsets]


def _grouped_offsets(pieces: list[str], text: str) -> list[int] | None:
    """Offsets when the per-token decodes do not concatenate to `text`.

    One pass over the pieces already decoded, no further decoding. A piece
    holding U+FFFD is a fragment: the token carries part of a character whose
    remaining bytes are in its neighbours, so its own decode is undecodable
    and its length means nothing. A run of fragments ends at the next piece
    that decoded cleanly, and the text between belongs to the run.

    The run's characters go to its first token and the rest of the run gets
    none, so the list stays one longer than the token count and merely stops
    strictly increasing. Every token keeps an index, an id, a mask bit and a
    logprob. Only which token inside a shared character owns it is arbitrary,
    and there is no non-arbitrary answer to that.
    """
    offsets = [0]
    cursor = 0
    index = 0
    while index < len(pieces):
        piece = pieces[index]
        if "\ufffd" not in piece:
            cursor += len(piece)
            offsets.append(cursor)
            index += 1
            continue
        stop = index
        while stop < len(pieces) and "\ufffd" in pieces[stop]:
            stop += 1
        cursor = len(text) if stop == len(pieces) else text.find(pieces[stop], cursor)
        if cursor < 0:
            return None
        offsets.extend([cursor] * (stop - index))
        index = stop
    return offsets if cursor == len(text) else None


def _block_count(content: Any) -> int:
    if isinstance(content, list):
        return len(content)
    return 0 if content is None else 1


def _char_count(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                for key in ("text", "thinking"):
                    value = block.get(key)
                    if isinstance(value, str):
                        total += len(value)
        return total
    return 0


class TokenTrace:
    """Message graph and committed turns for one trajectory."""

    def __init__(self, trajectory_id: str) -> None:
        self.trajectory_id = trajectory_id
        self._nodes: dict[str, TokenNode] = {}
        # (parent_key, message_hash) -> node ids. A list, because several
        # token-distinct nodes may carry the same message under one parent.
        self._children: dict[tuple[str, str], list[str]] = {}
        self._transitions: list[Transition] = []
        self._transition_by_assistant: dict[str, int] = {}
        self._tools_by_hash: dict[str, tuple[ToolSpec, ...] | None] = {}
        # Materialized prompt+completion for the most recent transitions.
        # Bridging needs the previous turn's exact tokens, and rebuilding them
        # walks the whole path and copies every token -- O(context) per turn,
        # so O(context^2) over a long trajectory. A turn almost always bridges
        # off a recent transition, so a handful of entries removes that pass
        # while staying bounded: this holds a few contexts, not one per turn.
        self._recent_tokens: OrderedDict[int, tuple[tuple[int, ...], tuple[int, ...]]] = OrderedDict()
        # Per position, the last message seen there with its canonical form and
        # hash. A client must resend the whole conversation every turn and must
        # resend it verbatim for prefix reuse to engage, so without this the
        # proxy re-serialized and re-hashed the entire history on every turn --
        # O(context) per turn, O(context^2) over a session, and the largest
        # single cost at long context.
        self._canonical_cache: list[tuple[Message, Message, str]] = []
        self._revision = 0
        self._sealed = False
        # Tokens held by this trace, counting each node once. What a cache of
        # traces is actually sized by.
        self.stored_tokens = 0
        # Capture-time rendering only. The trace owns no tokenizer; the proxy
        # supplies the renderer it already used to construct the exact IDs.
        self._render_text: Any = None
        # The id the chat template opens a turn with, so a node can cut its
        # decoded text where a path would cut it.
        self._turn_start_token: int | None = None
        # Classified prefix divergences seen on this trace, by class. Only
        # populated under TOKENS_AUDIT_PREFIX, and never non-zero in a healthy
        # run -- every class here is a bug, because both sides are tokens that
        # passed through this process.
        self.audit_failures: dict[str, int] = {}
        self.last_audit_failure: str | None = None

    def set_text_renderer(self, decode: Any, turn_start_token: int | None = None) -> None:
        """Provide the capture-time rendering required by committed nodes."""
        self._render_text = decode
        self._turn_start_token = turn_start_token

    _RECENT_TOKEN_CACHE = 2

    def _remember_tokens(
        self, transition_id: int, prompt_ids: tuple[int, ...], completion_ids: tuple[int, ...]
    ) -> None:
        self._recent_tokens[transition_id] = (prompt_ids, completion_ids)
        self._recent_tokens.move_to_end(transition_id)
        while len(self._recent_tokens) > self._RECENT_TOKEN_CACHE:
            self._recent_tokens.popitem(last=False)

    # -- introspection -----------------------------------------------------
    @property
    def revision(self) -> int:
        return self._revision

    @property
    def sealed(self) -> bool:
        return self._sealed

    def seal(self) -> None:
        self._sealed = True

    def node(self, node_id: str) -> TokenNode:
        return self._nodes[node_id]

    def nodes(self) -> list[TokenNode]:
        return list(self._nodes.values())

    def transition(self, transition_id: int) -> Transition:
        return self._transitions[transition_id]

    def leaves(self) -> list[str]:
        parents = {node.parent_node_id for node in self._nodes.values() if node.parent_node_id}
        return [node_id for node_id in self._nodes if node_id not in parents]

    def register(self, node: TokenNode) -> None:
        """Insert a node without validation. Used when rebuilding from the graph."""
        self._nodes[node.node_id] = node
        self._children.setdefault((node.parent_key, node.message_hash), []).append(node.node_id)
        self.stored_tokens += len(node.token_ids)

    def register_transition(self, transition: Transition) -> None:
        self._transitions.append(transition)
        self._transition_by_assistant[transition.assistant_node_id] = transition.transition_id
        self._tools_by_hash.setdefault(transition.tools_hash, None)

    # -- path helpers ------------------------------------------------------
    def path_to(self, node_id: str) -> tuple[str, ...]:
        path: list[str] = []
        current: str | None = node_id
        while current is not None:
            path.append(current)
            current = self._nodes[current].parent_node_id
        path.reverse()
        return tuple(path)

    def tokens_for(self, node_ids: Iterable[str]) -> tuple[int, ...]:
        tokens: list[int] = []
        for node_id in node_ids:
            tokens.extend(self._nodes[node_id].token_ids)
        return tuple(tokens)

    def transition_prompt_ids(self, transition_id: int) -> tuple[int, ...]:
        """The exact prompt IDs a committed transition sent to inference."""
        cached = self._recent_tokens.get(transition_id)
        if cached is not None:
            self._recent_tokens.move_to_end(transition_id)
            return cached[0]
        prompt_ids, completion_ids = self._materialize_transition(transition_id)
        self._remember_tokens(transition_id, prompt_ids, completion_ids)
        return prompt_ids

    def transition_completion_ids(self, transition_id: int) -> tuple[int, ...]:
        cached = self._recent_tokens.get(transition_id)
        if cached is not None:
            self._recent_tokens.move_to_end(transition_id)
            return cached[1]
        # The completion is the assistant node's own tail, so this needs no
        # walk even on a miss.
        transition = self._transitions[transition_id]
        assistant = self._nodes[transition.assistant_node_id]
        if assistant.sampled_start is None:
            raise TokenError("transition assistant node has no sampled boundary")
        return assistant.token_ids[assistant.sampled_start :]

    def compare_prefix(
        self, node_ids: Sequence[str], prompt_token_ids: Sequence[int], reused: int
    ) -> PrefixComparison:
        """Do the stored node deltas reproduce the prompt's reused prefix?

        Only run under ``TOKENS_AUDIT_PREFIX``. Prefix integrity is structural
        on the normal path -- see ``_validate`` -- and proving it again costs a
        pass over the whole conversation per turn. This is the answer to "is the
        graph diverging from what we send?" when something has gone wrong, and
        the tests use it to hold the structural claim honest.

        It returns a classified report rather than a boolean, because "it
        diverged" is detectable and not diagnosable: the class says which part
        of the machinery to look at, and the offset says where.
        """
        return compare_prefix(
            [self._nodes[node_id] for node_id in node_ids],
            prompt_token_ids,
            reused,
            decode=self._render_text,
        )

    def _materialize_transition(self, transition_id: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Rebuild a transition's prompt and completion by walking its path.

        Only reached on a cache miss -- a rehydrated trace, or a bridge off
        something older than the last few turns.
        """
        transition = self._transitions[transition_id]
        assistant = self._nodes[transition.assistant_node_id]
        if assistant.sampled_start is None:
            raise TokenError("transition assistant node has no sampled boundary")
        path = self.path_to(assistant.node_id)
        tokens = list(self.tokens_for(path[:-1]))
        tokens.extend(assistant.token_ids[: assistant.sampled_start])
        return tuple(tokens), assistant.token_ids[assistant.sampled_start :]

    # -- turn preparation --------------------------------------------------
    def _canonical_pairs(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> list[tuple[Message, str]]:
        """Canonical form and hash per message, reusing unchanged positions.

        A dict comparison settles whether position `i` still holds what it held
        last turn; canonicalizing it again means a JSON round trip and a SHA-256
        over the message. The first position that differs invalidates the rest,
        because a changed message changes every hash that follows it.
        """
        cache = self._canonical_cache
        pairs: list[tuple[Message, str]] = []
        for index, message in enumerate(messages):
            if index < len(cache):
                seen, canonical, digest = cache[index]
                if message == seen:
                    pairs.append((canonical, digest))
                    continue
                del cache[index:]
            canonical, digest = canonical_with_hash(message)
            cache.append((dict(message), canonical, digest))
            pairs.append((canonical, digest))
        del cache[len(messages) :]
        return pairs

    def prepare_turn(
        self, messages: Sequence[Mapping[str, Any]], *, tools: Sequence[Mapping[str, Any]] | None = None
    ) -> PendingTurn:
        if self._sealed:
            raise TokenError("cannot prepare a turn on a sealed trace")
        if not messages:
            raise TokenError("a chat completion request must contain at least one message")
        canonical_pairs = self._canonical_pairs(messages)
        canonical = tuple(item[0] for item in canonical_pairs)
        hashes = tuple(item[1] for item in canonical_pairs)
        declared = canonical_tools(tools) if tools is not None else None
        tools_hash = stable_hash(list(declared) if declared is not None else [])

        matched = self._longest_message_prefix(canonical, hashes)
        bridge = self._find_bridge_transition(matched, tools_hash)
        return PendingTurn(
            revision=self._revision,
            messages=canonical,
            tools=declared,
            tools_hash=tools_hash,
            bridge_transition_id=bridge,
            matched_node_ids=matched,
        )

    def _longest_message_prefix(
        self, messages: Sequence[Message], hashes: Sequence[str] | None = None
    ) -> tuple[str, ...]:
        """Longest message-space prefix, keeping token-distinct candidates alive.

        ``hashes`` are the callers' already-computed message hashes; recomputing
        them here doubled the serialization work on every turn.
        """
        if hashes is None:
            hashes = [stable_hash(message) for message in messages]
        candidates: list[tuple[str, tuple[str, ...]]] = [(ROOT_PARENT_KEY, ())]
        best: tuple[str, ...] = ()
        for message, message_hash in zip(messages, hashes, strict=True):
            next_candidates: list[tuple[str, tuple[str, ...]]] = []
            for parent_key, path in candidates:
                for node_id in self._children.get((parent_key, message_hash), ()):
                    if self._nodes[node_id].message != message:
                        continue
                    next_candidates.append((node_id, path + (node_id,)))
            if not next_candidates:
                break
            candidates = next_candidates
            # Prefer the most recently created candidate at equal length; node
            # IDs sort by creation time.
            best = max((path for _key, path in candidates), key=lambda path: path[-1])
        return best

    def _longest_exact_prefix(
        self, messages: Sequence[Message], prompt_token_ids: Sequence[int]
    ) -> tuple[str, ...]:
        """Tighten the message-space match using the actual rendered prompt."""
        candidates: list[tuple[str, tuple[str, ...], int]] = [(ROOT_PARENT_KEY, (), 0)]
        best: tuple[str, ...] = ()
        for message in messages:
            message_hash = stable_hash(message)
            next_candidates: list[tuple[str, tuple[str, ...], int]] = []
            for parent_key, path, offset in candidates:
                for node_id in self._children.get((parent_key, message_hash), ()):
                    node = self._nodes[node_id]
                    if node.message != message:
                        continue
                    end = offset + len(node.token_ids)
                    if tuple(prompt_token_ids[offset:end]) != node.token_ids:
                        continue
                    next_path = path + (node_id,)
                    next_candidates.append((node_id, next_path, end))
                    if len(next_path) > len(best) or (
                        len(next_path) == len(best) and next_path[-1] > best[-1]
                    ):
                        best = next_path
            if not next_candidates:
                break
            candidates = next_candidates
        return best

    def _find_bridge_transition(self, matched_node_ids: Sequence[str], tools_hash: str) -> int | None:
        """The most recent completed model call the renderer may bridge from.

        Reuse must resume at a real inference boundary -- an assistant node that
        terminates a transition -- and the tools hash must match, or the
        re-rendered prompt would differ from the stored tokens.
        """
        for node_id in reversed(list(matched_node_ids)):
            transition_id = self._transition_by_assistant.get(node_id)
            if transition_id is None:
                continue
            if self._transitions[transition_id].tools_hash != tools_hash:
                continue
            return transition_id
        return None

    # -- commit ------------------------------------------------------------
    def commit(self, pending: PendingTurn, result: ModelTurnResult, *, exchange_id: str) -> CommitResult:
        """Atomically add one exact inference result to the graph."""
        if self._sealed:
            raise TokenError("cannot commit to a sealed trace")
        if pending.revision != self._revision:
            raise TokenError(
                f"stale pending turn: prepared at revision {pending.revision}, "
                f"trace is at revision {self._revision}"
            )
        self._validate(pending, result)

        parent_id, message_chunks, assistant_scaffold, message_start = self._plan_commit(pending, result)

        staged: list[TokenNode] = []
        reused: list[str] = []
        routed = result.routed_experts
        # Also the parent's cumulative token count, which is why it is read
        # off the node rather than recomputed by walking the path.
        routed_cursor = self._nodes[parent_id].cumulative_tokens if parent_id is not None else 0
        depth = self._nodes[parent_id].depth + 1 if parent_id is not None else 0

        for offset, token_ids in enumerate(message_chunks):
            message = pending.messages[message_start + offset]
            existing = self._find_exact_child(parent_id, message, token_ids, sampled_start=None)
            if existing is not None:
                # An identical child already exists: reuse it rather than
                # duplicating it. This is what makes a retry idempotent.
                reused.append(existing)
                parent_id = existing
                routed_cursor += len(token_ids)
                depth = self._nodes[existing].depth + 1
                continue
            node = TokenNode(
                node_id=new_node_id(),
                parent_node_id=parent_id,
                message=message,
                token_ids=token_ids,
                sampled_start=None,
                completion_logprobs=(),
                routed_experts=(
                    routed[routed_cursor : routed_cursor + len(token_ids)] if routed is not None else None
                ),
                depth=depth,
                exchange_id=exchange_id,
                parent_cumulative_tokens=routed_cursor,
            )
            node.attach_text(self._render_text, self._turn_start_token)
            staged.append(node)
            parent_id = node.node_id
            routed_cursor += len(token_ids)
            depth += 1

        assistant_tokens = tuple(assistant_scaffold) + result.completion_ids
        assistant = TokenNode(
            node_id=new_node_id(),
            parent_node_id=parent_id,
            message=result.assistant_message,
            token_ids=assistant_tokens,
            sampled_start=len(assistant_scaffold),
            completion_logprobs=result.completion_logprobs,
            routed_experts=(
                routed[routed_cursor : routed_cursor + len(assistant_tokens)]
                if routed is not None
                else None
            ),
            depth=depth,
            exchange_id=exchange_id,
            parent_cumulative_tokens=routed_cursor,
        )
        assistant.attach_text(self._render_text, self._turn_start_token)
        existing_assistant = self._find_exact_child(
            parent_id, result.assistant_message, assistant_tokens, sampled_start=len(assistant_scaffold)
        )
        if existing_assistant is not None:
            assistant_node_id = existing_assistant
            reused.append(existing_assistant)
        else:
            staged.append(assistant)
            assistant_node_id = assistant.node_id

        parent_output_node_id = None
        input_leaf = parent_id
        if parent_id is not None and self._nodes.get(parent_id) is not None:
            candidate = self._nodes[parent_id]
            if candidate.author == "model" and candidate.node_id in self._transition_by_assistant:
                parent_output_node_id = candidate.node_id
        if pending.bridge_transition_id is not None and result.reused_prefix_length:
            parent_output_node_id = self._transitions[pending.bridge_transition_id].assistant_node_id

        # Nothing is published until every attribution check has passed.
        for node in staged:
            self.register(node)

        transition = Transition(
            transition_id=len(self._transitions),
            assistant_node_id=assistant_node_id,
            tools_hash=pending.tools_hash,
            stop_reason=result.stop_reason,
            model=result.model,
            sampling_params=result.sampling_params,
            prompt_token_count=len(result.prompt_token_ids),
            completion_token_count=len(result.completion_ids),
        )
        self.register_transition(transition)
        # The next turn's bridge wants exactly these two, and they are already
        # in hand. Caching them here is what keeps a long trajectory linear.
        self._remember_tokens(
            transition.transition_id, result.prompt_token_ids, result.completion_ids
        )
        self._tools_by_hash[pending.tools_hash] = pending.tools
        self._revision += 1

        return CommitResult(
            transition_id=transition.transition_id,
            assistant_node_id=assistant_node_id,
            created_node_ids=tuple(node.node_id for node in staged),
            reused_node_ids=tuple(reused),
            input_leaf_node_id=input_leaf,
            parent_output_node_id=parent_output_node_id,
            matched_message_count=len(pending.matched_node_ids),
            reused_prefix_length=result.reused_prefix_length,
            branched=len(pending.matched_node_ids) < len(pending.messages)
            and len(pending.matched_node_ids) > 0,
        )

    def _validate(self, pending: PendingTurn, result: ModelTurnResult) -> None:
        if result.reused_prefix_length < 0 or result.reused_prefix_length > len(result.prompt_token_ids):
            raise TokenError("reused_prefix_length is outside the prompt token range")
        if len(result.prompt_message_indices) != len(result.prompt_token_ids) - result.reused_prefix_length:
            raise TokenError(
                "prompt message indices must cover the prompt tokens after the reused prefix"
            )
        if len(result.completion_ids) != len(result.completion_logprobs):
            raise TokenError("completion IDs and logprobs must have the same length")
        if not result.completion_ids:
            raise TokenError("cannot commit an empty completion")
        if result.routed_experts is not None and len(result.routed_experts) != (
            len(result.prompt_token_ids) + len(result.completion_ids)
        ):
            raise TokenError("routed-expert data must cover the full prompt and completion")
        if result.reused_prefix_length:
            if pending.bridge_transition_id is None:
                raise TokenError("a reused prompt prefix requires a bridge transition")
            transition = self._transitions[pending.bridge_transition_id]
            expected_length = transition.prompt_token_count + transition.completion_token_count
            if result.reused_prefix_length != expected_length:
                raise TokenError("reused_prefix_length does not match the bridge transition")
            # This length check is the whole prefix check. The tokens behind it
            # are the ones these nodes were committed from -- handed to the
            # bridge and returned unchanged -- so agreeing on where the prefix
            # ends is agreeing on the prefix. Comparing it token by token walked
            # the whole conversation every turn to re-derive that.
            if AUDIT_PREFIX:
                prefix_node_ids = self.path_to(transition.assistant_node_id)
                report = self.compare_prefix(
                    prefix_node_ids, result.prompt_token_ids, result.reused_prefix_length
                )
                if not report.ok:
                    self.audit_failures[report.kind or "unknown"] = (
                        self.audit_failures.get(report.kind or "unknown", 0) + 1
                    )
                    self.last_audit_failure = report.describe()
                    if AUDIT_REFUSES:
                        raise TokenError(
                            "stored node deltas do not match the renderer bridge prefix: "
                            + report.describe()
                        )
                    logger.error(
                        "prefix audit failed for %s: %s", self.trajectory_id, report.describe()
                    )

    def _plan_commit(
        self, pending: PendingTurn, result: ModelTurnResult
    ) -> tuple[str | None, list[tuple[int, ...]], tuple[int, ...], int]:
        if result.reused_prefix_length:
            assert pending.bridge_transition_id is not None
            transition = self._transitions[pending.bridge_transition_id]
            prefix_node_ids = self.path_to(transition.assistant_node_id)
            # The prefix needs no checking here. It is not something a renderer
            # handed us to be trusted: it is the tokens these very nodes were
            # committed from, passed to the bridge and returned unchanged under
            # a contract the library refuses rather than breaks, with the
            # boundary probed where a trim would move it. Re-reading it cost a
            # pass over the whole conversation on every turn to re-derive what
            # the previous turn already established.
            tail_tokens = result.prompt_token_ids[result.reused_prefix_length :]
            tail_indices = result.prompt_message_indices
            message_start = len(prefix_node_ids)
            chunks, scaffold = attribute_prompt_tokens(
                tail_tokens,
                tail_indices,
                message_start=message_start,
                message_count=len(pending.messages),
            )
            return transition.assistant_node_id, chunks, scaffold, message_start

        # A full render reuses the longest candidate that matches the actual
        # rendered prompt token-for-token.
        exact = self._longest_exact_prefix(pending.messages, result.prompt_token_ids)
        parent_id = exact[-1] if exact else None
        consumed = self._nodes[exact[-1]].cumulative_tokens if exact else 0
        chunks, scaffold = attribute_prompt_tokens(
            result.prompt_token_ids[consumed:],
            result.prompt_message_indices[consumed:],
            message_start=len(exact),
            message_count=len(pending.messages),
        )
        return parent_id, chunks, scaffold, len(exact)

    def _find_exact_child(
        self,
        parent_id: str | None,
        message: Mapping[str, Any],
        token_ids: Sequence[int],
        *,
        sampled_start: int | None,
    ) -> str | None:
        canonical = canonical_message(message)
        message_hash = stable_hash(canonical)
        parent_key = parent_id or ROOT_PARENT_KEY
        for node_id in reversed(self._children.get((parent_key, message_hash), ())):
            node = self._nodes[node_id]
            if (
                node.message == canonical
                and node.token_ids == tuple(token_ids)
                and node.sampled_start == sampled_start
            ):
                return node_id
        return None


def attribute_prompt_tokens(
    token_ids: Sequence[int],
    message_indices: Sequence[int],
    *,
    message_start: int,
    message_count: int,
) -> tuple[list[tuple[int, ...]], tuple[int, ...]]:
    """Split rendered prompt tokens into per-message deltas.

    Scaffold tokens (index ``-1``) are attributed to the message that *follows*
    them, because a chat template emits a role header before the message body.
    Scaffold after the last message belongs to no message: that is the
    generation prompt, and it becomes the assistant node's non-sampled prefix.

    Walking backwards is what makes "the following message" cheap to determine.
    """
    if len(token_ids) != len(message_indices):
        raise TokenError("token attribution arrays must have equal length")
    if message_start > message_count:
        raise TokenError("message_start cannot exceed message_count")

    owners: list[int | None] = [None] * len(token_ids)
    next_owner: int | None = None
    for position in range(len(token_ids) - 1, -1, -1):
        index = message_indices[position]
        if index >= 0:
            if index >= message_count:
                raise TokenError(
                    f"renderer message index {index} is outside the expected range "
                    f"[{message_start}, {message_count})"
                )
            if index < message_start:
                # Belongs to an already-committed message; treat as scaffold for
                # the next message in the tail.
                owners[position] = next_owner
                continue
            next_owner = index
            owners[position] = index
        elif index == -1:
            owners[position] = next_owner
        else:
            raise TokenError(f"renderer message index must be -1 or non-negative, got {index}")

    chunks: list[list[int]] = [[] for _ in range(message_start, message_count)]
    scaffold: list[int] = []
    last_owner = message_start
    # strict: an owner per token is an invariant, not a coincidence.
    for token_id, owner in zip(token_ids, owners, strict=True):
        if owner is None:
            scaffold.append(token_id)
            continue
        if owner < last_owner:
            raise TokenError("renderer token attribution is not in message order")
        last_owner = owner
        chunks[owner - message_start].append(token_id)
    return [tuple(chunk) for chunk in chunks], tuple(scaffold)


def convert_routed_experts(value: Any) -> RoutedExperts | None:
    if value is None:
        return None
    return tuple(
        tuple(tuple(int(expert) for expert in layer) for layer in token) for token in value
    )
