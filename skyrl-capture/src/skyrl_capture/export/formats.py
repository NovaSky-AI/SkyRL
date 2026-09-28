"""The four export formats, one per use case.

* ``graph`` -- one line per trace: metadata plus the node tree. The lossless
  archive everything else is a projection of.
* ``replay`` -- one line per session: arrival offset, then turns carrying the
  verbatim request payload. Requests only; replay does not need responses.
* ``text_samples`` -- one row per root-to-leaf path, for distillation and
  speculative-decoding training.
* ``token_samples`` -- the same envelope with exact token IDs and a loss mask,
  for RL on a token-in/token-out policy.

Exports are deterministic and versioned: given the same selected snapshot, the
same bytes come out.

Two rules hold across the sample formats and are easy to get wrong:

**Rows are never dropped.** ``trainable`` is the only filtering mechanism, so
the row count means the same thing under every flag combination and an
abandoned branch stays visible for mismatch analysis rather than vanishing.

**``abandoned`` means the path's leaf is a model output whose parent has other
children** -- not merely a childless leaf, which every path has. One
definition covers both the discarded original of a repair and the unpicked
samples of a best-of-N.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from skyrl_capture.domain.graph import GraphNode
from skyrl_capture.export.view import TrajectoryView
from skyrl_capture.version import DERIVATION_VERSION, SCHEMA_VERSION


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _timestamp(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _trajectory_header(view: TrajectoryView) -> dict[str, Any]:
    trajectory = view.trajectory
    capture = trajectory.public()["capture"]
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": trajectory.id,
        "project": trajectory.project,
        # Where the row came from. A training row that cannot say which run and
        # step produced it cannot be compared against another step's.
        "run_id": trajectory.run_id,
        "task_id": trajectory.task_id,
        "step": trajectory.step,
        "mode": trajectory.mode,
        "status": trajectory.status,
        "command_result": trajectory.command_result,
        "labels": trajectory.labels,
        "annotations": trajectory.annotations,
        "upstream": {
            key: trajectory.upstream_snapshot.get(key)
            for key in ("type", "url", "model", "tokenizer")
        },
        "integrity": {
            "complete": capture["complete"],
            "calls_missing": capture["calls_missing"],
            "calls_after_close": capture["calls_after_close"],
        },
        "bodies": trajectory.bodies,
        "created_at": _timestamp(trajectory.created_at),
        "finished_at": _timestamp(trajectory.finished_at),
    }


# -- graph -----------------------------------------------------------------
def _call_block(view: TrajectoryView, node: GraphNode) -> dict[str, Any] | None:
    """The model call this node was the output of, if it was one."""
    exchange = view.exchange_by_id.get(node.exchange_id)
    if exchange is None or exchange["output_node_id"] != node.id:
        return None
    block = {
        "endpoint": exchange["path"],
        "model": exchange["model"],
        "status": exchange["http_status"],
        "streaming": exchange["streaming"],
        "finish_reason": exchange["completion_reason"],
        "tools_hash": exchange["tools_hash"],
        "sampling": _jsonable(exchange["sampling"] or {}),
        "usage": _jsonable(exchange["usage"]),
        "timing": {
            "started_at": _timestamp(exchange["request_start_at"]),
            "ended_at": _timestamp(exchange["response_end_at"]),
            "duration_ms": exchange["duration_ms"],
            "ttft_ms": exchange["ttft_ms"],
        },
    }
    if exchange["max_output_tokens"] is not None:
        block["sampling"]["max_tokens"] = exchange["max_output_tokens"]
    if exchange["transport_error"] or exchange["provider_error"]:
        block["error"] = {
            "transport": exchange["transport_error"],
            "provider": _jsonable(exchange["provider_error"]),
        }
    return block


def graph_records(view: TrajectoryView) -> Iterator[dict[str, Any]]:
    """One line: the trajectory, then its node tree.

    An exchange is not a record type. A model call produces exactly one model
    node, so the call attaches to that node -- and a node with more than one
    entry in ``children`` is a fork, which is the whole representation of
    branching.
    """
    nodes: list[dict[str, Any]] = []
    for node_id in view.node_order:
        node = view.nodes[node_id]
        record: dict[str, Any] = {
            "node_id": node.id,
            "parent": node.parent_id,
            "children": list(view.children.get(node.id, [])),
            "role": node.role,
            "author": node.author,
            "message": node.message,
            "message_hash": node.message_hash,
            "context_hash": node.context_hash,
        }
        if node.token_ids:
            record["tokens"] = {
                "ids": node.token_ids,
                "sampled_start": node.sampled_start,
                "sampled_mask": node.sampled_mask,
                "logprobs": node.logprobs,
                "routed_experts": node.routed_experts,
            }
        call = _call_block(view, node)
        if call is not None:
            record["call"] = call
        if node.id in view.uncertain_node_ids:
            # Shown, and labelled. This turn's tokens are exact; what is not
            # known is whether the client received the response, and a graph
            # that quietly dropped it would hide a branch that exists.
            record["delivery_uncertain"] = True
        nodes.append(record)

    yield {**_trajectory_header(view), "derivation_version": DERIVATION_VERSION, "nodes": nodes}


# -- replay ----------------------------------------------------------------
def _linear_segments(view: TrajectoryView) -> list[list[str]]:
    """Maximal linear runs of exchanges, split at every fork.

    An exchange whose delivery was never confirmed ends the segment it is in
    and contributes no segment of its own: a replay reproduces traffic a client
    is known to have received, and reissuing a call whose response may never
    have arrived would replay a conversation that may never have happened.
    """
    segments: list[list[str]] = []
    uncertain = view.uncertain_exchange_ids
    for root in view.exchange_roots:
        if root in uncertain:
            continue
        stack = [[root]]
        while stack:
            segment = stack.pop()
            while True:
                children = [
                    child
                    for child in view.exchange_children.get(segment[-1], [])
                    if child not in uncertain
                ]
                if len(children) == 1:
                    segment.append(children[0])
                    continue
                segments.append(segment)
                for child in children:
                    stack.append([child])
                break
    segments.sort(key=lambda item: (view.exchange_by_id[item[0]]["sequence"], item[0]))
    return segments


def _payload_for(view: TrajectoryView, exchange: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": exchange["model"], "messages": messages}
    if exchange["max_output_tokens"] is not None:
        payload["max_tokens"] = exchange["max_output_tokens"]
    if exchange["tools"]:
        payload["tools"] = _jsonable(exchange["tools"])
    payload.update(_jsonable(exchange["sampling"] or {}))
    return payload


def replay_records(view: TrajectoryView, *, origin: Any = None) -> list[dict[str, Any]]:
    """One line per session, for replaying captured traffic at a scale factor.

    A **session** is a maximal linear run of calls between forks: in
    ``A -> B -> C1 -> D1`` with a second branch ``B -> C2 -> D2``, the sessions
    are ``[A, B]``, ``[C1, D1]`` and ``[C2, D2]``. A trajectory yields more
    than one only when it branches.

    Scheduling is relative, not absolute, because absolute offsets bake one
    rollout's latencies into the artifact:

    * a session with a ``parent_session_id`` starts ``delay_ms`` after that
      parent's **last turn ends**, so a replay against a slower or faster model
      still issues it after the work it continues from;
    * a session without one starts ``arrival_ms`` after the first call in the
      export -- which is what places several trajectories on one timeline.

    ``arrival_ms`` is kept for every session regardless, so an as-observed
    replay that ignores parentage is still possible.

    **Known limitation.** Parentage here is the graph's, which is exact prefix
    evidence and nothing more. Where an agent fans out and then folds the
    results back in, the fan-in is invisible: the synthesizing call looks like
    another child of the fork, and its ``delay_ms`` silently includes however
    long the siblings happened to take in this rollout. Replaying that against
    a much slower model can therefore issue it earlier than its inputs would be
    ready. Inferring the join would mean guessing at an execution graph the
    proxy cannot see, which this system deliberately does not do.
    """
    if not view.exchanges:
        return []

    if origin is None:
        origin = min(exchange["request_start_at"] for exchange in view.exchanges)
    chains = _linear_segments(view)
    if not chains:
        return []
    session_of = {
        exchange_id: f"{view.trajectory.id}-s{index:04d}"
        for index, chain in enumerate(chains)
        for exchange_id in chain
    }
    # When each session's last turn finished, so a child can be scheduled from
    # its parent's end rather than from the clock.
    ends = {
        session_of[chain[-1]]: view.exchange_by_id[chain[-1]]["response_end_at"] for chain in chains
    }

    records: list[dict[str, Any]] = []
    for chain in chains:
        turns: list[dict[str, Any]] = []
        for position, exchange_id in enumerate(chain):
            exchange = view.exchange_by_id[exchange_id]
            first_in_session = position == 0
            # A session root replays the complete context it observed; its
            # parent, if any, is a different session.
            messages = (
                view.exchange_path_messages(exchange)
                if first_in_session
                else view.exchange_tail_messages(exchange)
            )
            turns.append(
                {
                    "delay_ms": _delay_before(view, exchange, first_in_session=first_in_session),
                    "method": exchange["method"],
                    "endpoint": exchange["path"],
                    "payload": _payload_for(view, exchange, messages),
                }
            )
        head = view.exchange_by_id[chain[0]]
        # Every field is present on every record, absence written as null or an
        # empty list rather than a missing key -- the same convention the turn
        # level uses, so a consumer never has to test for one.
        parent = view.exchange_parent.get(chain[0])
        parent_session = session_of[parent] if parent is not None else None
        delay: int | None = None
        if parent_session is not None:
            parent_end = ends.get(parent_session)
            if parent_end is not None:
                delay = int(
                    max(0.0, (head["request_start_at"] - parent_end).total_seconds() * 1000.0)
                )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "session_id": session_of[chain[0]],
                "trajectory_id": view.trajectory.id,
                "parent_session_id": parent_session,
                "delay_ms": delay,
                "arrival_ms": int(
                    max(0.0, (head["request_start_at"] - origin).total_seconds() * 1000.0)
                ),
                "forks": [
                    session_of[child]
                    for child in view.exchange_children.get(chain[-1], [])
                    if child in session_of
                ],
                "turns": turns,
            }
        )
    return records


def _delay_before(view: TrajectoryView, exchange: Any, *, first_in_session: bool) -> int | None:
    """Observed wait before this call, clamped to zero for replay.

    Measured from the *parent* exchange, never from whichever call happened to
    arrive before it: once a trajectory branches the previous arrival is a
    sibling. Capture keeps the signed value; a replayer needs it non-negative,
    and this is the only place it is clamped.
    """
    if first_in_session:
        return None
    parent_id = view.exchange_parent.get(exchange["id"])
    if parent_id is None:
        return None
    parent = view.exchange_by_id[parent_id]
    if parent["response_end_at"] is None:
        return None
    gap = (exchange["request_start_at"] - parent["response_end_at"]).total_seconds() * 1000.0
    return int(max(0.0, gap))


# -- samples ---------------------------------------------------------------
def _abandoned(view: TrajectoryView, leaf: GraphNode) -> bool:
    """Is this path one of several endings from the same point?

    A childless leaf is every path's ending, so that alone marks nothing. What
    this detects is a leaf the model produced whose parent has other children:
    the trajectory forked just before the end, and this is one of the results.

    It does *not* identify a branch that lost. Capture sees that two
    continuations came from one context and has no evidence of which the agent
    went on to use, so in a best-of-N every sample is abandoned, the kept one
    included. A repair is the asymmetric case: the model's original is
    abandoned and the harness's corrected version continues past it.
    """
    if leaf.author != "model":
        return False
    siblings = view.children.get(leaf.parent_id, [])
    return len(siblings) > 1


def _sample_conditions(view: TrajectoryView, path: list[GraphNode]) -> dict[str, Any]:
    """What the sampled messages on this path were produced under.

    **Tools**, because a chat template renders their schemas into the prompt:
    the same messages under a different tool set are a different data point.
    They are uniform along a path by construction -- tools are part of node
    identity, so a call that changed them cannot reuse the prefix and starts
    its own branch instead.

    **Model**, because it produced these generations. It is part of node
    identity too, so a path has exactly one -- a workload that switches model
    mid-run starts a new root rather than continuing, and its samples are
    separate rows that a consumer can filter by model.

    Sampling parameters are deliberately absent. They decide what a *replay*
    sends, and `replay` carries them; a training row is the text and the
    conditions that shaped the prompt, and temperature shaped neither.
    """
    tools: Any = None
    models: set[Any] = set()
    for node in path:
        if node.author != "model":
            continue
        exchange = view.exchange_by_id.get(node.exchange_id)
        if exchange is None or exchange["output_node_id"] != node.id:
            continue
        models.add(exchange["model"])
        tools = exchange["tools"]
    # One element by construction: both tools and model are part of node
    # identity, so a path cannot span two of either.
    return {
        "model": models.pop() if len(models) == 1 else None,
        "tools": _jsonable(tools),
    }


def _sample_envelope(
    view: TrajectoryView, index: int, path: list[GraphNode], *, abandoned: bool
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "path_id": f"{view.trajectory.id}-p{index:04d}",
        "trajectory_id": view.trajectory.id,
        "node_ids": [node.id for node in path],
        "abandoned": abandoned,
        "labels": view.trajectory.labels,
        "annotations": view.trajectory.annotations,
        **_sample_conditions(view, path),
    }


def _paths(view: TrajectoryView) -> list[list[GraphNode]]:
    return [view.path_to(leaf_id) for leaf_id in view.leaves()]


def text_sample_records(
    view: TrajectoryView, *, allow_repeated_targets: bool = False, mask_abandoned: bool = False
) -> list[dict[str, Any]]:
    """One row per root-to-leaf conversation.

    Rows are never dropped. ``trainable`` carries every filtering decision, so
    a fully masked row remains a legible record of a branch the agent walked
    away from.

    **A sampled message is a training target in exactly one row by default.**
    Branches share their ancestors, so without that the orchestrator's reply in
    a four-way fan-out would be a target four times, weighting one generation as
    if the model had produced it four times. ``allow_repeated_targets`` opts
    into the other behaviour, which is a deliberate choice rather than a
    default worth having.
    """
    rows: list[dict[str, Any]] = []
    claimed: set[str] = set()
    for index, path in enumerate(_paths(view)):
        if not path:
            continue
        abandoned = _abandoned(view, path[-1])
        messages: list[dict[str, Any]] = []
        reasons: set[str] = set()
        for node in path:
            trainable = node.author == "model"
            if trainable and node.id in view.uncertain_node_ids:
                # Exact, and possibly never delivered. Training on it would
                # teach the model from a turn the agent may never have seen.
                trainable = False
                reasons.add("delivery_uncertain")
            if trainable and not allow_repeated_targets and node.id in claimed:
                trainable = False
                reasons.add("repeated_target")
            if trainable and mask_abandoned and abandoned:
                trainable = False
                reasons.add("abandoned")
            if trainable:
                claimed.add(node.id)
            # `message` is the message as captured -- role, content, and
            # anything structured like tool_calls. Repeating role and content
            # beside it would be a lossy copy of a field that is already there.
            messages.append(
                {
                    "node_id": node.id,
                    "author": node.author,
                    "trainable": trainable,
                    "message": node.message,
                }
            )
        row = _sample_envelope(view, index, path, abandoned=abandoned)
        row["trainable_count"] = sum(1 for item in messages if item["trainable"])
        if row["trainable_count"] == 0 and reasons:
            row["masked_reason"] = sorted(reasons)[0]
        row["messages"] = messages
        rows.append(row)
    return rows


def token_sample_records(
    view: TrajectoryView,
    *,
    allow_repeated_targets: bool = False,
    mask_abandoned: bool = False,
    overlong_filtering: bool = False,
) -> list[dict[str, Any]]:
    """One row per root-to-leaf path, in token space.

    Each sampled node is trainable at most once unless
    ``allow_repeated_targets`` is set -- parity requirement 12, since forks
    share their ancestor assistant nodes and training them once per branch
    double-counts the same sampled tokens.

    Same envelope as ``text_samples``; the payload is the exact prompt and
    response token IDs with a loss mask. A client-repaired message has no
    sampled tokens, so it lands in the prompt and never in the mask -- the
    ``author`` rule enforced by tokenization rather than by a check.
    """
    rows: list[dict[str, Any]] = []
    claimed: set[str] = set()
    trajectory = view.trajectory
    for index, path in enumerate(_paths(view)):
        if not path:
            continue
        abandoned = _abandoned(view, path[-1])
        token_ids: list[int] = []
        node_spans: list[list[int]] = []
        trainable: list[bool] = []
        logprobs: list[float] = []
        routed: list[Any] = []
        has_routed = True
        stop_reason: str | None = None
        reasons: set[str] = set()

        for node in path:
            node_tokens = node.token_ids
            # A span per node, in `node_ids` order, including the nodes that
            # contribute nothing -- an empty span keeps the two lists parallel,
            # which is the whole point of emitting them.
            node_spans.append([len(token_ids), len(token_ids) + len(node_tokens)])
            if not node_tokens:
                continue
            token_ids.extend(node_tokens)
            mask = node.sampled_mask or [False] * len(node_tokens)
            repeated = (
                not allow_repeated_targets
                and node.sampled_start is not None
                and node.id in claimed
            )
            # Exact tokens, possibly never delivered. The tokens stay in the
            # prompt -- they are what the model produced -- and the mask
            # refuses to train on them.
            undelivered = node.id in view.uncertain_node_ids
            suppressed = repeated or undelivered or (mask_abandoned and abandoned)
            if suppressed:
                trainable.extend([False] * len(node_tokens))
                if repeated:
                    reasons.add("repeated_target")
                if undelivered:
                    reasons.add("delivery_uncertain")
                if mask_abandoned and abandoned:
                    reasons.add("abandoned")
            else:
                trainable.extend(mask)
                if any(mask):
                    claimed.add(node.id)
            logprobs.extend(node.logprobs or [0.0] * len(node_tokens))
            node_routed = node.routed_experts
            if node_routed is None:
                has_routed = False
            elif has_routed:
                routed.extend(node_routed)
            payload_tokens = node.tokens
            if payload_tokens.get("stop_reason"):
                stop_reason = payload_tokens["stop_reason"]

        if not token_ids:
            continue
        # One flat sequence with a mask over it. A prompt/response split is a
        # single-turn idea: on a multi-turn path everything after the first
        # generation is "response", including the user and tool messages that
        # follow it, so the split names nothing a consumer can use. The mask
        # says which positions the model produced, at any depth.
        loss_mask = [int(value) for value in trainable]
        if overlong_filtering and stop_reason == "context_length":
            loss_mask = [0] * len(loss_mask)
            reasons.add("context_length")

        row = _sample_envelope(view, index, path, abandoned=abandoned)
        row.update(
            {
                "trainable_count": sum(loss_mask),
                "input_ids": token_ids,
                # Where each of `node_ids` sits in the flat arrays. Without it
                # a row's turn boundaries live only in the graph, and the
                # train-once rule cannot be checked from an export at all.
                "node_spans": node_spans,
                "loss_mask": loss_mask,
                "rollout_logprobs": logprobs,
                "rollout_expert_indices": routed if has_routed and routed else None,
                "stop_reason": stop_reason,
                # `model` comes from the call, via the envelope: the upstream's
                # configured model is often unset, and the call's is what
                # actually produced these tokens.
                "tokenizer": trajectory.upstream_snapshot.get("tokenizer"),
            }
        )
        if row["trainable_count"] == 0 and reasons:
            row["masked_reason"] = sorted(reasons)[0]
        rows.append(row)
    return rows
