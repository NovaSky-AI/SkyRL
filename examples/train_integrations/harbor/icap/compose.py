"""Turn inference-capture's ``token-samples`` rows into a ``GeneratorOutput``.

This is the whole SkyRL-side adapter. inference-capture stays framework
neutral -- it never imports SkyRL and ``GeneratorOutput`` does not appear in
it -- so the mapping to whatever shape training wants lives here, where it can
change without a release on the other side.

``trajectory.export("token-samples")`` returns one row per root-to-leaf branch
of the message graph:

    {"path_id": ..., "trajectory_id": ..., "node_ids": [...], "abandoned": bool,
     "labels": [...], "annotations": {...}, "trainable_count": int,
     "input_ids": [...], "loss_mask": [...], "rollout_logprobs": [...],
     "rollout_expert_indices": ... | None, "stop_reason": ..., "tokenizer": ...}

A linear rollout is one row. A summarization is two -- the pre-compaction path
and the rewritten one -- because a rewritten history stops matching at the last
unchanged message and branches there. A sub-agent fan-out is more.

Capture has already applied the rule that a sampled node reachable from several
branches is trainable in exactly one of them, so summing ``loss_mask`` across a
trajectory's rows never counts the same sampled tokens twice. It also never
drops a row: a fully-masked row carries ``trainable_count == 0`` and usually a
``masked_reason``, which is information a missing row would not have.

Two decisions stay here rather than in capture, because only the harness knows
them: a trial that timed out or errored is masked, and whether a trajectory
that branched may contribute more than one row.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from skyrl.train.generators.base import GeneratorOutput

logger = logging.getLogger(__name__)

# Outcomes the harness knows and capture does not.
MASKED_STOP_REASONS = frozenset({"agent_timeout", "error"})


def split_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Split one exported branch into prompt and response.

    The boundary is the first token this row may train on, not the first turn.
    Everything before it is context the model was given; everything after is a
    mix of what the model produced and what the harness replayed, and the mask
    says which is which -- which is why a multi-turn path cannot be described
    by a prompt/response pair without the mask travelling with it.
    """
    input_ids: List[int] = list(row["input_ids"])
    loss_mask: List[int] = [int(value) for value in row["loss_mask"]]
    logprobs: List[float] = list(row.get("rollout_logprobs") or [0.0] * len(input_ids))

    if not (len(loss_mask) == len(input_ids) == len(logprobs)):
        raise ValueError(
            f"row {row.get('path_id')} is inconsistent: {len(input_ids)} tokens, "
            f"{len(loss_mask)} mask, {len(logprobs)} logprobs"
        )

    try:
        first = loss_mask.index(1)
    except ValueError:
        # Nothing to learn from: masked by capture's train-once rule, by
        # overlong filtering, or because the branch is pure replay.
        return None

    routed = row.get("rollout_expert_indices")
    return {
        "prompt_token_ids": input_ids[:first],
        "response_ids": input_ids[first:],
        "loss_mask": loss_mask[first:],
        "rollout_logprobs": logprobs[first:],
        "rollout_expert_indices": routed,
    }


def _placeholder() -> Dict[str, Any]:
    """A masked row. The batch keeps its shape rather than losing an entry."""
    return {
        "prompt_token_ids": [0],
        "response_ids": [0],
        "loss_mask": [0],
        "rollout_logprobs": [0.0],
        "rollout_expert_indices": None,
    }


def compose(
    exports: Sequence[Sequence[Dict[str, Any]]],
    *,
    trajectory_ids: Sequence[Any],
    rewards: Sequence[float],
    stop_reasons: Sequence[str],
    step_wise: bool,
    generation_times: Optional[Sequence[float]] = None,
) -> GeneratorOutput:
    """Build a ``GeneratorOutput`` from one export per trajectory.

    ``step_wise`` decides whether a trajectory may contribute more than one
    row. With it off, a trajectory that branched has no single answer to "what
    is this trajectory's sample", so it is masked rather than guessed at. A
    summarizing agent branches by design and hits that every time, which is why
    summarization wants step-wise on.
    """
    if not (len(exports) == len(trajectory_ids) == len(rewards) == len(stop_reasons)):
        raise ValueError("compose() inputs must be the same length, one entry per trajectory")

    prompt_token_ids: List[List[int]] = []
    response_ids: List[List[int]] = []
    loss_masks: List[List[int]] = []
    rollout_logprobs: List[List[float]] = []
    expert_indices: List[Any] = []
    out_rewards: List[float] = []
    out_stop_reasons: List[str] = []
    out_trajectory_ids: List[Any] = []
    out_times: List[float] = []

    for index, rows in enumerate(exports):
        stop_reason = stop_reasons[index]
        trainable = [] if stop_reason in MASKED_STOP_REASONS else [s for s in map(split_row, rows) if s]

        if not step_wise and len(trainable) > 1:
            logger.warning(
                "masking trajectory %s: it produced %d trainable branches and step_wise "
                "is off, so there is no single row for it. A summarizing agent branches "
                "by design -- turn step_wise on to train those rows.",
                trajectory_ids[index],
                len(trainable),
            )
            trainable = []

        for row in trainable or [_placeholder()]:
            prompt_token_ids.append(row["prompt_token_ids"])
            response_ids.append(row["response_ids"])
            loss_masks.append(row["loss_mask"])
            rollout_logprobs.append(row["rollout_logprobs"])
            expert_indices.append(row["rollout_expert_indices"])
            # One reward covers a whole branched tree, which is what a fan-out
            # of sub-agents needs, so every row from a trajectory carries it.
            out_rewards.append(rewards[index])
            out_stop_reasons.append(stop_reason)
            out_trajectory_ids.append(trajectory_ids[index])
            if generation_times is not None:
                out_times.append(generation_times[index])

    return GeneratorOutput(
        prompt_token_ids=prompt_token_ids,
        response_ids=response_ids,
        rewards=out_rewards,
        loss_masks=loss_masks,
        stop_reasons=out_stop_reasons,
        rollout_logprobs=rollout_logprobs,
        rollout_expert_indices=(expert_indices if any(entry is not None for entry in expert_indices) else None),
        trajectory_ids=out_trajectory_ids,
        trajectory_generation_times=out_times or None,
        rollout_metrics=None,
    )
