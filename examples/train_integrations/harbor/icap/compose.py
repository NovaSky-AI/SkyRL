"""Turn inference-capture's ``token_samples`` rows into a ``GeneratorOutput``.

This is the whole SkyRL-side adapter. inference-capture stays framework
neutral -- it never imports SkyRL and ``GeneratorOutput`` does not appear in
it -- so the mapping to whatever shape training wants lives here, where it can
change without a release on the other side.

``trajectory.finish(format="token_samples")`` returns one row per root-to-leaf branch
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

One decision stays here rather than in capture, because only the harness knows
it: a trial that timed out or errored is masked.

**Grouping.** One Harbor execution is one physical rollout with one reward,
and capture may export several complete paths for it. Those paths are sample
shards of the same rewarded execution, not independent reward observations, so
graph shape must not change what a rollout is worth.

SkyRL already has machinery for "several rows, one rollout, one advantage":
step-wise training. This reuses it rather than generalizing the non-step-wise
contract, which needs no trainer change at all. A rollout's paths are emitted
contiguously under one ``TrajectoryID``, every path carries the rollout's
reward, and ``is_last_step`` marks the last of them. The marker means
"representative reward row and end of this group" here, not "chronologically
final LLM turn" -- an imperfect name for what it is being used for, and worth
knowing when reading the trainer.

Because every path in a group carries the same reward, which path is marked
does not change the advantage. It does change what step-wise *evaluation*
keeps: metrics retain only the marked row, so token-exact parity across every
path has to read the generator output before that filtering.

Two settings go with this shape, and `_require_grouped_output` refuses to
start without them:

* ``generator.step_wise_trajectories=true`` -- lets one input rollout emit
  several output rows without touching SkyRL's validation or trainer;
* ``generator.merge_stepwise_output=false`` -- prefix merging exists to
  recombine sequential per-turn rows where ``prompt[i] + response[i]`` is a
  prefix of ``prompt[i+1]``. These rows are already complete multi-turn
  samples, so merging would at best be redundant and at worst fuse two paths
  that merely look like a prefix of one another.

TODO(zero-variance): ``trainer_utils.zero_variance_filter`` counts rows, not
trajectories. It drops a group when more than one *row* shares a reward with
no spread, but its own contract is "groups with <=1 live trajectory are always
kept" -- so one rollout that emits several rows looks like several
trajectories that happen to agree. The fix is to take the variance over the
``is_last_step`` rows, which is one per rollout.

This is not new and not made worse here: true step-wise already emits one row
per turn and hits it identically. It stays latent because GRPO runs use more
than one repetition, and duplication does not move ``max - min``. It bites at
``n_samples_per_prompt=1``, by either route. Eval-only runs are unaffected.
Before training with a single repetition that can produce several paths,
either fix the filter or disable it.
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


def stepwise_rows(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One complete path, cut back into the per-turn rows it was built from.

    Not used to train. This is the inverse of what makes a captured path a
    single sample, and it exists so parity against the harness-side TITO
    collector can be checked token for token.

    The sibling integration emits, per turn, the full prompt before that turn
    and that turn's completion with an all-ones mask. A captured path holds
    the same information differently: one rendered sequence whose loss mask
    marks the sampled spans. Each contiguous run of ones is one turn, so turn
    ``t`` is ``input_ids[:start_t]`` and ``input_ids[start_t:end_t]`` -- the
    same two arrays the sibling built directly.

    Comparing there rather than after composition is the point. Step-wise
    evaluation keeps only the ``is_last_step`` row, so a parity check run over
    evaluated output would compare one path out of a rollout's several and
    pass while the rest went unverified.

    A rollout that branched has no counterpart on the other side at all -- the
    sibling integration refuses summarization, because compaction breaks its
    own token accounting. Parity is therefore checked on rollouts that did not
    branch, and branching is what capture is for rather than something to
    reconcile.
    """
    input_ids: List[int] = list(row["input_ids"])
    loss_mask: List[int] = [int(value) for value in row["loss_mask"]]
    if len(loss_mask) != len(input_ids):
        raise ValueError(
            f"row {row.get('path_id')} is inconsistent: {len(input_ids)} tokens, "
            f"{len(loss_mask)} mask"
        )
    logprobs = list(row.get("rollout_logprobs") or [])

    turns: List[Dict[str, Any]] = []
    index = 0
    while index < len(loss_mask):
        if not loss_mask[index]:
            index += 1
            continue
        start = index
        while index < len(loss_mask) and loss_mask[index]:
            index += 1
        turns.append(
            {
                "prompt_token_ids": input_ids[:start],
                "response_ids": input_ids[start:index],
                "loss_mask": [1] * (index - start),
                "rollout_logprobs": logprobs[start:index] if logprobs else None,
            }
        )
    return turns


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
    step_wise: bool = True,
    generation_times: Optional[Sequence[float]] = None,
) -> GeneratorOutput:
    """Build a ``GeneratorOutput`` from one export per trajectory.

    Every trainable path becomes one complete multi-turn row. A trajectory
    that branched contributes several, emitted contiguously under its own
    ``TrajectoryID`` with the rollout's reward on each and ``is_last_step``
    marking the last -- see the module docstring for why that marker is being
    borrowed.

    ``step_wise`` is accepted and ignored. It used to decide whether a
    branched trajectory could contribute more than one row; masking those was
    always a workaround for a contract that could not express a group, and a
    summarizing agent branches by design. Deprecated: remove it once no call
    site passes it.
    """
    if not step_wise:
        logger.warning(
            "compose(step_wise=False) is ignored: a branched trajectory now emits "
            "one row per path, grouped by trajectory id. Remove the argument."
        )
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
    out_is_last_step: List[bool] = []

    for index, rows in enumerate(exports):
        stop_reason = stop_reasons[index]
        trainable = [] if stop_reason in MASKED_STOP_REASONS else [s for s in map(split_row, rows) if s]

        group = trainable or [_placeholder()]
        for position, row in enumerate(group):
            prompt_token_ids.append(row["prompt_token_ids"])
            response_ids.append(row["response_ids"])
            loss_masks.append(row["loss_mask"])
            rollout_logprobs.append(row["rollout_logprobs"])
            expert_indices.append(row["rollout_expert_indices"])
            # One reward covers a whole branched tree, which is what a fan-out
            # of sub-agents needs, so every row from a trajectory carries it.
            out_rewards.append(rewards[index])
            out_stop_reasons.append(stop_reason)
            # Contiguous, under one id: that grouping is what lets the trainer
            # compute this rollout's advantage once and broadcast it.
            out_trajectory_ids.append(trajectory_ids[index])
            # The last path of the group, not the last turn of a conversation.
            # Every path carries the same reward, so which one is marked does
            # not change the advantage -- only what step-wise evaluation keeps.
            out_is_last_step.append(position == len(group) - 1)
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
        is_last_step=out_is_last_step,
        rollout_metrics=None,
    )
