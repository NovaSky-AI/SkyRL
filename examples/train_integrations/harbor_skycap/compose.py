"""skycap's samples as a step-wise ``GeneratorOutput``.

``finish`` returns one sample per root-to-leaf path of the trajectory's context
graph. A linear rollout is one path; a summarization, a stripped-reasoning
replay, a subagent or a reply the harness discarded adds more. skycap has
already made each sampled message a training target in at most one path, so
the paths' loss masks never count a token twice. With ``skycap.train_paths``
set to ``final`` or a custom rule, the samples are the rows that rule picks.

One Harbor trial is one rollout with one reward, however many paths it has.
SkyRL's step-wise shape already says "several rows, one rollout": a trial's
paths are emitted contiguously under its ``TrajectoryID``, the last marked
``is_last_step``. Every row carries the trial's reward, since every sampled
token in the trial earned it. The trainer computes the advantage once per
trial, from the last row, and applies it to all of the trial's rows. Each row is a complete multi-turn sample, split at its first
trained token: everything before is prompt, and the loss mask says which of the
rest the model sampled.

The masking policy is the sibling's: an instance with any rollout that timed
out or failed is masked whole, and a trial that hit the context limit trains
with its reward unless overlong filtering is on.

With R3, each row carries its own routes: one per token of the path, prompt and
history included, each from the forward pass that ran it. The path's last token
was never forwarded, so its row is left out; the trainer pads it, as it does for
SkyRL's own trace.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from skycap import Sample

from skyrl.backends.skyrl_train.utils.routed_experts import (
    RoutedExpertIndices,
    compact_routed_expert_indices,
)
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_DTYPE,
    SAMPLE_SUPPORT_PADDING,
)
from skyrl.train.generators.base import GeneratorOutput, TrajectoryID
from skyrl.train.generators.utils import get_rollout_metrics

MASKED_STOP_REASONS = frozenset({"agent_timeout", "error"})


@dataclass
class TrialOutcome:
    """One trial: skycap's samples, plus what only Harbor knows."""

    trajectory_id: TrajectoryID
    samples: List[Sample] = field(default_factory=list)
    reward: float = 0.0
    # One of: "complete", "context_length", "agent_timeout", "error".
    stop_reason: str = "complete"
    e2e_time: Optional[float] = None
    # Model calls whose prompt skycap had to render instead of extending the previous tokens:
    # expected for a harness that edits or compacts its history, a surprise for an append-only one.
    unbridged_calls: int = 0
    # R3: the trial finished, but a trained path lacked routed experts.
    missing_routes: bool = False


@dataclass
class _Row:
    prompt: List[int]
    response: List[int]
    loss_mask: List[int]
    logprobs: List[float]
    support: Optional[List[List[int]]] = None
    routes: Optional[RoutedExpertIndices] = None
    placeholder: bool = False


def split(sample: Sample) -> Optional[_Row]:
    """A sample as prompt and response, split at its first trained token. None if it trains nothing."""
    if sample.input_ids is None or sample.loss_mask is None:
        raise ValueError("skycap samples need tokens; run skycap in token mode")
    try:
        first = sample.loss_mask.index(1)
    except ValueError:
        return None
    return _Row(
        prompt=sample.input_ids[:first],
        response=sample.input_ids[first:],
        loss_mask=sample.loss_mask[first:],
        logprobs=(
            list(sample.logprobs[first:]) if sample.logprobs is not None else [0.0] * (len(sample.input_ids) - first)
        ),
        support=sample.sampling_mask[first:] if sample.sampling_mask is not None else None,
        routes=_routes(sample),
    )


def _routes(sample: Sample) -> Optional[RoutedExpertIndices]:
    """The path's routes, less the last token's, which the engine never forwarded."""
    routed = sample.routed_experts
    if routed is None or len(routed) != len(sample.input_ids):
        return None
    return compact_routed_expert_indices(np.asarray(routed[:-1]))


def _placeholder() -> _Row:
    """A masked row: the batch keeps one entry for a rollout that contributes nothing."""
    return _Row(prompt=[0], response=[0], loss_mask=[0], logprobs=[0.0], placeholder=True)


def compose(
    outcomes: List[TrialOutcome],
    *,
    overlong_filtering: bool,
    top_k: int = -1,
    sample_support: bool = False,
    routed_experts: bool = False,
) -> GeneratorOutput:
    """``top_k`` sets the width of sampler-support rows, as SkyRL's own capture pads them.

    ``sample_support`` says the engine returns sampler support, so a batch with nothing to train
    still carries padded support rows, as every other batch will. ``routed_experts`` says the same
    of routes (R3); every trained row must then have them.
    """
    masked_instances = {o.trajectory_id.instance_id for o in outcomes if o.stop_reason in MASKED_STOP_REASONS}

    groups: List[List[_Row]] = []
    trained: List[TrialOutcome] = []
    for outcome in outcomes:
        rows: List[_Row] = []
        if outcome.trajectory_id.instance_id not in masked_instances:
            rows = [row for row in map(split, outcome.samples) if row is not None]
            if rows:
                trained.append(outcome)
            if outcome.stop_reason == "context_length" and overlong_filtering:
                for row in rows:
                    row.loss_mask = [0] * len(row.loss_mask)
        groups.append(rows or [_placeholder()])

    real = [row for rows in groups for row in rows if not row.placeholder]
    support = _sample_support(groups, real, top_k, expected=sample_support)
    routes = _rollout_routes(groups, real) if routed_experts else None

    out: Dict[str, List[Any]] = {
        key: []
        for key in (
            "prompt_token_ids",
            "response_ids",
            "rewards",
            "loss_masks",
            "stop_reasons",
            "rollout_logprobs",
            "trajectory_ids",
            "is_last_step",
        )
    }
    times: List[Optional[float]] = []
    for outcome, rows in zip(outcomes, groups):
        masked = outcome.trajectory_id.instance_id in masked_instances
        for position, row in enumerate(rows):
            last = position == len(rows) - 1
            out["prompt_token_ids"].append(row.prompt)
            out["response_ids"].append(row.response)
            out["rewards"].append(0.0 if masked else outcome.reward)
            out["loss_masks"].append(row.loss_mask)
            out["stop_reasons"].append("error" if masked else outcome.stop_reason)
            out["rollout_logprobs"].append(row.logprobs)
            out["trajectory_ids"].append(outcome.trajectory_id)
            out["is_last_step"].append(last)
            times.append(outcome.e2e_time)

    return GeneratorOutput(
        **out,
        trajectory_generation_times=None if any(t is None for t in times) else times,
        rollout_expert_indices=routes,
        rollout_sample_support=support,
        rollout_metrics=_metrics(outcomes, trained, masked_instances),
    )


def _sample_support(
    groups: List[List[_Row]], real: List[_Row], top_k: int, expected: bool
) -> Optional[List[np.ndarray]]:
    if (not real and not expected) or (real and all(row.support is None for row in real)):
        return None
    if any(row.support is None for row in real):
        raise ValueError("some captured paths have sampler support and some don't")
    widest = max((len(ids) for row in real for ids in row.support), default=1)
    if top_k > 0 and widest > top_k:
        raise ValueError(f"a sampler support row has {widest} ids, more than top_k={top_k}")
    top_k = top_k if top_k > 0 else max(widest, 1)
    arrays = []
    for rows in groups:
        for row in rows:
            array = np.full((len(row.response), top_k), SAMPLE_SUPPORT_PADDING, dtype=SAMPLE_SUPPORT_DTYPE)
            for index, ids in enumerate(row.support or ()):
                array[index, : len(ids)] = ids
            arrays.append(array)
    return arrays


def _rollout_routes(groups: List[List[_Row]], real: List[_Row]) -> Optional[List[RoutedExpertIndices]]:
    # Only a row that still trains needs its own routes; one whose loss mask overlong filtering
    # cleared trains nothing, so it may go without.
    missing = sum(row.routes is None and any(row.loss_mask) for row in real)
    if missing:
        # The generator turns a trial without routes into an error, so this is a bug, not a rollout.
        raise ValueError(f"{missing} of {len(real)} trained paths have no routed experts")
    shaped = next((row.routes for row in real if row.routes is not None), None)
    if shaped is None:
        # Nothing trains, and no shape to pad the rest with: this step replays nothing.
        return None
    # A row without routes gets one route of distinct experts, as the trainer's own padding does
    # (``replay_padding_row``): Megatron's dispatcher needs ``tokens * topk`` distinct slots.
    layers, topk = shaped.shape[1:]
    dummy = np.broadcast_to(np.arange(topk, dtype=shaped.dtype), (1, layers, topk)).copy()
    return [dummy if row.routes is None else row.routes for rows in groups for row in rows]


def _metrics(outcomes: List[TrialOutcome], trained: List[TrialOutcome], masked_instances: set) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if trained:
        # One entry per rollout: its longest path, which holds the most of what it generated.
        responses = [max((s.input_ids for s in o.samples), key=len) for o in trained]
        times = [o.e2e_time for o in trained]
        metrics = get_rollout_metrics(
            responses,
            [o.reward for o in trained],
            trajectory_completion_times=None if any(t is None for t in times) else times,
        )
        num_turns = sum(len(s.targets) for o in trained for s in o.samples)
        metrics["generate/skycap/avg_num_turns"] = num_turns / len(trained)
        metrics["generate/skycap/avg_num_paths"] = sum(len(o.samples) for o in trained) / len(trained)
        metrics["generate/skycap/trajectories_context_length_exceeded"] = sum(
            o.stop_reason == "context_length" for o in trained
        )
    metrics["generate/skycap/num_timeout_trajectories"] = sum(o.stop_reason == "agent_timeout" for o in outcomes)
    metrics["generate/skycap/num_error_trajectories"] = sum(o.stop_reason == "error" for o in outcomes)
    metrics["generate/skycap/num_masked_instances"] = len(masked_instances)
    # Which trajectories to open in the viewer: each forks at its first unbridged call.
    metrics["generate/skycap/num_unbridged_trajectories"] = sum(o.unbridged_calls > 0 for o in outcomes)
    metrics["generate/skycap/num_unbridged_calls"] = sum(o.unbridged_calls for o in outcomes)
    metrics["generate/skycap/num_missing_route_trajectories"] = sum(o.missing_routes for o in outcomes)
    return metrics
