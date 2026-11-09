"""Streamed training (``trainer.fully_async.store_groups_in_ray``): a mini-batch planned from its stored summaries.

The fully async trainer collects stored groups and trains each policy DP rank on the shard
``plan_streamed_mini_batch`` assigns it; the driver never loads a token payload.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

import numpy as np
import torch

from skyrl.backends.skyrl_train.rollout_shard import RolloutPayloadKey, RolloutShard
from skyrl.backends.skyrl_train.utils.loss_normalization import (
    MinibatchLossNormalization,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    GROUP_LOCAL_ADVANTAGE_ESTIMATORS,
    ROW_SEPARABLE_POLICY_LOSSES,
    LossReduction,
    batch_normalize_advantages,
    loss_reduction_row_scales,
)
from skyrl.backends.skyrl_train.workers.worker_utils import (
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_MAX_KEY,
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY,
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_MIN_KEY,
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_STD_KEY,
)
from skyrl.train.config import AlgorithmConfig, SkyRLTrainConfig
from skyrl.train.dataset.bin_packing import Balanced
from skyrl.train.dataset.preprocess import RowCollation
from skyrl.train.generators.base import (
    RolloutMetadataView,
    RolloutSummary,
    sample_support_metrics,
)
from skyrl.train.trainer import advantage_metrics, outcome_advantages
from skyrl.train.utils.trainer_utils import zero_variance_filter

if TYPE_CHECKING:
    from skyrl.train.fully_async_trainer import GeneratedOutputGroup


def unsupported_streaming_settings(cfg: SkyRLTrainConfig, *, has_critic: bool) -> List[str]:
    """The settings ``trainer.fully_async.store_groups_in_ray`` cannot train exactly, as requirements.

    Streamed training plans a mini-batch from its stored summaries: each trajectory's advantage
    comes from its group's rewards, the loss reduction becomes one factor per row, and each policy
    DP rank trains on its own trajectories in one forward_backward with the step's denominators.
    That equals training the whole mini-batch when every loss term sums independent per-row terms.
    """
    algorithm = cfg.trainer.algorithm
    megatron_config = cfg.trainer.policy.megatron_config
    # transformer_config_kwargs override the explicit field on the Megatron provider.
    moe_aux_loss_coeff = (megatron_config.transformer_config_kwargs or {}).get(
        "moe_aux_loss_coeff", megatron_config.moe_aux_loss_coeff
    )
    requirements = {
        f"an advantage estimator in {sorted(e.value for e in GROUP_LOCAL_ADVANTAGE_ESTIMATORS)}": (
            algorithm.advantage_estimator in GROUP_LOCAL_ADVANTAGE_ESTIMATORS
        ),
        f"a policy loss in {sorted(loss.value for loss in ROW_SEPARABLE_POLICY_LOSSES)}": (
            algorithm.policy_loss_type in ROW_SEPARABLE_POLICY_LOSSES
        ),
        f"a loss reduction other than {LossReduction.TOKEN_MEAN_LEGACY.value}": (
            algorithm.loss_reduction != LossReduction.TOKEN_MEAN_LEGACY
        ),
        "one update epoch": cfg.trainer.update_epochs_per_batch == 1,
        # Ranks train different numbers of rows; token batching equalizes their microbatch counts.
        "token-based microbatching (trainer.max_tokens_per_microbatch > 0)": cfg.trainer.max_tokens_per_microbatch > 0,
        "no critic": not has_critic,
        "no KL loss (the reference model is not streamed)": not algorithm.use_kl_loss,
        "no KL-in-reward": not algorithm.use_kl_in_reward,
        "no MoE aux loss": not moe_aux_loss_coeff,
        "no step-wise output merging": not cfg.generator.merge_stepwise_output,
        "no vision-language generator": not cfg.generator.vision_language_generator,
        # The MTP draft loss scores every attended token, including those of trajectories a stream leaves out.
        "no MTP draft heads (policy.megatron_config.mtp_num_layers)": not cfg.trainer.policy.megatron_config.mtp_num_layers,
    }
    return [name for name, satisfied in requirements.items() if not satisfied]


# The unstreamed trainer's ``policy/rollout_train_logprobs_abs_diff_*`` compares its old-logprob
# forward pass with the rollout. A streamed step that computes old logprobs logs them from the
# worker's diff of the logprobs its one optimizer step trains on, reduced over microbatches.
ROLLOUT_TRAIN_LOGPROB_DIFF_ALIASES = {
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_MAX_KEY: "rollout_train_logprobs_abs_diff_max",
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_MIN_KEY: "rollout_train_logprobs_abs_diff_min",
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY: "rollout_train_logprobs_abs_diff_mean",
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_STD_KEY: "rollout_train_logprobs_abs_diff_std",
}


# Trainer hooks a streamed step never calls: it trains from the stored groups' summaries instead.
STREAM_BYPASSED_HOOKS = (
    "postprocess_generator_output",
    "convert_generation_group_mini_batch_to_training_input",
    "convert_to_training_input",
    "fwd_logprobs_values_reward",
    "apply_reward_kl_penalty",
    "compute_advantages_and_returns",
    "train_critic_and_policy",
)


@dataclass(frozen=True)
class StreamedRolloutRows:
    """A streamed mini-batch's rows as the driver holds them: their metadata, no token payloads."""

    # The kept groups' rows, as ``log_rollout_metrics`` saw them.
    view: RolloutMetadataView
    uids: List[str]
    # Each row's advantage as ``compute_advantages_and_returns`` gives it: before z-scoring
    # (``advantage_batch_normalize``) and the loss reduction. Every response token carries it.
    advantages: List[float]
    response_lengths: List[int]
    loss_tokens: List[int]
    # The loss tokens each row trains: 0 for a row of a group the zero-variance filter drops.
    trained_loss_tokens: List[int]


@dataclass(frozen=True)
class StreamedPlan:
    """A streamed mini-batch planned from its stored summaries: what each policy DP rank trains on."""

    shards: List[RolloutShard]
    normalization: MinibatchLossNormalization
    compute_old_logprobs: bool
    # Metrics that follow from the plan alone (advantages, filtering, batch shape).
    metrics: Dict[str, float]
    # The kept groups' rows, for ``log_streamed_rollouts`` and ``on_step_end``.
    rows: StreamedRolloutRows
    # Each shard's rows' advantages before z-scoring and the loss reduction, for its batch dump.
    unscaled_advantages: List[Tuple[float, ...]]


def plan_streamed_mini_batch(
    groups: List["GeneratedOutputGroup"],
    algorithm: AlgorithmConfig,
    *,
    dp_size: int,
    collation: RowCollation,
    skip_policy_forward: Callable[[bool], bool],
) -> StreamedPlan:
    """What each policy DP rank trains on this step, from the groups' stored summaries; loads nothing.

    Advantages are computed as the unstreamed trainer computes them: from each trajectory's
    last-step reward within its group, over every trajectory (a token-level reward counts as its
    sum, the score every group-local estimator takes). Only trajectories with loss tokens, in
    groups the zero-variance filter keeps, train and count in the denominators.
    """
    trajectories = [(group.uid, t) for group in groups for t in group.generator_output.trajectories]
    uids = [uid for uid, _ in trajectories]
    rewards = [t.summary.rewards[-1] for _, t in trajectories]
    loss_tokens = [sum(t.summary.num_loss_tokens) for _, t in trajectories]
    response_tokens = torch.tensor([sum(t.summary.response_lengths) for _, t in trajectories], dtype=torch.float32)

    final_rewards = torch.tensor(rewards, dtype=torch.float32)
    advantages = outcome_advantages(algorithm, final_rewards[:, None], np.array(uids))[0][:, 0]
    # Every response token of a trajectory carries its advantage, as in the unstreamed batch.
    metrics: Dict[str, float] = advantage_metrics(final_rewards, advantages, token_counts=response_tokens)
    unscaled = advantages
    if algorithm.advantage_batch_normalize:
        advantages = batch_normalize_advantages(advantages, response_tokens)

    trained = [i for i, count in enumerate(loss_tokens) if count > 0]
    if algorithm.zero_variance_filter:
        # As ``postprocess_generator_output`` filters: over every row's reward and loss tokens.
        row_uids = [uid for uid, t in trajectories for _ in range(t.num_rows)]
        kept_rows = zero_variance_filter(
            [r for _, t in trajectories for r in t.summary.rewards],
            row_uids,
            tol=algorithm.zero_variance_filter_tol,
            num_loss_tokens=[n for _, t in trajectories for n in t.summary.num_loss_tokens],
        )
        kept_uids = {row_uids[i] for i in kept_rows}
        metrics["reward/num_zero_variance_filtered"] = len(set(uids)) - len(kept_uids)
        trained = [i for i in trained if uids[i] in kept_uids]
    trained_set = set(trained)

    # Rows of the trained trajectories, in training order.
    prompt_of: Dict[str, int] = {}
    row_trajectory, row_loss_tokens, row_prompts, row_lengths = [], [], [], []
    for i in trained:
        uid, trajectory = trajectories[i]
        summary = trajectory.summary
        for row in range(trajectory.num_rows):
            row_trajectory.append(i)
            row_loss_tokens.append(summary.num_loss_tokens[row])
            row_prompts.append(prompt_of.setdefault(uid, len(prompt_of)))
            row_lengths.append(summary.row_lengths[row])
    normalization = MinibatchLossNormalization.from_rows(row_loss_tokens, row_lengths)
    scales = loss_reduction_row_scales(
        algorithm.loss_reduction,
        row_loss_tokens,
        row_prompts,
        num_sequences=normalization.num_sequences,
        num_loss_tokens=normalization.num_loss_tokens,
        max_seq_len=algorithm.max_seq_len,
    )
    row_advantages: Dict[int, List[float]] = defaultdict(list)
    for trajectory_index, scale in zip(row_trajectory, scales, strict=True):
        row_advantages[trajectory_index].append(advantages[trajectory_index].item() * scale)

    # Each rank trains on whole trajectories, balanced by tokens. A rank left without any trains
    # the smallest trained trajectory (routes and all) as padding, so it still runs the step's
    # collectives.
    trajectory_tokens = [sum(trajectory.summary.row_lengths) for _, trajectory in trajectories]
    padding = min(trained or range(len(trajectories)), key=lambda i: trajectory_tokens[i])

    def shard(members: List[int], advantages: Tuple[float, ...], padding: bool = False) -> RolloutShard:
        stored = [trajectories[i][1] for i in members]
        return RolloutShard(
            payload_refs=tuple(t.payload for t in stored),
            payload_rows=tuple(t.num_rows for t in stored),
            prompt_lengths=tuple(n for t in stored for n in t.summary.prompt_lengths),
            response_lengths=tuple(n for t in stored for n in t.summary.response_lengths),
            row_advantages=advantages,
            collation=collation,
            padding=padding,
        )

    shards, shard_unscaled = [], []
    for bin_ in Balanced.partition([trajectory_tokens[i] for i in trained], dp_size):
        if bin_:
            members = [trained[j] for j in bin_]
            shards.append(shard(members, tuple(a for i in members for a in row_advantages[i])))
            shard_unscaled.append(tuple(unscaled[i].item() for i in members for _ in row_advantages[i]))
        else:
            shards.append(shard([padding], (0.0,) * trajectories[padding][1].num_rows, padding=True))
            shard_unscaled.append((0.0,) * trajectories[padding][1].num_rows)
    rows_summary = RolloutSummary.cat([t.summary for _, t in trajectories])
    # Over every row of the kept groups, as the unstreamed batch counts them.
    metrics["generate/batch_num_seq"] = len(rows_summary.rewards)
    metrics["generate/batch_padded_seq_len"] = max(rows_summary.row_lengths)
    # DP ranks left without a trajectory to train, which train a padding copy instead.
    metrics["async/num_padding_dp_shards"] = sum(shard.padding for shard in shards)
    if rows_summary.sample_support is not None:
        metrics.update(sample_support_metrics(rows_summary.sample_support))
    # A microbatch collates rows of any trajectory that trains, so they must store the same fields.
    group_of = [group for group in groups for _ in group.generator_output.trajectories]
    payload_keys = {group_of[i].generator_output.payload_keys for i in (*trained, padding)}
    if len(payload_keys) > 1:
        raise ValueError(f"stored groups carry different payload keys: {sorted(map(sorted, payload_keys))}")
    has_rollout_logprobs = RolloutPayloadKey.ROLLOUT_LOGPROBS in next(iter(payload_keys))
    rows = StreamedRolloutRows(
        view=RolloutMetadataView.cat([group.metadata_view() for group in groups]),
        uids=[uid for uid, t in trajectories for _ in range(t.num_rows)],
        advantages=[unscaled[i].item() for i, (_, t) in enumerate(trajectories) for _ in range(t.num_rows)],
        response_lengths=rows_summary.response_lengths,
        loss_tokens=rows_summary.num_loss_tokens,
        trained_loss_tokens=[
            n if i in trained_set else 0 for i, (_, t) in enumerate(trajectories) for n in t.summary.num_loss_tokens
        ],
    )
    return StreamedPlan(
        shards=shards,
        normalization=normalization,
        compute_old_logprobs=not skip_policy_forward(has_rollout_logprobs),
        metrics=metrics,
        rows=rows,
        unscaled_advantages=shard_unscaled,
    )
