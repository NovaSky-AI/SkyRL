"""Unified reward metrics for SkyRL and Tinker.

This module provides shared metric calculation functions used by both:
- SkyRL trainer (skyrl_train/trainer.py, skyrl_train/utils/trainer_utils.py)
- Tinker integration (integrations/fleet/entrypoints/main_fleet_tinker.py)

All metrics follow the same naming convention for WandB logging:
- reward/{group}/pass_at_{n} - Pass@n metric for group
- reward/{group}/variance_per_prompt - Mean within-prompt reward variance (GRPO learning signal)
- reward/{group}/mean_positive_reward - Mean of positive rewards for group

Rewards can be in two formats:
- Scalar rewards: List[float] - one reward per trajectory
- Token-level rewards: List[List[float]] - per-token rewards per trajectory (summed to scalar)
"""

from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np


def flatten_rewards(rewards: Union[List[float], List[List[float]]]) -> List[float]:
    """Flatten rewards to scalar format.

    Handles both scalar rewards (List[float]) and token-level rewards (List[List[float]]).
    For token-level rewards, sums each trajectory's rewards into a single scalar.

    Args:
        rewards: Either List[float] (scalar per trajectory) or
                 List[List[float]] (token-level per trajectory)

    Returns:
        List[float]: Flattened scalar rewards, one per trajectory
    """
    if not rewards:
        return []

    flat_rewards: List[float] = []
    for r in rewards:
        if isinstance(r, list):
            # Token-level rewards: sum to get trajectory reward
            flat_rewards.append(float(sum(r)))
        else:
            flat_rewards.append(float(r))
    return flat_rewards


def sanitize_metric_key(key: str) -> str:
    """Sanitize metric key for wandb (replace / with _).

    Args:
        key: Raw metric key that may contain slashes

    Returns:
        Sanitized key with slashes replaced by underscores
    """
    return key.replace("/", "_")


def compute_pass_at_n(
    rewards: Union[List[float], List[List[float]]],
    uids: List[str],
) -> float:
    """Compute pass@n: fraction of unique prompts with at least one positive reward.

    For each unique prompt (identified by uid), if ANY of its rollouts has a positive
    reward, that prompt counts as a "pass". This metric measures how often the model
    can succeed at least once when given multiple attempts.

    Args:
        rewards: List of rewards (one per rollout). Can be scalar (List[float])
                 or token-level (List[List[float]]) - will be flattened.
        uids: List of unique IDs (one per rollout, same uid = same prompt)

    Returns:
        Float between 0.0 and 1.0 representing the fraction of prompts that passed
    """
    flat_rewards = flatten_rewards(rewards)
    uid_to_rewards: Dict[str, List[float]] = defaultdict(list)
    for uid, reward in zip(uids, flat_rewards):
        uid_to_rewards[uid].append(reward)

    if not uid_to_rewards:
        return 0.0

    passed = sum(1 for r_list in uid_to_rewards.values() if any(r > 0 for r in r_list))
    return passed / len(uid_to_rewards)


def compute_variance_per_prompt(
    rewards: Union[List[float], List[List[float]]],
    uids: List[str],
) -> float:
    """Compute mean within-prompt reward variance (GRPO learning signal).

    For GRPO to learn, there must be variance in rewards within each prompt's rollouts.
    If all rollouts for a prompt get the same reward, there's no learning signal.

    This metric computes the variance of rewards for each prompt, then returns the
    mean variance across all prompts.

    Args:
        rewards: List of rewards (one per rollout). Can be scalar (List[float])
                 or token-level (List[List[float]]) - will be flattened.
        uids: List of unique IDs (one per rollout, same uid = same prompt)

    Returns:
        Mean variance across prompts. Higher = more learning signal.
        Returns 0.0 if no prompts or all prompts have single rollouts.
    """
    flat_rewards = flatten_rewards(rewards)
    uid_to_rewards: Dict[str, List[float]] = defaultdict(list)
    for uid, reward in zip(uids, flat_rewards):
        uid_to_rewards[uid].append(reward)

    if not uid_to_rewards:
        return 0.0

    # Compute variance for each prompt (need at least 2 samples for variance)
    variances = []
    for r_list in uid_to_rewards.values():
        if len(r_list) >= 2:
            variances.append(float(np.var(r_list)))

    return float(np.mean(variances)) if variances else 0.0


def compute_reward_metrics(
    rewards: Union[List[float], List[List[float]]],
    uids: List[str],
    n_samples_per_prompt: int,
) -> Dict[str, float]:
    """Compute core reward metrics.

    Args:
        rewards: List of rewards (one per rollout). Can be scalar (List[float])
                 or token-level (List[List[float]]) - will be flattened.
        uids: List of unique IDs for pass@n grouping
        n_samples_per_prompt: Number of samples per prompt (used in metric key name)

    Returns:
        Dictionary with keys:
            - "pass_at_{n}": Pass@n metric
            - "variance_per_prompt": Mean within-prompt reward variance (GRPO learning signal)
            - "mean_positive_reward": Mean of positive rewards only
    """
    # Flatten rewards once for efficiency (each sub-function would otherwise flatten again)
    flat_rewards = flatten_rewards(rewards)
    pass_at_n = compute_pass_at_n(flat_rewards, uids)
    variance = compute_variance_per_prompt(flat_rewards, uids)
    positive_rewards = [r for r in flat_rewards if r > 0]
    mean_positive = float(np.mean(positive_rewards)) if positive_rewards else 0.0

    return {
        f"pass_at_{n_samples_per_prompt}": pass_at_n,
        "variance_per_prompt": variance,
        "mean_positive_reward": mean_positive,
    }


def compute_per_group_metrics(
    rewards: Union[List[float], List[List[float]]],
    uids: List[str],
    groups: List[str],
    n_samples_per_prompt: int,
    prefix: str = "reward",
) -> Dict[str, float]:
    """Compute metrics grouped by a key (env_key, data_source, etc).

    This function computes reward metrics for each group separately, enabling
    per-environment analysis in training and evaluation.

    Args:
        rewards: List of rewards (one per rollout). Can be scalar (List[float])
                 or token-level (List[List[float]]) - will be flattened.
        uids: List of unique IDs for pass@n grouping within each group
        groups: List of group keys (e.g., env_key or data_source per rollout)
        n_samples_per_prompt: Number of samples per prompt (used in metric key name)
        prefix: Metric prefix ("reward" for training, "eval" for evaluation)

    Returns:
        Dictionary with keys like:
            - "{prefix}/{group}/avg_score"
            - "{prefix}/{group}/pass_at_{n}"
            - "{prefix}/{group}/mean_positive_reward"
    """
    # Flatten rewards once before grouping
    flat_rewards = flatten_rewards(rewards)

    # Group data by group key
    group_data: Dict[str, Dict[str, List[Any]]] = defaultdict(lambda: {"rewards": [], "uids": []})
    for reward, uid, group in zip(flat_rewards, uids, groups):
        group_key = group if group is not None else "unknown"
        group_data[group_key]["rewards"].append(reward)
        group_data[group_key]["uids"].append(uid)

    metrics: Dict[str, float] = {}
    for group_key, data in group_data.items():
        sanitized = sanitize_metric_key(group_key)
        group_metrics = compute_reward_metrics(data["rewards"], data["uids"], n_samples_per_prompt)
        for metric_name, value in group_metrics.items():
            metrics[f"{prefix}/{sanitized}/{metric_name}"] = value

    return metrics
