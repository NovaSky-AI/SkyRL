"""
Reward functions for task generation RL.

Computes:
    R(task) = llm_validity * (alpha * var(raw_scores) + (p_hint - p_raw))

Components:
    - var(raw_scores): Variance of k raw (no-hint) evaluator rollouts.
      Measures difficulty calibration — maximized at p_raw ≈ 0.5
      (Bernoulli variance = 0.25). Tasks at the evaluator's frontier.
    - p_hint - p_raw: Hint gap — mean(hinted) minus mean(raw).
      Positive when hints help, meaning the task is hard but solvable.
      Captures learnability beyond current capability.
    - llm_validity: LLM-as-a-judge gate (0/1). Kills reward for broken tasks.
    - alpha: Weight balancing variance (frontier difficulty) vs hint gap (learnability). Default 1.0 (equal weight).
"""

from typing import Dict, List


def compute_variance(scores: List[float]) -> float:
    """Compute variance of binary rollout scores.

    Args:
        scores: List of binary (0/1) rollout outcomes.

    Returns:
        Variance in [0, 0.25]. Zero when all same, max at p=0.5.
    """
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    return sum((s - mean) ** 2 for s in scores) / len(scores)


def compute_hint_gap(raw_scores: List[float], hinted_scores: List[float]) -> float:
    """Compute hint gap: mean(hinted) - mean(raw).

    Positive when hints help the evaluator solve the task.
    Zero or negative when hints don't help (task too easy or too hard).

    Args:
        raw_scores: Scores from evaluator rollouts without hints.
        hinted_scores: Scores from evaluator rollouts with hints.

    Returns:
        Hint gap in [-1, 1].
    """
    if not raw_scores or not hinted_scores:
        return 0.0
    p_raw = sum(raw_scores) / len(raw_scores)
    p_hint = sum(hinted_scores) / len(hinted_scores)
    return p_hint - p_raw


def compute_task_reward(
    raw_scores: List[float],
    hinted_scores: List[float],
    validity: float = 1.0,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """Compute the full task generation reward.

    R = validity * (alpha * var(raw) + (p_hint - p_raw))

    Args:
        raw_scores: Scores from k evaluator rollouts without hints.
        hinted_scores: Scores from k evaluator rollouts with hints.
        validity: LLM-as-a-judge gate (0.0 or 1.0).
        alpha: Weight for variance term.

    Returns:
        Dict with all reward components and total.
    """
    p_raw = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0
    p_hint = sum(hinted_scores) / len(hinted_scores) if hinted_scores else 0.0
    var_raw = compute_variance(raw_scores)
    hint_gap = p_hint - p_raw
    total = validity * (alpha * var_raw + hint_gap)

    return {
        "validity": validity,
        "p_raw": p_raw,
        "p_hint": p_hint,
        "var_raw": var_raw,
        "hint_gap": hint_gap,
        "alpha": alpha,
        "total": total,
    }
