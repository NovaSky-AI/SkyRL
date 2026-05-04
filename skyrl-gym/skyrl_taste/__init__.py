"""skyrl_taste: thin async wrapper around the taste-judge for SkyRL GRPO.

Public API:
    score_trajectory_async(task, actions, outcome) -> Optional[float]
    get_judge_provider_info() -> {"taste_judge_provider", "taste_judge_model"}

Returns a value in [0, 1] (rescaled from the 1-5 weighted_total) or None
when the judge is disabled / errored.
"""

from .judge import score_trajectory_async, get_judge_provider_info

__all__ = ["score_trajectory_async", "get_judge_provider_info"]
