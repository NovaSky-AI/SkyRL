"""skyrl_gym.taste — async wrapper around the taste-judge for SkyRL GRPO.

Public API:
    score_trajectory_async(task, actions, outcome) -> Optional[float]
    get_judge_provider_info() -> {"taste_judge_provider", "taste_judge_model"}

Provider routing (env vars, read at call-time so swaps don't require a
restart of the process -- only a fresh rollout):
- ``SKYRL_TASTE_PROVIDER`` in {"anthropic", "openai", "openrouter"}.
  Default: "openrouter" (cheapest production path).
- ``SKYRL_TASTE_MODEL``: model identifier. Default
  "anthropic/claude-haiku-4.5" (an OpenRouter slug). For provider="anthropic"
  this would be e.g. "claude-sonnet-4-6"; for provider="openai" e.g.
  "gpt-4o-mini".
- ``SKYRL_TASTE_BLIND_OUTCOME``: "1" (default) suppresses the verifier
  outcome from the judge prompt. Stream 4 found that exposing the outcome
  causes taste scores to correlate ~0.7 with verifier (outcome bleed) and
  collapses the shaping signal.

Behavior:
- The underlying judges are *synchronous* and use blocking SDKs. We run
  them in `asyncio.to_thread(...)` so they do not stall the event loop.
  SkyRL's generator runs each rollout's `step_async` as its own asyncio
  task, so judge calls across the GRPO group naturally overlap.
- Returns the rubric's `weighted_total`, rescaled from [1, 5] -> [0, 1] so
  the blended reward stays in [0, 1] and existing pass@n / signal-ratio
  metrics in `integrations/fleet/reward_metrics.py` keep working.
- Returns None on:
    * `SKYRL_TASTE_DISABLED=1` (env-var bypass)
    * The underlying judge returning a None-shaped result (parse / API failure)
  The caller is expected to fall back to verifier-only reward when None.
- Screenshots are NOT passed in this version (text-only judge). Trade-off:
  text-only keeps judge latency around 1-3 s/trajectory and avoids blowing
  the judge's context with 50-80 base64 PNGs per browser_use rollout. We
  lose direct ui_grounding signal, but the judge can still infer it from
  action targets + tool errors. Re-enable screenshots later by sampling the
  `tool_result` image_url blocks out of `chat_history` and threading them
  through the judge call with `screenshots=...`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("skyrl_gym.taste")

# Make the research-side judge importable. In a packaged install this would
# be a sibling import; for the launch-ready integration we add the research
# tree to sys.path so we don't have to vendor it.
_RESEARCH_JUDGE_DIR = Path(__file__).resolve().parents[2] / "research" / "judge"
if _RESEARCH_JUDGE_DIR.is_dir() and str(_RESEARCH_JUDGE_DIR) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_JUDGE_DIR))

try:
    from judge import (  # type: ignore
        score_trajectory as _score_trajectory_anthropic,
        score_trajectory_gpt4o as _score_trajectory_openai,
        score_trajectory_openrouter as _score_trajectory_openrouter,
    )
except Exception as e:  # pragma: no cover
    logger.warning("could not import research judge: %s", e)
    _score_trajectory_anthropic = None  # type: ignore[assignment]
    _score_trajectory_openai = None  # type: ignore[assignment]
    _score_trajectory_openrouter = None  # type: ignore[assignment]


_DEFAULT_PROVIDER = "openrouter"
_DEFAULT_MODEL = "anthropic/claude-haiku-4.5"


def _resolve_provider() -> tuple[str, str, bool, Optional[Callable[..., dict]]]:
    """Read SKYRL_TASTE_PROVIDER / SKYRL_TASTE_MODEL / SKYRL_TASTE_BLIND_OUTCOME
    and return (provider, model, blind_outcome, callable). The callable is
    None if the corresponding research-side function failed to import."""
    provider = os.environ.get("SKYRL_TASTE_PROVIDER", _DEFAULT_PROVIDER).strip().lower()
    model = os.environ.get("SKYRL_TASTE_MODEL", _DEFAULT_MODEL)
    blind_outcome = os.environ.get("SKYRL_TASTE_BLIND_OUTCOME", "1") == "1"

    if provider == "anthropic":
        return provider, model, blind_outcome, _score_trajectory_anthropic
    if provider == "openai":
        return provider, model, blind_outcome, _score_trajectory_openai
    if provider == "openrouter":
        return provider, model, blind_outcome, _score_trajectory_openrouter
    logger.warning(
        "unknown SKYRL_TASTE_PROVIDER=%r; falling back to %s",
        provider,
        _DEFAULT_PROVIDER,
    )
    return _DEFAULT_PROVIDER, model, blind_outcome, _score_trajectory_openrouter


def _rescale_to_unit_interval(weighted_total: Optional[float]) -> Optional[float]:
    """Rescale weighted_total from [1, 5] (rubric) to [0, 1] (RL reward).

    Returns None passthrough; clips defensively.
    """
    if weighted_total is None:
        return None
    try:
        v = (float(weighted_total) - 1.0) / 4.0
    except (TypeError, ValueError):
        return None
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def get_judge_provider_info() -> dict[str, str]:
    """Return the resolved (provider, model) for run-once metric logging."""
    provider, model, _, _ = _resolve_provider()
    return {"taste_judge_provider": provider, "taste_judge_model": model}


async def score_trajectory_async(
    task: str,
    actions: list[dict[str, Any]],
    outcome: bool,
) -> Optional[float]:
    """Async-friendly entrypoint to the taste judge.

    Args:
        task: natural-language task description (`task_config["prompt"]`).
        actions: ordered list of action dicts pulled from the trajectory.
        outcome: bool from the verifier (verifier_reward >= 1.0).

    Returns:
        A scalar in [0, 1] = rescaled `weighted_total`, or None if the
        judge is disabled or failed. The caller must treat None as
        "fall back to verifier-only reward".
    """
    if os.environ.get("SKYRL_TASTE_DISABLED") == "1":
        # Hard kill switch for runtime rollback.
        return None

    provider, model, blind_outcome, fn = _resolve_provider()
    if fn is None:
        logger.warning(
            "taste judge module unavailable for provider=%s; returning None",
            provider,
        )
        return None

    # Run the blocking judge in a thread so we don't stall the event loop.
    # screenshots=None: see module docstring for the rationale.
    try:
        result = await asyncio.to_thread(
            fn,
            task,
            actions,
            outcome,
            None,  # screenshots
            model,
            blind_outcome,
        )
    except Exception as e:
        logger.warning("taste judge (%s) raised in thread: %s", provider, e)
        return None

    if not isinstance(result, dict):
        return None

    if result.get("error"):
        # The judge already logged; signal fall-back.
        return None

    return _rescale_to_unit_interval(result.get("weighted_total"))
