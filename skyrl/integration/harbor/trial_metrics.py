"""Per-attempt metrics of Harbor trials: how long each attempt spent in each of Harbor's phases
(sandbox start, agent, verifier), how many attempts were retries, how many failed, by
exception type, and what the agent did: its turns and tool calls, and for Codex's ``apply_patch``
how many patches went through, read from the trajectory Harbor writes for the attempt.

They describe the environment run, not how its tokens were collected, so every generator that
runs Harbor trials logs the same ``generate/harbor/*`` keys. They also show a slow or failing
sandbox start that a retry rescued, which the trajectory-level counts can't.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np
from loguru import logger

PHASES = ("environment_setup", "agent_execution", "verifier")
PREFIX = "generate/harbor"
#: The agent's trajectory in Harbor's ATIF format, relative to the trial directory.
TRAJECTORY_PATH = Path("agent", "trajectory.json")
#: Codex's patch tool, and what its output contains when the patch was applied.
APPLY_PATCH = "apply_patch"
PATCH_APPLIED = "Success. Updated the following files"


@dataclass
class ToolUse:
    """What the agent did in one attempt, from its trajectory."""

    agent_turns: int = 0
    tool_calls: int = 0
    patches_applied: int = 0
    patches_rejected: int = 0


@dataclass
class Attempt:
    """One attempt at a trial: its phase durations in seconds (None where Harbor has no
    timestamps), and the exception it ended with, if any."""

    environment_setup_time: Optional[float] = None
    agent_execution_time: Optional[float] = None
    verifier_time: Optional[float] = None
    exception: Optional[str] = None
    tool_use: Optional[ToolUse] = None


class TrialAttempts:
    """Every attempt one trial took, retries included, in order.

    A generator calls ``start()`` as each attempt begins, ``record()`` once Harbor's
    ``Trial.run()`` returns, and ``fail()`` if the attempt raises.
    """

    def __init__(self) -> None:
        self.attempts: List[Attempt] = []

    def start(self) -> None:
        self.attempts.append(Attempt())

    def record(self, result: Any) -> None:
        """Harbor's phase times, exception and agent trajectory for the current attempt, from its ``TrialResult``.

        Harbor sets ``environment_setup.finished_at`` in a ``finally``, so an attempt whose
        sandbox start timed out still has a duration.
        """
        attempt = self.attempts[-1]
        for phase in PHASES:
            setattr(attempt, f"{phase}_time", _seconds(getattr(result, phase, None)))
        exception_info = getattr(result, "exception_info", None)
        attempt.exception = exception_info.exception_type if exception_info else None
        attempt.tool_use = read_tool_use(getattr(result, "trial_uri", None))

    def fail(self, error: BaseException) -> None:
        """The current attempt raised. An exception Harbor already reported for it is the cause, so it stays."""
        attempt = self.attempts[-1]
        attempt.exception = attempt.exception or type(error).__name__


def trial_metrics(trials: Sequence[TrialAttempts]) -> Dict[str, Any]:
    """The ``generate/harbor/*`` metrics of one ``generate()`` call.

    An attempt that failed at sandbox start counts with its time up to the failure; a phase
    missing either timestamp is skipped.
    """
    attempts = [a for trial in trials for a in trial.attempts]
    metrics: Dict[str, Any] = {
        f"{PREFIX}/num_attempts": len(attempts),
        f"{PREFIX}/num_retried_attempts": sum(max(len(trial.attempts) - 1, 0) for trial in trials),
        # Always present, so a healthy step charts as zero rather than a gap.
        f"{PREFIX}/num_failed_attempts": sum(a.exception is not None for a in attempts),
    }
    for phase in PHASES:
        times = [t for t in (getattr(a, f"{phase}_time") for a in attempts) if t is not None]
        if times:
            arr = np.asarray(times, dtype=np.float64)
            metrics[f"{PREFIX}/{phase}_time_mean"] = float(np.mean(arr))
            metrics[f"{PREFIX}/{phase}_time_p90"] = float(np.percentile(arr, 90))
            metrics[f"{PREFIX}/{phase}_time_max"] = float(np.max(arr))
    for attempt in attempts:
        if attempt.exception is not None:
            key = f"{PREFIX}/num_failed_attempts/{attempt.exception}"
            metrics[key] = metrics.get(key, 0) + 1
    metrics.update(_tool_use_metrics([a.tool_use for a in attempts if a.tool_use is not None]))
    return metrics


def _tool_use_metrics(uses: List[ToolUse]) -> Dict[str, float]:
    """Averages over the attempts with a trajectory, masked and failed ones included; the patch counts only
    when some attempt called ``apply_patch``."""
    if not uses:
        return {}
    metrics = {
        f"{PREFIX}/avg_agent_turns": float(np.mean([u.agent_turns for u in uses])),
        f"{PREFIX}/avg_tool_calls": float(np.mean([u.tool_calls for u in uses])),
        f"{PREFIX}/frac_attempts_without_tool_calls": float(np.mean([u.tool_calls == 0 for u in uses])),
    }
    applied = sum(u.patches_applied for u in uses)
    rejected = sum(u.patches_rejected for u in uses)
    if applied + rejected:
        metrics[f"{PREFIX}/avg_patches_applied"] = applied / len(uses)
        metrics[f"{PREFIX}/avg_patches_rejected"] = rejected / len(uses)
        metrics[f"{PREFIX}/patch_success_rate"] = applied / (applied + rejected)
    return metrics


def tool_use(trajectory: Mapping[str, Any]) -> ToolUse:
    """The turns and tool calls of an ATIF trajectory: each agent step is a turn, and an ``apply_patch`` call
    whose output doesn't report the patch applied counts as rejected."""
    use = ToolUse()
    for step in trajectory["steps"]:
        if step.get("source") != "agent":
            continue
        use.agent_turns += 1
        outputs = {
            result.get("source_call_id"): str(result.get("content") or "")
            for result in (step.get("observation") or {}).get("results") or ()
        }
        for call in step.get("tool_calls") or ():
            use.tool_calls += 1
            if call.get("function_name") == APPLY_PATCH:
                if PATCH_APPLIED in outputs.get(call.get("tool_call_id"), ""):
                    use.patches_applied += 1
                else:
                    use.patches_rejected += 1
    return use


def read_tool_use(trial_uri: Optional[str]) -> Optional[ToolUse]:
    """The attempt's ``ToolUse``, or None when its trial directory isn't local or has no readable trajectory
    (e.g. a sandbox that never started, or an agent that writes none)."""
    uri = urlparse(trial_uri or "")
    if uri.scheme != "file":
        return None
    path = Path(url2pathname(uri.path)) / TRAJECTORY_PATH
    try:
        with open(path) as f:
            return tool_use(json.load(f))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        logger.warning(
            f"{path}: not a readable agent trajectory ({type(error).__name__}: {error}), left out of tool metrics"
        )
        return None


def _seconds(timing: Any) -> Optional[float]:
    """A Harbor ``TimingInfo``'s duration, or None if it lacks either timestamp."""
    started = getattr(timing, "started_at", None)
    finished = getattr(timing, "finished_at", None)
    if started is None or finished is None:
        return None
    try:
        return (finished - started).total_seconds()
    except (TypeError, AttributeError):
        return None
