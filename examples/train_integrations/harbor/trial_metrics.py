"""Per-attempt metrics of Harbor trials: how long each attempt spent in each of Harbor's phases
(sandbox start, agent, verifier), how many attempts were retries, and how many failed, by
exception type.

They describe the environment run, not how its tokens were collected, so every generator that
runs Harbor trials logs the same ``generate/harbor/*`` keys. They also show a slow or failing
sandbox start that a retry rescued, which the trajectory-level counts can't.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PHASES = ("environment_setup", "agent_execution", "verifier")
PREFIX = "generate/harbor"


@dataclass
class Attempt:
    """One attempt at a trial: its phase durations in seconds (None where Harbor has no
    timestamps), and the exception it ended with, if any."""

    environment_setup_time: Optional[float] = None
    agent_execution_time: Optional[float] = None
    verifier_time: Optional[float] = None
    exception: Optional[str] = None


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
        """Harbor's phase times and exception for the current attempt, from its ``TrialResult``.

        Harbor sets ``environment_setup.finished_at`` in a ``finally``, so an attempt whose
        sandbox start timed out still has a duration.
        """
        attempt = self.attempts[-1]
        for phase in PHASES:
            setattr(attempt, f"{phase}_time", _seconds(getattr(result, phase, None)))
        exception_info = getattr(result, "exception_info", None)
        attempt.exception = exception_info.exception_type if exception_info else None

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
    return metrics


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
