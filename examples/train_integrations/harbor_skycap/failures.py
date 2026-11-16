"""Harness-specific failure detection, separate from the generator's training policy."""

from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from urllib.request import url2pathname

if TYPE_CHECKING:
    from harbor.models.trial.result import TrialResult


class TrialFailure(str, Enum):
    """Failures understood by the existing rollout policy."""

    CONTEXT_LENGTH = "context_length"
    AGENT_TIMEOUT = "agent_timeout"
    ERROR = "error"


FailureClassifier = Callable[["TrialResult"], TrialFailure | None]


class FailureClassifierError(ValueError):
    """A classifier failed, so rerunning the trial would not fix the classification."""


CONTEXT_LENGTH_EXCEPTIONS = {"ContextLengthExceededError", "ContextWindowExceededError"}
CONTEXT_LENGTH_OUTPUT = re.compile(r"context\s*_length\s*_exceeded|ContextWindowExceededError")
AGENT_LOG_TAIL_BYTES = 64 * 1024


def default_failure_classifier(result: TrialResult) -> TrialFailure | None:
    """Keep the existing timeout/context-limit detection; leave other results to the verifier path.

    Installed agents can expose a context limit only through their output. Harbor's
    NonZeroAgentExitCodeError keeps its first 1000 characters, so also check the
    agent log's tail. A custom classifier can replace this heuristic entirely.
    Missing or unreadable local logs are not evidence of a context limit.
    """
    info = result.exception_info
    exception = info.exception_type if info else None
    if exception == "AgentTimeoutError":
        return TrialFailure.AGENT_TIMEOUT
    if exception in CONTEXT_LENGTH_EXCEPTIONS:
        return TrialFailure.CONTEXT_LENGTH
    if exception != "NonZeroAgentExitCodeError":
        return None
    if CONTEXT_LENGTH_OUTPUT.search(info.exception_message or ""):
        return TrialFailure.CONTEXT_LENGTH
    uri = urlparse(getattr(result, "trial_uri", None) or "")
    if uri.scheme != "file":
        return None
    for log in sorted(Path(url2pathname(uri.path), "agent").glob("*.txt")):
        try:
            with log.open("rb") as f:
                f.seek(max(0, log.stat().st_size - AGENT_LOG_TAIL_BYTES))
                if CONTEXT_LENGTH_OUTPUT.search(f.read().decode(errors="replace")):
                    return TrialFailure.CONTEXT_LENGTH
        except OSError:
            continue
    return None


def load_failure_classifier(spec: str | None) -> FailureClassifier:
    """Load a synchronous ``pkg.module:function`` at startup, or use the default."""
    if spec is None:
        return default_failure_classifier
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError(f"a failure classifier must be 'pkg.module:function', not {spec!r}")
    classifier = getattr(importlib.import_module(module), name)
    if (
        not callable(classifier)
        or inspect.iscoroutinefunction(classifier)
        or inspect.iscoroutinefunction(classifier.__call__)
    ):
        raise TypeError(f"failure classifier {spec!r} must be a synchronous callable")
    return classifier


def classify_failure(classifier: FailureClassifier, result: TrialResult) -> TrialFailure | None:
    """Reject classifier bugs instead of silently training on a misclassified result."""
    try:
        failure = classifier(result)
        if failure is not None and not isinstance(failure, TrialFailure):
            raise TypeError(f"expected TrialFailure or None, got {failure!r}")
        return failure
    except Exception as error:
        raise FailureClassifierError(f"failure classifier failed: {error}") from error
