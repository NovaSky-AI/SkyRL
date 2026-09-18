"""Types shared by the eval dispatchers and the eval backends."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from skyrl.train.generators.base import GeneratorInterface


@dataclass
class EvalResult:
    """One settled eval point.

    ``global_step`` is the step whose weights were evaluated -- under an asynchronous dispatcher,
    older than the step it is collected on. ``metrics`` is ``evaluate()``'s dict, namespaced
    ``eval/*``; empty when ``skipped_reason`` is set.
    """

    global_step: int
    metrics: Dict[str, float] = field(default_factory=dict)
    skipped_reason: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class EvalRequest:
    """One eval to run on a backend.

    ``global_step`` is the identity: at most one eval is triggered per step. ``export_dir`` is the
    HF checkpoint to load -- the run's export for that step (a local path or a cloud URI the
    backend fetches), or ``policy.model.path`` for step 0 (a local directory or a hub id).
    """

    global_step: int
    export_dir: str


@dataclass
class EvalLease:
    """A lease on a backend for one eval, held from ``EvalBackend.sync`` to ``EvalBackend.release``.

    ``generator`` is bound to the synced weight version. ``handle`` is whatever the backend needs
    to release the lease: nothing for the reserved engine group, a deployment handle for a hosted
    provider.
    """

    generator: "GeneratorInterface"
    handle: Any = None


class EvalSkip(Exception):
    """Skip this eval point with attribution instead of crashing.

    ``reason`` becomes the ``eval/skipped_{reason}`` marker the trainer writes at the evaluated step.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
