"""Eval dispatch: the seam between the training loops and ``evaluate()``.

The loops make three calls -- submit an eval of the current step, collect the results that have
settled, drain everything outstanding at a join point -- and write each returned result to the
tracker at the step it evaluated. The dispatcher runs the eval and fires the eval callbacks; it
never touches the tracker. ``BlockingEvalDispatcher`` completes the eval inside ``submit``; it is
the pre-dispatcher behaviour, inline on the training engines.
"""

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

RunEval = Callable[..., Awaitable[Dict[str, float]]]
"""``RayPPOTrainer.eval``'s signature. Dispatchers pass keyword arguments only."""
OnEvent = Callable[..., None]
"""``RayPPOTrainer._fire``: ``(event_name, **callback_input_fields)``."""


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


class BaseEvalDispatcher(abc.ABC):
    """Schedules evals and hands back their results.

    Contract: ``submit`` may return before the eval has run; ``get_completed`` never blocks and
    returns the results that have settled since the last call; ``drain`` blocks until every
    submitted eval has settled and returns those results. Every result is returned exactly once;
    the caller writes it to the tracker at ``EvalResult.global_step``.

    Callbacks are the dispatcher's to fire, through ``on_event``, and always on the caller's
    thread -- never from a background task, where a callback would run at whatever point the loop
    happens to be suspended and its exception would surface late or not at all. ``on_eval_start``
    fires when an eval is admitted, before it runs; a skipped eval fires nothing. ``on_eval_end``
    fires for every admitted eval, no later than the ``get_completed`` / ``drain`` that returns
    its result -- with empty metrics if a dispatcher settled it as a skip after admission. Both
    carry the evaluated step, not the loop's current one.
    """

    def __init__(self, run_eval: RunEval, *, on_event: OnEvent):
        self._run_eval = run_eval
        self._on_event = on_event

    @abc.abstractmethod
    async def submit(self, global_step: int, **kwargs: Any) -> None:
        """Request an eval of the weights as of ``global_step``. ``kwargs`` go to the unit of work."""

    @abc.abstractmethod
    def get_completed(self) -> List[EvalResult]:
        """Return the results that have settled since the last call. Never blocks."""

    @abc.abstractmethod
    async def drain(self) -> List[EvalResult]:
        """Block until every outstanding eval settles; return those results."""

    async def close(self) -> None:
        """Release anything the dispatcher owns. Called once, after the final ``drain``."""


class BlockingEvalDispatcher(BaseEvalDispatcher):
    """Runs the eval inline on the training engines.

    ``submit`` fires both callbacks around the unit of work and awaits it, so the result is
    returned by the ``get_completed`` that follows. Exceptions propagate out of ``submit`` -- an
    eval crash still stops training.
    """

    def __init__(self, run_eval: RunEval, *, on_event: OnEvent):
        super().__init__(run_eval, on_event=on_event)
        self._done: List[EvalResult] = []

    async def submit(self, global_step: int, **kwargs: Any) -> None:
        self._on_event("on_eval_start", global_step=global_step)
        started = time.monotonic()
        metrics = await self._run_eval(**kwargs)
        duration = time.monotonic() - started
        self._on_event("on_eval_end", global_step=global_step, metrics=metrics)
        self._done.append(EvalResult(global_step=global_step, metrics=metrics, duration_seconds=duration))

    def get_completed(self) -> List[EvalResult]:
        done, self._done = self._done, []
        return done

    async def drain(self) -> List[EvalResult]:
        return self.get_completed()
