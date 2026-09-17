"""``SingleAsyncEvalDispatcher``: evals run on an ``EvalBackend``, one at a time, without blocking
the training loop.

The dispatcher owns *when and how many evals are outstanding*: the queue bound and the overflow
policy from ``trainer.eval_dispatch``. The backend owns *what weight version is loaded where*. The
trainer owns the HF exports the evals load, and asks ``pending_steps`` which ones it may not delete.
"""

import asyncio
import os
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from loguru import logger

from skyrl.train.eval.backend import EvalBackend
from skyrl.train.eval.dispatcher import BaseEvalDispatcher, OnEvent, RunEval
from skyrl.train.eval.types import EvalRequest, EvalResult, EvalSkip

if TYPE_CHECKING:
    from skyrl.train.config.config import TrainerConfig


class SingleAsyncEvalDispatcher(BaseEvalDispatcher):
    """Non-blocking eval against an ``EvalBackend``, one eval running at a time.

    ``submit`` admits the eval and returns; the eval runs as an ``asyncio`` task that syncs the
    step's export onto the backend, evaluates, and settles into an ``EvalResult`` -- a skipped
    point rather than a training crash for every failure. Exactly one eval runs at a time
    (``asyncio.Lock``): the reserved engine group holds one weight version, so the others wait,
    each with its export on disk, and lateness is a reported ``eval/lag_steps``, never a wrong
    number. Parallel evals are a different dispatcher, not a knob here.

    When the queue already holds ``max_queue_size`` evals -- admitted and not yet collected, the
    running one included -- ``overflow_policy`` decides: ``backpressure`` blocks the caller on the
    oldest eval, exactly one, then admits; ``skip`` drops the point and records ``busy``. The
    final-step eval is submitted with ``force=True`` and is never skipped.

    Callbacks fire on the caller's thread only: ``on_eval_start`` at admission, ``on_eval_end``
    when a result is collected (``get_completed`` / ``drain`` / a backpressure settle inside
    ``submit``), with the async-only keys ``eval/lag_steps`` and ``eval/duration_seconds`` already
    folded into the metrics.
    """

    runs_inline = False

    def __init__(
        self,
        trainer_cfg: "TrainerConfig",
        *,
        backend: EvalBackend,
        run_eval: RunEval,
        on_event: OnEvent,
        current_step: Callable[[], int],
    ):
        super().__init__(run_eval, on_event=on_event)
        # The whole trainer config, not just ``eval_dispatch``: the export an eval loads is derived
        # from ``export_path`` and ``policy.model.path`` at submit time.
        self._trainer_cfg = trainer_cfg
        self._cfg = trainer_cfg.eval_dispatch
        self._backend = backend
        # Late-bound: the loop's global_step at collection time, for eval/lag_steps.
        self._current_step = current_step
        # One synced version at a time. asyncio.Lock binds to its loop on first use, so building it
        # here, outside the loop, is fine. Waiters are woken FIFO.
        self._run_lock = asyncio.Lock()
        # (global_step, task) in admission order. Holding the reference matters: an unreferenced
        # task can be garbage-collected mid-flight.
        self._pending: Deque[Tuple[int, asyncio.Task]] = deque()
        self._done: List[EvalResult] = []

    async def submit(self, global_step: int, *, force: bool = False, **_: Any) -> None:
        # Caller kwargs are deliberately not forwarded. The sync loop passes its vllm_metrics_scraper,
        # which measures the TRAINING engines' eval window; that window is empty under this dispatcher.
        # TODO (kyuds): figure out good way to propagate caller kwargs.
        self._collect_finished()
        if len(self._pending) >= self._cfg.max_queue_size:
            if self._cfg.overflow_policy == "skip" and not force:
                logger.warning(f"eval at step {global_step} skipped: {len(self._pending)} evals already queued")
                self._done.append(EvalResult(global_step=global_step, skipped_reason="busy"))
                return
            # Backpressure: block on the oldest queued eval, exactly one, then admit.
            await self._settle(self._pending.popleft()[1])
        # The HF checkpoint this eval loads. Step 0 is the launch weights (a local directory or a hub
        # id): ``eval_before_train`` on a fresh run needs no export. Any other step is the run's own
        # export, the directory ``save_models()`` writes at that step -- which relies on both training
        # loops running the save before the same step's eval site.
        if global_step == 0:
            export_dir = self._trainer_cfg.policy.model.path
        else:
            export_dir = os.path.join(self._trainer_cfg.export_path, f"global_step_{global_step}", "policy")
        req = EvalRequest(global_step=global_step, export_dir=export_dir)
        self._on_event("on_eval_start", global_step=global_step)
        task = asyncio.create_task(self._run_one(req, admitted_at=time.monotonic()), name=f"eval-{global_step}")
        self._pending.append((global_step, task))

    def pending_steps(self) -> Set[int]:
        """Steps whose export must not be deleted yet: queued or running, not yet collected.

        A snapshot, so the fully-async trainer's save thread can read it while the loop's coroutine
        (the only writer of the queue) is suspended on that thread.
        """
        return {step for step, _ in tuple(self._pending)}

    async def _run_one(self, req: EvalRequest, admitted_at: float) -> EvalResult:
        """Never raises: every outcome is an ``EvalResult``, so a failed eval is a skipped point."""
        metrics: Dict[str, float] = {}
        skipped_reason: Optional[str] = None
        lease = None
        try:
            async with self._run_lock:
                lease = await self._backend.sync(req)
                metrics = await self._run_eval(generator=lease.generator, global_step=req.global_step)
        except EvalSkip as e:
            skipped_reason = e.reason
        except asyncio.CancelledError:
            # Ours, from close(): absorbed deliberately so the task settles instead of unwinding.
            skipped_reason = "cancelled"
        except Exception:
            logger.exception(f"eval at step {req.global_step} crashed")
            skipped_reason = "crashed"
        finally:
            # Counts the time spent queued behind a straggler, deliberately; excludes the release.
            duration = time.monotonic() - admitted_at
            if lease is not None:
                try:
                    await asyncio.shield(self._backend.release(lease))
                except Exception:
                    logger.exception(f"releasing the eval lease for step {req.global_step} failed")
        return EvalResult(
            global_step=req.global_step,
            metrics=metrics,
            skipped_reason=skipped_reason,
            duration_seconds=duration,
        )

    def _collect_finished(self) -> None:
        # Head only: with one eval running at a time, completion order is submission order, and
        # reaping from the head keeps results in that order.
        while self._pending and self._pending[0][1].done():
            self._finish(self._pending.popleft()[1])

    async def _settle(self, task: asyncio.Task) -> None:
        await task  # _run_one never raises
        self._finish(task)

    def _finish(self, task: asyncio.Task) -> None:
        res = task.result()
        if res.skipped_reason is None:
            res.metrics["eval/lag_steps"] = self._current_step() - res.global_step
            res.metrics["eval/duration_seconds"] = res.duration_seconds
        self._on_event("on_eval_end", global_step=res.global_step, metrics=res.metrics)
        self._done.append(res)

    def get_completed(self) -> List[EvalResult]:
        self._collect_finished()
        done, self._done = self._done, []
        return done

    async def drain(self) -> List[EvalResult]:
        while self._pending:
            await self._settle(self._pending.popleft()[1])
        return self.get_completed()

    async def close(self) -> None:
        """Cancel what is in flight and release the backend.

        May raise: the trainer's ``finally`` wraps this call, so an exception here cannot mask the
        one that ended training.
        """
        for _, task in self._pending:
            task.cancel()
        await asyncio.gather(*(task for _, task in self._pending), return_exceptions=True)
        self._pending.clear()
        await self._backend.close()
