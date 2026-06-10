"""Single-slot async double-buffer for a deterministic per-step producer.

When a training loop's per-step input is built by a CPU-heavy producer that is
deterministic given the current loop state (e.g. a slice + collate over an
in-memory dataset), the producer for step ``N+1`` can run on a background
thread while step ``N``'s forward/backward runs on the GPU, hiding the producer
latency under GPU compute.

:class:`BatchPrefetcher` is the generic mechanism: it owns a single-worker
executor and a one-slot future, and exposes a strict submit/get contract that
turns any mis-wiring into a loud failure rather than a silent corruption of the
input order. The caller supplies a ``compute(step) -> batch`` producer and is
responsible for only ever prefetching a step whose input is knowable in advance
(i.e. no state change happens between submit and consume).
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional


class BatchPrefetcher:
    """Single-slot async double-buffer for a deterministic per-step producer.

    Correctness contract:

    * The producer is a single function ``compute(step) -> batch`` supplied by
      the caller; it must read the loop's *current* state. The caller only ever
      submits a step whose input is knowable while the current step runs — i.e.
      a step for which no state change happens between submit and consume. If
      the loop is about to mutate the producer's inputs (e.g. a dataset
      reshuffle at an epoch boundary), it must :meth:`clear` the in-flight batch
      first and compute the next step synchronously.
    * At most one batch is in flight (``max_workers=1``). The submitted step is
      recorded so :meth:`get` can assert the retrieved batch matches the step
      the caller expects.
    * The producer must construct fresh outputs per call and mutate no shared
      state, so the worker thread genuinely overlaps with the GPU stream
      (NumPy/torch release the GIL during the heavy array ops).

    This component owns only its executor and a one-slot future; it holds no
    reference to caller state, which keeps the threading surface tiny.
    """

    def __init__(self, compute: Callable[[int], Any], thread_name_prefix: str = "batch-prefetch"):
        self._compute = compute
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name_prefix)
        self._future: Optional[Future] = None
        self._pending_step: Optional[int] = None

    def submit(self, step: int) -> None:
        """Schedule the producer for ``step`` on the background thread.

        Must not be called while a batch is already in flight (the single-slot
        invariant); drain via :meth:`get` before resubmitting.
        """
        assert self._future is None, (
            f"prefetch slot already occupied (pending step {self._pending_step}); "
            f"call get() before submitting step {step}"
        )
        self._pending_step = step
        self._future = self._executor.submit(self._compute, step)

    def has_pending(self) -> bool:
        return self._future is not None

    def pending_step(self) -> Optional[int]:
        return self._pending_step

    def get(self, expected_step: int) -> Any:
        """Block until the in-flight batch is ready and return it.

        Asserts the in-flight batch was produced for ``expected_step`` so a
        mis-wired submit/get pairing fails loudly instead of silently feeding
        the wrong batch into training.
        """
        assert self._future is not None, "get() called with no pending prefetch"
        assert self._pending_step == expected_step, (
            f"prefetched step {self._pending_step} != expected step {expected_step}; "
            f"refusing to serve a mismatched batch"
        )
        future = self._future
        self._future = None
        self._pending_step = None
        # Propagates any exception raised inside the worker thread.
        return future.result()

    def clear(self) -> None:
        """Drop any in-flight batch.

        Waits for the worker to finish (so it cannot still be reading caller
        state) and discards the result. Used before the caller mutates the
        producer's inputs, so a prefetch scheduled against the old state can
        never be consumed.
        """
        if self._future is not None:
            self._future.result()
        self._future = None
        self._pending_step = None

    def shutdown(self) -> None:
        """Drain any in-flight batch and join the worker thread."""
        self.clear()
        self._executor.shutdown(wait=True)
