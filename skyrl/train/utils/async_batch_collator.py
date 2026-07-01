"""Single-slot async double-buffer for deterministic per-step collation."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional


class AsyncBatchCollator:
    """Run ``compute(step)`` in a one-worker, one-future buffer.

    The caller may only submit steps whose inputs will not change before
    consumption. ``get`` checks the expected step so stale batches fail loudly.
    """

    def __init__(self, compute: Callable[[int], Any], thread_name_prefix: str = "batch-collate"):
        self._compute = compute
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name_prefix)
        self._future: Optional[Future] = None
        self._pending_step: Optional[int] = None

    def submit(self, step: int) -> None:
        """Schedule ``compute(step)``; call ``get`` before submitting again."""
        assert self._future is None, (
            f"collate-ahead slot already occupied (pending step {self._pending_step}); "
            f"call get() before submitting step {step}"
        )
        self._pending_step = step
        self._future = self._executor.submit(self._compute, step)

    def has_pending(self) -> bool:
        return self._future is not None

    def pending_step(self) -> Optional[int]:
        return self._pending_step

    def get(self, expected_step: int) -> Any:
        """Return the in-flight batch for ``expected_step``."""
        assert self._future is not None, "get() called with no in-flight batch"
        assert self._pending_step == expected_step, (
            f"collated-ahead step {self._pending_step} != expected step {expected_step}; "
            f"refusing to serve a mismatched batch"
        )
        future = self._future
        self._future = None
        self._pending_step = None
        # Propagates any exception raised inside the worker thread.
        return future.result()

    def clear(self) -> None:
        """Drain and discard the in-flight batch, propagating worker errors."""
        if self._future is not None:
            self._future.result()
        self._future = None
        self._pending_step = None

    def shutdown(self) -> None:
        """Drain any in-flight batch and join the worker thread."""
        self.clear()
        self._executor.shutdown(wait=True)
