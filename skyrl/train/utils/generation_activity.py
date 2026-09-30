"""Measure the union of overlapping generation calls."""

import time
from contextlib import contextmanager
from typing import Callable, Optional


class GenerationActivity:
    """Track active wall time without summing concurrent call durations."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, publish: Optional[Callable[[int], None]] = None):
        self._clock = clock
        self._publish = publish
        self._count = 0
        self._since = None
        self._total = 0.0

    @property
    def seconds(self) -> float:
        """Return completed and currently active interval duration."""
        return self._total + (self._clock() - self._since if self._since is not None else 0.0)

    @contextmanager
    def active(self):
        """Mark a call active and always close it on exception or cancellation."""
        if self._count == 0:
            self._since = self._clock()
        self._count += 1
        self._emit()
        try:
            yield
        finally:
            self._count -= 1
            if self._count == 0:
                self._total += self._clock() - self._since
                self._since = None
            self._emit()

    def _emit(self):
        if self._publish is not None:
            self._publish(self._count)
