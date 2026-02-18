"""Shared test utilities and fixtures for tinker tests."""

import time
from typing import Callable


def wait_for_condition(
    condition_fn: Callable[[], bool],
    timeout_sec: float = 10,
    poll_interval_sec: float = 0.1,
) -> bool:
    """Poll until condition_fn returns True or timeout is reached. Returns True if condition was met."""
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        if condition_fn():
            return True
        time.sleep(poll_interval_sec)
    return False
