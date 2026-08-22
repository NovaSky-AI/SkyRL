"""Fill a controller-side batch buffer with a locally sized thread pool.

The pool parallelises first-touch page faults without changing process-wide torch settings.
Callbacks own disjoint row ranges, and their copies release the GIL.
"""

import functools
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from skyrl.utils.cpu_topology import pool_workers

# Extra threads beyond this cap take cores from colocated actors without improving throughput.
MAX_FILL_WORKERS = 32
# Leave room for Ray services in the same cgroup.
RESERVED_FILL_CORES = 8


@functools.cache
def default_fill_workers() -> int:
    return pool_workers(cap=MAX_FILL_WORKERS, reserved=RESERVED_FILL_CORES)


def fill_batch_rows(
    fill_row: Callable[[int], None],
    num_rows: int,
    *,
    workers: int | None = None,
) -> None:
    """Call ``fill_row`` for every row, possibly in parallel.

    Each callback must write to a disjoint row range.
    """
    if num_rows < 0:
        raise ValueError(f"row count must be non-negative, got {num_rows}")
    if num_rows == 0:
        return
    if workers is None:
        workers = default_fill_workers()
    if workers < 1:
        raise ValueError(f"worker count must be positive, got {workers}")

    workers = min(workers, num_rows)
    if workers == 1:
        for index in range(num_rows):
            fill_row(index)
        return

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="skyrl-batch-fill") as pool:
        # Eagerly consume the map so worker exceptions surface here.
        list(pool.map(fill_row, range(num_rows)))
