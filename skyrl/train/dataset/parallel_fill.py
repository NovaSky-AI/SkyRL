"""Fill a large controller-side batch buffer from a locally-sized thread pool.

Controller-side collation runs single-threaded on the whole global batch before DP sharding, so
its cost lands on every training step and does not shard away. For a multi-GiB buffer the fill is
dominated by first-touch page faults on a fresh mapping rather than by the copies themselves, and
the trainer driver runs under ``@ray.remote(num_cpus=1)`` with ``OMP_NUM_THREADS=1``, so torch
cannot fault those pages in parallel on its own.

``torch.set_num_threads`` is the wrong lever here: it is process-global, so on the fully-async
path it would also reach the generation coroutines running concurrently with collation. A local
pool keeps the effect scoped to this fill. Each callback owns a disjoint row range and the copies
release the GIL.
"""

import functools
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from skyrl.utils.cpu_topology import pool_workers

# Copy throughput keeps scaling to the cap; past it the extra threads only take cores from
# colocated actors.
MAX_FILL_WORKERS = 32
# Leave room for raylet, the GCS, the dashboard and the log monitor in the same cgroup.
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
    """Call ``fill_row(index)`` for every index in ``range(num_rows)``, in parallel.

    ``fill_row`` must write a row range that no other index touches; nothing here serialises
    overlapping writes.
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
        # list() forces the map so an exception in a worker surfaces here rather than being dropped.
        list(pool.map(fill_row, range(num_rows)))
