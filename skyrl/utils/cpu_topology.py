"""How many CPUs this process may actually keep busy, and how to size a pool from it.

``os.cpu_count()`` reports the machine, not the container. Under Ray a worker runs with
``OMP_NUM_THREADS=1`` and ``torch.get_num_threads() == 1`` without the process itself being
restricted, so neither of those can size a pool either. The two limits that do bind are the
affinity mask (cpuset / taskset pinning) and the CFS quota, and they are independent.
"""

import os
from typing import Optional, Tuple

# A container's own cgroup appears at the root of its cgroup namespace, so these paths already
# describe this process (``/proc/self/cgroup`` reads ``0::/``) and need no prefix join. Module
# level so tests can point them at fixtures.
CGROUP_V2_CPU_MAX_PATH = "/sys/fs/cgroup/cpu.max"
CGROUP_V1_CPU_QUOTA_PATH = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
CGROUP_V1_CPU_PERIOD_PATH = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"

# cgroup v2 spells "no quota" as this literal; v1 spells it as a negative quota.
CGROUP_V2_CPU_MAX_UNLIMITED = "max"


def _read_cgroup_file(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _read_cgroup_v2_cpu_max() -> Optional[Tuple[float, float]]:
    """``(quota, period)`` from cgroup v2 ``cpu.max``, or ``None`` if absent or unlimited."""
    try:
        quota_text, period_text = _read_cgroup_file(CGROUP_V2_CPU_MAX_PATH).split()
        if quota_text == CGROUP_V2_CPU_MAX_UNLIMITED:
            return None
        return float(quota_text), float(period_text)
    except (OSError, ValueError):
        return None


def _read_cgroup_v1_cpu_max() -> Optional[Tuple[float, float]]:
    """``(quota, period)`` from cgroup v1 ``cpu.cfs_*_us``, or ``None`` if absent or unlimited."""
    try:
        quota = float(_read_cgroup_file(CGROUP_V1_CPU_QUOTA_PATH).strip())
        period = float(_read_cgroup_file(CGROUP_V1_CPU_PERIOD_PATH).strip())
    except (OSError, ValueError):
        return None
    if quota < 0:
        return None
    return quota, period


def cgroup_cpu_quota() -> Optional[int]:
    """Whole CPUs this process' CFS quota permits, or ``None`` when no quota applies.

    Kubernetes ``limits.cpu`` becomes this quota, and flytekit's ``pod_spec_from_resources`` ends
    with ``limits = limits or requests``, so a pod declaring only ``requests.cpu`` still carries
    one. ``sched_getaffinity`` cannot see it: a quota caps CPU *time*, not which CPUs are runnable.
    """
    limits = _read_cgroup_v2_cpu_max() or _read_cgroup_v1_cpu_max()
    if limits is None:
        return None
    quota, period = limits
    if quota <= 0 or period <= 0:
        return None
    # Floor a fractional allowance, but a sub-CPU quota still gets one worker.
    return max(1, int(quota // period))


def permitted_cpu_cores() -> int:
    """CPUs this process can actually keep busy: the lesser of its affinity mask and its quota.

    ``sched_getaffinity`` honours cpuset/taskset pinning and is Linux-only; elsewhere
    ``cpu_count`` is the closest read available.
    """
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count() or 1
    quota = cgroup_cpu_quota()
    if quota is None:
        return affinity
    return min(affinity, quota)


def pool_workers(*, cap: int, reserved: int, cores: Optional[int] = None) -> int:
    """Pool size for a CPU-bound thread pool: capped, otherwise permitted cores minus a reserve.

    ``reserved`` leaves room for the colocated processes sharing this cgroup -- under Ray that is
    raylet, the GCS, the dashboard and the log monitor. Oversubscribing a CFS quota buys a few
    percent of throughput for constant throttling.
    """
    if cap < 1:
        raise ValueError(f"pool cap must be positive, got {cap}")
    if reserved < 0:
        raise ValueError(f"reserved cores must be non-negative, got {reserved}")
    if cores is None:
        cores = permitted_cpu_cores()
    return max(1, min(cap, cores - reserved))
