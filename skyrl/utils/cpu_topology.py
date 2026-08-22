"""Determine usable CPUs from process affinity and cgroup quota."""

import os
from typing import Optional, Tuple

# Container cgroup namespaces expose the current cgroup at these paths. Module-level constants
# let tests replace them with fixtures.
CGROUP_V2_CPU_MAX_PATH = "/sys/fs/cgroup/cpu.max"
CGROUP_V1_CPU_QUOTA_PATH = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
CGROUP_V1_CPU_PERIOD_PATH = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"

# cgroup v2 uses this literal for an unlimited quota.
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
    """Return whole CPUs permitted by CFS, or ``None`` when no quota applies."""
    limits = _read_cgroup_v2_cpu_max() or _read_cgroup_v1_cpu_max()
    if limits is None:
        return None
    quota, period = limits
    if quota <= 0 or period <= 0:
        return None
    # Floor a fractional allowance, but a sub-CPU quota still gets one worker.
    return max(1, int(quota // period))


def permitted_cpu_cores() -> int:
    """Return the lesser of the process affinity and cgroup quota."""
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count() or 1
    quota = cgroup_cpu_quota()
    if quota is None:
        return affinity
    return min(affinity, quota)


def pool_workers(*, cap: int, reserved: int, cores: Optional[int] = None) -> int:
    """Size a pool from permitted cores, a cap, and a reserve for colocated processes."""
    if cap < 1:
        raise ValueError(f"pool cap must be positive, got {cap}")
    if reserved < 0:
        raise ValueError(f"reserved cores must be non-negative, got {reserved}")
    if cores is None:
        cores = permitted_cpu_cores()
    return max(1, min(cap, cores - reserved))
