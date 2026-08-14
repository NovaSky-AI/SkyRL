"""
uv run --isolated --extra dev pytest tests/utils/test_cpu_topology.py
"""

import os
from pathlib import Path

import pytest

from skyrl.utils import cpu_topology
from skyrl.utils.cpu_topology import cgroup_cpu_quota, permitted_cpu_cores, pool_workers


@pytest.fixture
def cgroup_paths(monkeypatch, tmp_path: Path):
    """Point every cgroup path at a fixture directory; each file is absent until written."""
    paths = {
        "CGROUP_V2_CPU_MAX_PATH": tmp_path / "cpu.max",
        "CGROUP_V1_CPU_QUOTA_PATH": tmp_path / "cpu.cfs_quota_us",
        "CGROUP_V1_CPU_PERIOD_PATH": tmp_path / "cpu.cfs_period_us",
    }
    for name, path in paths.items():
        monkeypatch.setattr(cpu_topology, name, str(path))
    return paths


def test_cgroup_v2_quota(cgroup_paths):
    cgroup_paths["CGROUP_V2_CPU_MAX_PATH"].write_text("400000 100000\n")

    assert cgroup_cpu_quota() == 4


def test_cgroup_v2_max_literal_is_unlimited(cgroup_paths):
    cgroup_paths["CGROUP_V2_CPU_MAX_PATH"].write_text("max 100000\n")

    assert cgroup_cpu_quota() is None


def test_cgroup_v1_quota(cgroup_paths):
    cgroup_paths["CGROUP_V1_CPU_QUOTA_PATH"].write_text("200000\n")
    cgroup_paths["CGROUP_V1_CPU_PERIOD_PATH"].write_text("100000\n")

    assert cgroup_cpu_quota() == 2


def test_cgroup_v1_negative_quota_is_unlimited(cgroup_paths):
    cgroup_paths["CGROUP_V1_CPU_QUOTA_PATH"].write_text("-1\n")
    cgroup_paths["CGROUP_V1_CPU_PERIOD_PATH"].write_text("100000\n")

    assert cgroup_cpu_quota() is None


def test_missing_cgroup_files(cgroup_paths):
    assert cgroup_cpu_quota() is None


def test_unreadable_cgroup_values(cgroup_paths):
    cgroup_paths["CGROUP_V2_CPU_MAX_PATH"].write_text("not-a-quota 100000\n")
    cgroup_paths["CGROUP_V1_CPU_QUOTA_PATH"].write_text("")
    cgroup_paths["CGROUP_V1_CPU_PERIOD_PATH"].write_text("")

    assert cgroup_cpu_quota() is None


@pytest.mark.parametrize(("quota", "expected"), [(150000, 1), (50000, 1), (100000, 1), (250000, 2)])
def test_fractional_quota_floors_to_at_least_one(cgroup_paths, quota, expected):
    cgroup_paths["CGROUP_V2_CPU_MAX_PATH"].write_text(f"{quota} 100000\n")

    assert cgroup_cpu_quota() == expected


@pytest.mark.parametrize(("affinity", "quota_cpus", "expected"), [(16, 4, 4), (4, 16, 4), (8, 8, 8)])
def test_permitted_cores_is_the_lesser_of_affinity_and_quota(cgroup_paths, monkeypatch, affinity, quota_cpus, expected):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(affinity)))
    cgroup_paths["CGROUP_V2_CPU_MAX_PATH"].write_text(f"{quota_cpus * 100000} 100000\n")

    assert permitted_cpu_cores() == expected


def test_permitted_cores_without_quota_is_the_affinity_mask(cgroup_paths, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(12)))

    assert permitted_cpu_cores() == 12


@pytest.mark.parametrize(
    ("cap", "reserved", "cores", "expected"),
    [
        (32, 8, 64, 32),  # cap binds
        (32, 8, 24, 16),  # reserve binds
        (32, 8, 8, 1),  # reserve would leave nothing, so keep one worker
        (1, 0, 64, 1),  # cap of one
    ],
)
def test_pool_workers(cap, reserved, cores, expected):
    assert pool_workers(cap=cap, reserved=reserved, cores=cores) == expected


def test_pool_workers_defaults_to_permitted_cores(cgroup_paths, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(64)))
    cgroup_paths["CGROUP_V2_CPU_MAX_PATH"].write_text("1200000 100000\n")

    assert pool_workers(cap=32, reserved=4) == 8


@pytest.mark.parametrize("cap", [0, -1])
def test_pool_workers_rejects_non_positive_cap(cap):
    with pytest.raises(ValueError, match="cap must be positive"):
        pool_workers(cap=cap, reserved=0, cores=8)


def test_pool_workers_rejects_negative_reserve():
    with pytest.raises(ValueError, match="reserved cores must be non-negative"):
        pool_workers(cap=8, reserved=-1, cores=8)
