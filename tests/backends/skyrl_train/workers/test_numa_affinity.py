"""CPU tests for ``DistributedTorchRayActor._set_numa_affinity``."""

import ctypes.util

from skyrl.backends.skyrl_train.workers import worker as worker_module
from skyrl.backends.skyrl_train.workers.worker import DistributedTorchRayActor


def test_numa_affinity_is_skipped_without_libnuma(monkeypatch):
    monkeypatch.setattr(worker_module, "_SET_AFFINITY", False)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
    actor = DistributedTorchRayActor(
        world_size=1,
        rank=0,
        local_rank=0,
        master_addr="localhost",
        master_port=12345,
        sequence_parallel_size=1,
    )
    actor._set_numa_affinity(0)
    assert worker_module._SET_AFFINITY
