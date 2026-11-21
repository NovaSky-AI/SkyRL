"""CPU tests for ``DistributedTorchRayActor._set_numa_affinity``."""

import ctypes.util

from skyrl.backends.skyrl_train.workers import worker as worker_module
from skyrl.backends.skyrl_train.workers.worker import DistributedTorchRayActor


def test_numa_affinity_is_skipped_without_libnuma(monkeypatch):
    monkeypatch.setattr(worker_module, "_SET_AFFINITY", False)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
    # The constructor writes os.environ and replaces the global logging handlers.
    # _set_numa_affinity needs no instance state before the libnuma lookup.
    actor = object.__new__(DistributedTorchRayActor)
    actor._set_numa_affinity(0)
    assert worker_module._SET_AFFINITY
