"""Ray resources acquired by a setup step that fails are released before the error propagates."""

import time
from types import SimpleNamespace

import pytest
import ray
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)

from skyrl.backends.skyrl_train.workers import worker
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.train.utils.utils import (
    ResolvedPlacementGroup,
    cleanup_on_error,
    get_ray_pg_ready_with_timeout,
)


@ray.remote
class _WorkerThatFailsToJoin:
    def __init__(self, **kwargs):
        pass

    def get_master_addr_port(self):
        return "127.0.0.1", 0

    def init_worker_process_group(self):
        raise RuntimeError("process group failed")

    def ping(self):
        pass


class _RecordingWorkerType:
    """Spawns ``_WorkerThatFailsToJoin`` actors the way PPORayActorGroup does and keeps their handles."""

    def __init__(self):
        self.handles = []

    def options(self, **options):
        spawn = _WorkerThatFailsToJoin.options(**options).remote

        def remote(**kwargs):
            self.handles.append(spawn(**kwargs))
            return self.handles[-1]

        return SimpleNamespace(remote=remote)


def _died(actor, timeout_s: float = 10) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        try:
            ray.get(actor.ping.remote())
        except ray.exceptions.RayActorError:
            return True
        time.sleep(0.1)
    return False


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_cleanup_on_error_reraises_the_original_error(cleanup_fails):
    calls = []

    def cleanup():
        calls.append("cleanup")
        if cleanup_fails:
            raise RuntimeError("cleanup failed")

    with pytest.raises(ValueError, match="build failed"):
        with cleanup_on_error(cleanup, "Cleanup"):
            raise ValueError("build failed")

    assert calls == ["cleanup"]


def test_pg_ready_timeout_removes_the_placement_group():
    # No node has this many CPUs, so the placement group can never become ready.
    pg = placement_group([{"CPU": 10_000}])

    with pytest.raises(RuntimeError, match="Failed to create placement group"):
        get_ray_pg_ready_with_timeout(pg, timeout=1)

    assert placement_group_table(pg)["state"] == "REMOVED"


@pytest.mark.parametrize("owns_pg", [True, False])
def test_actor_group_that_fails_to_start_releases_what_it_acquired(monkeypatch, owns_pg):
    # CPU stand-in for the GPU bundles the group reserves when it is given no placement group.
    created_pgs = []

    def cpu_placement_group(bundles, strategy):
        created_pgs.append(placement_group([{"CPU": bundle["CPU"]} for bundle in bundles], strategy=strategy))
        return created_pgs[-1]

    monkeypatch.setattr(worker, "placement_group", cpu_placement_group)
    # One bundle for two workers, so the group does not probe the bundles for GPU ids.
    shared_pg = None if owns_pg else placement_group([{"CPU": 1}])
    if shared_pg is not None:
        ray.get(shared_pg.ready())
    worker_type = _RecordingWorkerType()

    with pytest.raises(RuntimeError, match="process group failed"):
        PPORayActorGroup(
            cfg=None,
            num_nodes=1,
            num_gpus_per_node=2,
            ray_actor_type=worker_type,
            pg=None if owns_pg else ResolvedPlacementGroup(shared_pg),
            num_gpus_per_actor=0,
        )

    assert len(worker_type.handles) == 2
    assert all(_died(actor) for actor in worker_type.handles)
    if owns_pg:
        assert placement_group_table(created_pgs[0])["state"] == "REMOVED"
    else:
        # The caller's placement group is left for the caller to remove.
        assert placement_group_table(shared_pg)["state"] == "CREATED"
        remove_placement_group(shared_pg)
