from types import SimpleNamespace

import pytest
import ray

from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.backends.skyrl_train.workers.worker_dispatch import GPUState, WorkerDispatch


class _FakeActorGroup:
    def __init__(self):
        self.shutdown_called = False
        self.offload_called = False

    @property
    def is_shutdown(self):
        return False

    def offload_to_cpu(self):
        self.offload_called = True

    def can_restore_init_model(self):
        return True

    def shutdown(self):
        self.shutdown_called = True


def _dispatch(*, hard_evict: bool = False, barrier: bool = True):
    placement = SimpleNamespace(
        colocated_worker_memory_barrier=barrier,
        colocated_worker_residual_hbm_threshold_gb=1.0,
        colocated_ref_hard_evict_on_breach=hard_evict,
    )
    dispatch = object.__new__(WorkerDispatch)
    dispatch.cfg = SimpleNamespace(trainer=SimpleNamespace(placement=placement))
    dispatch._actor_groups = {"ref": _FakeActorGroup()}
    dispatch._gpu_state = {"ref": GPUState(model_on_gpu=True)}
    return dispatch


def test_inactive_offload_skips_extra_stats_call_when_barrier_is_disabled():
    dispatch = _dispatch(barrier=False)
    dispatch.release_cuda_memory = lambda _model: pytest.fail("release_cuda_memory should not be called")

    dispatch._offload_inactive_model("ref")

    assert dispatch._actor_groups["ref"].offload_called
    assert dispatch._gpu_state["ref"] == GPUState()


def test_inactive_offload_collects_stats_only_for_enabled_barrier():
    dispatch = _dispatch()
    calls = []
    dispatch.release_cuda_memory = lambda model: calls.append(model) or [{"nvml_used_bytes": 0}]

    dispatch._offload_inactive_model("ref")

    assert calls == ["ref"]
    assert dispatch._actor_groups["ref"].offload_called
    assert dispatch._gpu_state["ref"] == GPUState()


def test_worker_residual_hbm_uses_largest_available_metric():
    assert (
        WorkerDispatch._worker_residual_hbm_bytes({"nvml_used_bytes": 0, "reserved_bytes": 123, "allocated_bytes": 45})
        == 123
    )
    assert WorkerDispatch._worker_residual_hbm_bytes({"nvml_error": "unavailable"}) is None
    assert WorkerDispatch._worker_residual_hbm_bytes({"nvml_error": "unavailable", "reserved_bytes": 123}) is None


def test_worker_memory_barrier_rejects_unknown_measurement():
    dispatch = _dispatch()

    with pytest.raises(RuntimeError, match="missing memory measurement"):
        dispatch._enforce_inactive_worker_memory_barrier("ref", [{"nvml_error": "unavailable"}])


def test_worker_memory_barrier_hard_evicts_restorable_ref_on_breach():
    dispatch = _dispatch(hard_evict=True)

    dispatch._enforce_inactive_worker_memory_barrier(
        "ref",
        [{"nvml_used_bytes": 2 * 1024**3, "reserved_bytes": 2 * 1024**3}],
    )

    assert dispatch._actor_groups["ref"].shutdown_called
    assert dispatch._gpu_state["ref"] == GPUState()


def test_hard_evicted_group_restarts_only_after_colocated_model_is_offloaded():
    events = []

    class _ResidentPolicyGroup:
        is_shutdown = False

        def offload_to_cpu(self):
            events.append("offload_policy")

    class _EvictedRefGroup:
        is_shutdown = True

        def restart_actors(self):
            events.append("restart_ref")
            self.is_shutdown = False

        def restore_init_model(self):
            events.append("init_ref")

    dispatch = object.__new__(WorkerDispatch)
    dispatch.colocate_all = True
    dispatch.colocate_policy_ref = False
    dispatch.cfg = SimpleNamespace(
        trainer=SimpleNamespace(placement=SimpleNamespace(colocated_worker_memory_barrier=False))
    )
    dispatch._actor_groups = {
        "policy": _ResidentPolicyGroup(),
        "ref": _EvictedRefGroup(),
    }
    dispatch._gpu_state = {
        "policy": GPUState(model_on_gpu=True, optimizer_on_gpu=True),
        "ref": GPUState(),
    }

    dispatch._ensure_on_gpu("ref", need_optimizer=False, need_model=True)

    assert events == ["offload_policy", "restart_ref", "init_ref"]
    assert dispatch._gpu_state["policy"] == GPUState()
    assert dispatch._gpu_state["ref"] == GPUState(model_on_gpu=True, optimizer_on_gpu=False)


def test_actor_group_shutdown_waits_for_actor_death(monkeypatch):
    events = []

    class _RemoteProbe:
        def remote(self):
            events.append("probe")
            return object()

    class _ActorDead(ray.exceptions.RayActorError):
        def __init__(self):
            Exception.__init__(self)

    actor = SimpleNamespace(get_mesh_rank=_RemoteProbe())
    group = object.__new__(PPORayActorGroup)
    group.actor_infos = []
    group._actor_handlers = [actor]

    monkeypatch.setattr(ray, "kill", lambda handle, no_restart: events.append(("kill", handle, no_restart)))

    def _dead_actor(_ref, timeout):
        events.append(("wait", timeout))
        raise _ActorDead()

    monkeypatch.setattr(ray, "get", _dead_actor)

    group.shutdown()

    assert events == [("kill", actor, True), "probe", ("wait", 1.0)]
    assert group._actor_handlers == []
    assert group.actor_infos == []
