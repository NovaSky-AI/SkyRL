import multiprocessing

import pytest
import ray

from skyrl.utils.worker_setup import worker_setup_fn


@pytest.fixture(scope="module")
def ray_with_setup_hook():
    """Initialize Ray with the worker_process_setup_hook."""
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        runtime_env={
            "worker_process_setup_hook": "skyrl.utils.worker_setup.worker_setup_fn",
        },
    )
    yield
    ray.shutdown()


def test_worker_setup_fn_sets_spawn():
    """Test that worker_setup_fn sets the mp start method to spawn."""
    # Reset to default first
    multiprocessing.set_start_method("fork", force=True)
    worker_setup_fn()
    assert multiprocessing.get_start_method() == "spawn"


def test_worker_setup_fn_idempotent():
    """Test that calling worker_setup_fn twice doesn't raise."""
    multiprocessing.set_start_method("spawn", force=True)
    worker_setup_fn()  # should not raise
    assert multiprocessing.get_start_method() == "spawn"


@ray.remote
def _get_mp_start_method():
    return multiprocessing.get_start_method()


def test_worker_setup_hook_applied_in_ray_worker(ray_with_setup_hook):
    """Test that Ray workers have mp start method set to spawn via the hook."""
    result = ray.get(_get_mp_start_method.remote())
    assert result == "spawn"
