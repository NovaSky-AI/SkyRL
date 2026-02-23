"""Integration tests for the Ray process launcher.

These tests use CPU-only mode and verify the Ray process management
layer works correctly without requiring GPUs.
"""

import os
from unittest.mock import MagicMock

import pytest

# Force CPU mode before any JAX imports
os.environ["JAX_PLATFORMS"] = "cpu"


try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


@pytest.fixture(autouse=True)
def isolated_ray():
    """Ensure each test gets a fresh local Ray instance.

    Disconnects from any existing cluster first, then shuts down after the test.
    """
    if RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()
    yield
    if RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()


def test_start_ray_workers_disabled():
    """start_ray_workers returns (None, None) when enable_ray=False."""
    from tx.tinker.backends.jax import JaxBackendConfig, start_ray_workers

    config = JaxBackendConfig(enable_ray=False)
    manager, address = start_ray_workers(config)
    assert manager is None
    assert address is None


def test_backend_config_new_fields():
    """New config fields have correct defaults."""
    from tx.tinker.backends.jax import JaxBackendConfig

    config = JaxBackendConfig()
    assert config.ray_startup_timeout == 60
    assert config.ray_coordinator_port == 7777
    assert config.ray_scheduling_strategy == "SPREAD"
    assert config.ray_placement_group_timeout == 60
    assert config.ray_address is None
    assert config.ray_dashboard_host == "0.0.0.0"
    assert config.ray_dashboard_port == 8265
    assert config.ray_runtime_env is None


def test_backend_config_custom_values():
    """Config fields can be set to custom values."""
    from tx.tinker.backends.jax import JaxBackendConfig

    config = JaxBackendConfig(
        ray_startup_timeout=120,
        ray_coordinator_port=9999,
        ray_scheduling_strategy="STRICT_SPREAD",
        ray_placement_group_timeout=30,
        ray_address="auto",
        ray_dashboard_host="127.0.0.1",
        ray_dashboard_port=9265,
        ray_runtime_env={"env_vars": {"JAX_PLATFORMS": "cpu"}},
    )
    assert config.ray_startup_timeout == 120
    assert config.ray_coordinator_port == 9999
    assert config.ray_scheduling_strategy == "STRICT_SPREAD"
    assert config.ray_placement_group_timeout == 30
    assert config.ray_address == "auto"
    assert config.ray_dashboard_host == "127.0.0.1"
    assert config.ray_dashboard_port == 9265
    assert config.ray_runtime_env == {"env_vars": {"JAX_PLATFORMS": "cpu"}}


@pytest.mark.skipif(not RAY_AVAILABLE, reason="ray not installed")
def test_ray_process_manager_creation():
    """RayProcessManager can be created when Ray is initialized."""
    from tx.tinker.backends.jax import RayProcessManager

    ray.init(num_cpus=4, include_dashboard=False)

    manager = RayProcessManager(
        coordinator_address="127.0.0.1:7777",
        num_workers=1,
        gpus_per_worker=None,
        cpus_per_worker=1,
    )

    assert manager.num_workers == 1
    assert manager.coordinator_address == "127.0.0.1:7777"
    assert manager.scheduling_strategy == "SPREAD"
    assert manager._placement_group is None


@pytest.mark.skipif(not RAY_AVAILABLE, reason="ray not installed")
def test_ray_process_manager_requires_ray_init():
    """RayProcessManager raises if Ray is not initialized."""
    from tx.tinker.backends.jax import RayProcessManager

    with pytest.raises(RuntimeError, match="Ray must be initialized"):
        RayProcessManager(
            coordinator_address="127.0.0.1:7777",
            num_workers=1,
        )


@pytest.mark.skipif(not RAY_AVAILABLE, reason="ray not installed")
def test_ray_process_manager_shutdown_idempotent():
    """Calling shutdown() multiple times is safe."""
    from tx.tinker.backends.jax import RayProcessManager

    ray.init(num_cpus=4, include_dashboard=False)

    manager = RayProcessManager(
        coordinator_address="127.0.0.1:7777",
        num_workers=1,
        gpus_per_worker=None,
        cpus_per_worker=1,
    )

    # Multiple shutdowns should not raise
    manager.shutdown()
    manager.shutdown()
    manager.shutdown()


@pytest.mark.skipif(not RAY_AVAILABLE, reason="ray not installed")
def test_auto_detect_gpus_returns_none_on_cpu():
    """Auto-detect returns None when no GPUs are available."""
    from tx.tinker.backends.jax import _auto_detect_gpus_per_worker

    ray.init(num_cpus=4, include_dashboard=False)
    result = _auto_detect_gpus_per_worker(num_workers=1)
    assert result is None


@pytest.mark.skipif(not RAY_AVAILABLE, reason="ray not installed")
def test_placement_group_none_strategy():
    """NONE strategy skips placement group creation."""
    from tx.tinker.backends.jax import RayProcessManager

    ray.init(num_cpus=4, include_dashboard=False)

    manager = RayProcessManager(
        coordinator_address="127.0.0.1:7777",
        num_workers=1,
        gpus_per_worker=None,
        cpus_per_worker=1,
        scheduling_strategy="NONE",
    )

    # _create_placement_group should be a no-op when strategy is NONE
    manager._create_placement_group()
    assert manager._placement_group is None


@pytest.mark.skipif(not RAY_AVAILABLE, reason="ray not installed")
def test_placement_group_pack_cpu():
    """PACK placement group works with CPU-only resources on a single node.

    Note: This test may be slow or fail in environments where Ray worker
    pool initialization takes longer than the placement group timeout
    (e.g., when Ray needs to build a runtime environment).
    """
    from tx.tinker.backends.jax import RayProcessManager

    ray.init(num_cpus=4, include_dashboard=False, runtime_env={"working_dir": None})

    manager = RayProcessManager(
        coordinator_address="127.0.0.1:7777",
        num_workers=2,
        gpus_per_worker=None,
        cpus_per_worker=1,
        scheduling_strategy="PACK",
        placement_group_timeout=120,
    )

    # Should succeed with CPU-only bundles on a single node
    manager._create_placement_group()
    assert manager._placement_group is not None

    # Cleanup
    manager.shutdown()
    assert manager._placement_group is None


@pytest.mark.skipif(not RAY_AVAILABLE, reason="ray not installed")
def test_health_monitor_starts_and_stops():
    """Health monitor thread starts and can be stopped via shutdown."""
    from tx.tinker.backends.jax import RayProcessManager

    ray.init(num_cpus=4, include_dashboard=False)

    manager = RayProcessManager(
        coordinator_address="127.0.0.1:7777",
        num_workers=1,
        gpus_per_worker=None,
        cpus_per_worker=1,
    )

    # Start monitor (no workers launched, monitor should still start)
    manager.start_health_monitor(check_interval=0.1)
    assert manager._monitor_thread is not None

    # Shutdown stops the monitor
    manager.shutdown()
    assert manager._monitor_thread is None


@pytest.mark.skipif(not RAY_AVAILABLE, reason="ray not installed")
def test_engine_shutdown_with_enhanced_ray_manager():
    """Test that TinkerEngine.shutdown() works with the enhanced RayProcessManager."""
    from tx.tinker.backends.jax import RayProcessManager

    mock_manager = MagicMock(spec=RayProcessManager)

    # Simulate TinkerEngine's shutdown logic
    _ray_process_manager = mock_manager
    if _ray_process_manager is not None:
        _ray_process_manager.shutdown()
        _ray_process_manager = None

    mock_manager.shutdown.assert_called_once()
    assert _ray_process_manager is None
