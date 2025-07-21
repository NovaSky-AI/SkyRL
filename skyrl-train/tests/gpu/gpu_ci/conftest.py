import pytest
import ray


@pytest.fixture
def ray_init_fixture():
    if ray.is_initialized():
        ray.shutdown()
    # Disable SHM for CI environment - L4s don't support P2P access
    ray.init(runtime_env={"env_vars": {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}})
    yield
    # call ray shutdown after a test regardless
    ray.shutdown()
