import pytest
import ray
import torch
import torch.distributed as dist
from loguru import logger
from functools import lru_cache
from skyrl_train.utils.utils import peer_access_supported


@lru_cache(5)
def log_once(msg):
    logger.info(msg)
    return None


@pytest.fixture()
def ray_init_fixture():
    """Session-scoped Ray initialization with proper cleanup"""
    if ray.is_initialized():
        ray.shutdown()

    env_vars = {}
    if not peer_access_supported(max_num_gpus_per_node=2):
        log_once("Disabling NCCL P2P for CI environment")
        env_vars = {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}

    ray.init(runtime_env={"env_vars": env_vars})
    yield

    try:
        ray.kill(ray.get_actor("*", allow_unknown=True), no_restart=True)
    except Exception:
        pass

    ray.shutdown()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def gpu_cleanup():
    """Automatic GPU cleanup after each test"""
    yield
    # Clean up distributed process groups after each test
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        # Reset CUDA context to clear any lingering state
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def test_isolation():
    """Ensure complete test isolation by clearing all distributed state"""
    # Set environment variables to disable NCCL P2P for all tests
    import os

    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_SHM_DISABLE"] = "1"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"

    yield

    # Additional cleanup after each test
    try:
        # Force cleanup of any remaining distributed state
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def pytest_configure(config):
    """Configure pytest to skip tests that might have peer access issues"""
    import os
    import torch

    # Check if we're in a CI environment with potential peer access issues
    if os.environ.get("CI") == "true" and torch.cuda.is_available():
        # Force disable peer access in CI
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_SHM_DISABLE"] = "1"
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
