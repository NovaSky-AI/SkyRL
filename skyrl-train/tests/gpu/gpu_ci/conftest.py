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


@pytest.fixture(scope="function")
def ray_init_fixture():
    """Per-test Ray initialization with proper cleanup"""
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc

        gc.collect()
        torch.cuda.empty_cache()
