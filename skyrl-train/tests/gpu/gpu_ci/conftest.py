import pytest
import ray
from loguru import logger
from functools import lru_cache
from skyrl_train.utils.utils import peer_access_supported
import torch
import torch.distributed as dist


@lru_cache(5)
def log_once(msg):
    logger.info(msg)
    return None


@pytest.fixture()
def ray_init_fixture():
    if ray.is_initialized():
        ray.shutdown()

    env_vars = {}
    if not peer_access_supported(max_num_gpus_per_node=2):
        log_once("Disabling NCCL P2P for CI environment")
        env_vars = {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}

    ray.init(runtime_env={"env_vars": env_vars}, ignore_reinit_error=True)
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
