import pytest
import ray
from loguru import logger
from functools import lru_cache
from skyrl_train.utils.utils import get_env_vars_for_p2p_access


@lru_cache(5)
def log_once(msg):
    logger.info(msg)
    return None


@pytest.fixture
def ray_init_fixture():
    if ray.is_initialized():
        ray.shutdown()
    env_vars = {}
    overrides = get_env_vars_for_p2p_access(max_num_gpus_per_node=2)
    if overrides != {}:
        log_once(f"Disabling NCCL P2P for test environment, setting the following env vars: {overrides}")
        env_vars = overrides
    ray.init(runtime_env={"env_vars": env_vars})
    yield
    # call ray shutdown after a test regardless
    ray.shutdown()
