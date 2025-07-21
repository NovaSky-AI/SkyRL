import pytest
import ray
import os
from loguru import logger
from functools import lru_cache


@lru_cache(5)
def log_once(msg):
    logger.info(msg)
    return None


@pytest.fixture
def ray_init_fixture():
    if ray.is_initialized():
        ray.shutdown()
    # NOTE (sumanthrh): We disable SHM for CI environment by default - L4s don't support P2P access
    # if `CI=false`, then this will be overriden.
    env_vars = {}
    val = os.environ.get("CI", "").lower()
    if val in ("1", "true", "yes"):
        log_once("Disabling NCCL P2P for CI environment")
        env_vars = {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}
    ray.init(runtime_env={"env_vars": env_vars})
    yield
    # call ray shutdown after a test regardless
    ray.shutdown()
