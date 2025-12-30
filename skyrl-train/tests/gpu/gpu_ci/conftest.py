import os
import pytest
import ray
from loguru import logger
from functools import lru_cache
from skyrl_train.utils.utils import peer_access_supported


@lru_cache(5)
def log_once(msg):
    logger.info(msg)
    return None


@pytest.fixture
def ray_init_fixture():
    if ray.is_initialized():
        ray.shutdown()

    # TODO (team): maybe we should use the default config and use prepare_runtime_environment in some way
    env_vars = {"VLLM_USE_V1": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}

    # Pass through NCCL-related environment variables to Ray workers
    # LD_PRELOAD is critical to ensure all processes (PyTorch + vLLM) use the same NCCL version
    for env_key in ["VLLM_NCCL_SO_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS"]:
        if env_key in os.environ:
            env_vars[env_key] = os.environ[env_key]

    if not peer_access_supported(max_num_gpus_per_node=2):
        log_once("Disabling NCCL P2P for CI environment")
        env_vars.update(
            {
                "NCCL_P2P_DISABLE": "1",
                "NCCL_SHM_DISABLE": "1",
            }
        )

    logger.info(f"Initializing Ray with environment variables: {env_vars}")
    ray.init(runtime_env={"env_vars": env_vars})

    yield
    # call ray shutdown after a test regardless
    ray.shutdown()
