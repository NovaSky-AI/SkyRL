import os
from functools import lru_cache

import pytest
import ray
from loguru import logger

from skyrl.env_vars import SKYRL_PYTHONPATH_EXPORT
from skyrl.train.utils.utils import peer_access_supported


@lru_cache(5)
def log_once(msg):
    logger.info(msg)
    return None


def _build_ray_env_vars():
    env_vars = {
        "VLLM_USE_V1": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "_SKYRL_USE_NEW_INFERENCE": os.environ.get("_SKYRL_USE_NEW_INFERENCE", "0"),
    }

    if not peer_access_supported(max_num_gpus_per_node=2):
        log_once("Disabling NCCL P2P for CI environment")
        env_vars.update(
            {
                "NCCL_P2P_DISABLE": "1",
                "NCCL_SHM_DISABLE": "1",
            }
        )

    # needed for megatron tests
    env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env_vars["NVTE_FUSED_ATTN"] = "0"

    if SKYRL_PYTHONPATH_EXPORT:
        pythonpath = os.environ.get("PYTHONPATH")
        if pythonpath is None:
            raise RuntimeError("SKYRL_PYTHONPATH_EXPORT is set but PYTHONPATH is not defined in environment")
        env_vars["PYTHONPATH"] = pythonpath

    return env_vars


def _ray_init():
    if ray.is_initialized():
        ray.shutdown()

    # TODO (team): maybe we should use the default config and use prepare_runtime_environment in some way
    env_vars = _build_ray_env_vars()

    logger.info(f"Initializing Ray with environment variables: {env_vars}")
    ray.init(runtime_env={"env_vars": env_vars})


@pytest.fixture
def ray_init_fixture():
    _ray_init()
    yield
    ray.shutdown()


@pytest.fixture(scope="class")
def class_scoped_ray_init_fixture():
    _ray_init()
    yield
    ray.shutdown()
