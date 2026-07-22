import contextlib
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

    # Mirrors prepare_runtime_environment for the nccl weight-sync backend.
    # Without this, NCCL 2.28's cuMem-based commAlloc SEGV's on this driver.
    env_vars["NCCL_CUMEM_ENABLE"] = "0"

    if SKYRL_PYTHONPATH_EXPORT:
        pythonpath = os.environ.get("PYTHONPATH")
        if pythonpath is None:
            raise RuntimeError("SKYRL_PYTHONPATH_EXPORT is set but PYTHONPATH is not defined in environment")
        env_vars["PYTHONPATH"] = pythonpath

    return env_vars


def _ray_init(extra_env_vars: dict[str, str] | None = None):
    if ray.is_initialized():
        ray.shutdown()

    # TODO (team): maybe we should use the default config and use prepare_runtime_environment in some way
    env_vars = _build_ray_env_vars()
    if extra_env_vars:
        env_vars.update(extra_env_vars)

    runtime_env = {"env_vars": env_vars}
    # When running these GPU tests from a virtualenv against a pre-existing
    # (e.g. Anyscale-managed) Ray cluster whose base Python differs from the
    # driver's, the cluster's workers won't have the venv's packages (vllm,
    # skyrl, nixl). Setting SKYRL_RAY_PY_EXECUTABLE makes Ray launch this job's
    # workers with that interpreter so actors import the same env as the driver.
    # Opt-in only — default behavior (CI, fresh local cluster) is unchanged.
    py_executable = os.environ.get("SKYRL_RAY_PY_EXECUTABLE")
    if py_executable:
        runtime_env["py_executable"] = py_executable

    logger.info(f"Initializing Ray with runtime_env: {runtime_env}")
    ray.init(runtime_env=runtime_env)


@contextlib.contextmanager
def ray_init(extra_env_vars: dict[str, str] | None = None):
    _ray_init(extra_env_vars)
    try:
        yield
    finally:
        ray.shutdown()


@pytest.fixture
def ray_init_fixture():
    with ray_init():
        yield


@pytest.fixture(scope="class")
def class_scoped_ray_init_fixture():
    with ray_init():
        yield
