import contextlib
import os
from functools import lru_cache

import pytest
import ray
from loguru import logger

from skyrl.env_vars import (
    _SKYRL_USE_NEW_INFERENCE,
    SKYRL_LD_LIBRARY_PATH_EXPORT,
    SKYRL_PYTHONPATH_EXPORT,
)
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
        "_SKYRL_USE_NEW_INFERENCE": "1" if _SKYRL_USE_NEW_INFERENCE else "0",
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

    # Mirror prepare_runtime_environment: for multi-node tests over EFA, the
    # driver's LD_LIBRARY_PATH (e.g. /opt/amazon/efa/lib) must reach the Ray
    # workers so NCCL picks up the EFA provider. Set SKYRL_LD_LIBRARY_PATH_EXPORT=1.
    if SKYRL_LD_LIBRARY_PATH_EXPORT:
        ld_library_path = os.environ.get("LD_LIBRARY_PATH")
        if ld_library_path is None:
            raise RuntimeError("SKYRL_LD_LIBRARY_PATH_EXPORT is set but LD_LIBRARY_PATH is not defined in environment")
        env_vars["LD_LIBRARY_PATH"] = ld_library_path

    # Forward debugging/observability env vars to the Ray workers when set on the
    # driver. Useful for multi-node bring-up: NCCL_DEBUG / FI_* surface
    # NCCL+EFA init failures, and SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1 stops the
    # inference-server actors from redirecting their stdout to per-node log
    # files (so crash tracebacks reach the driver). vLLM additionally copies
    # NCCL_*/FI_* prefixed vars from the engine to its TP/PP workers.
    # HF_* / HUGGING_FACE_* let the workers find a pre-staged model cache
    # (e.g. HF_HOME on a large node-local disk for very big models) and use an
    # HF token; vLLM also copies HF_-prefixed vars to its TP/PP workers.
    # SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S is read at import time inside
    # the VLLMServerActor worker process, so it must be forwarded to take effect. Very
    # large models (e.g. 550B) loaded across multi-node PP engines can take well over
    # the 600s default to become healthy.
    _DEBUG_PASSTHROUGH = (
        "SKYRL_DUMP_INFRA_LOG_TO_STDOUT",
        "PYTORCH_CUDA_ALLOC_CONF",
        "SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S",
    )
    _DEBUG_PREFIXES = ("NCCL_", "FI_", "HF_", "HUGGING_FACE_", "VLLM_")
    for name, value in os.environ.items():
        if name in _DEBUG_PASSTHROUGH or name.startswith(_DEBUG_PREFIXES):
            env_vars.setdefault(name, value)

    return env_vars


def _ray_init(extra_env_vars: dict[str, str] | None = None):
    if ray.is_initialized():
        ray.shutdown()

    # TODO (team): maybe we should use the default config and use prepare_runtime_environment in some way
    env_vars = _build_ray_env_vars()
    if extra_env_vars:
        env_vars.update(extra_env_vars)

    logger.info(f"Initializing Ray with environment variables: {env_vars}")
    ray.init(runtime_env={"env_vars": env_vars})


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
