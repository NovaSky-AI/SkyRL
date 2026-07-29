import contextlib
import os
import sysconfig
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


def _pip_cudnn_env_vars():
    """Pin cuDNN to the `nvidia-cudnn-cu12` wheel that TE and torch were built against.

    `transformer_engine` prefers a system CUDA toolkit over pip wheels when it
    dlopens cuDNN: `_load_cuda_library_from_system` checks `$CUDNN_PATH`, then
    `$CUDA_HOME`, then `/usr/local/cuda`, and only falls back to site-packages if
    none of those have a `libcudnn.so*`. On a host with a system toolkit shipping
    an older cuDNN, TE therefore runs against that one instead of the wheel it
    was compiled against, and MLA-shaped fused attention (head_dim_qk=192 !=
    head_dim_v=128) finds no cuDNN engine at all on sm100.

    `CUDNN_PATH` redirects TE's dlopen; `LD_LIBRARY_PATH` is also required
    because `libcudnn.so.9` is only a shim that loads its sublibraries
    (`libcudnn_graph.so.9`, `libcudnn_engines_*.so.9`) through the normal loader
    search path -- those are what `cudnnGetVersion()` actually reports.
    """
    cudnn_lib = os.path.join(sysconfig.get_path("purelib"), "nvidia", "cudnn", "lib")
    if not os.path.isdir(cudnn_lib):
        return {}

    existing = os.environ.get("LD_LIBRARY_PATH")
    return {
        "CUDNN_PATH": os.path.dirname(cudnn_lib),
        "LD_LIBRARY_PATH": f"{cudnn_lib}:{existing}" if existing else cudnn_lib,
    }


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

    env_vars.update(_pip_cudnn_env_vars())

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
