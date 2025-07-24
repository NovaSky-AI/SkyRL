import pytest
import ray
import os
from loguru import logger
from functools import lru_cache
import torch


@lru_cache(5)
def log_once(msg):
    logger.info(msg)
    return None

def peer_access_supported():
    if not torch.cuda.is_available():
        return False
    
    device_count = torch.cuda.device_count()
    if device_count < 2:
        return False
    
    # Check P2P access between all GPU pairs
    for i in range(device_count):
        for j in range(device_count):
            if i != j:
                # This checks if device i can access device j's memory
                can_access = torch.cuda.can_device_access_peer(i, j)
                if not can_access:
                    return False
    
    return True

@pytest.fixture
def ray_init_fixture():
    if ray.is_initialized():
        ray.shutdown()
    env_vars = {}
    if peer_access_supported():
        log_once("Disabling NCCL P2P for CI environment")
        env_vars = {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}
    ray.init(runtime_env={"env_vars": env_vars})
    yield
    # call ray shutdown after a test regardless
    ray.shutdown()
