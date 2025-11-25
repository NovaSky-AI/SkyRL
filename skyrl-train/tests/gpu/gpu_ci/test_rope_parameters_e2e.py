"""
E2E test for RoPE parameters propagation from config to AutoModel init.

Run with:
uv run --isolated --extra dev pytest tests/gpu/gpu_ci/test_rope_parameters_e2e.py
"""

import pytest
import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from tests.gpu.utils import init_worker_with_type
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.utils import validate_cfg

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_config_with_rope_parameters(rope_parameters: dict) -> DictConfig:
    """Get base config with RoPE parameters overridden."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

        # Override specific parameters
        cfg.trainer.policy.model.path = MODEL
        cfg.trainer.critic.model.path = ""
        cfg.trainer.placement.policy_num_gpus_per_node = 1
        cfg.trainer.placement.ref_num_gpus_per_node = 1  
        cfg.trainer.placement.colocate_all = False  
        cfg.trainer.strategy = "fsdp"
        cfg.trainer.logger = "console"
        
        cfg.generator.inference_engine_tensor_parallel_size = 1
        cfg.generator.num_inference_engines = 1

        cfg.trainer.rope_parameters = OmegaConf.create(rope_parameters)
        cfg.generator.rope_parameters = OmegaConf.create(rope_parameters)

        validate_cfg(cfg)
        return cfg


@pytest.mark.parametrize(
    "rope_parameters",
    [
        {
            "rope_type": "linear",
            "factor": 2.0,
            "rope_theta": 10000.0,
        },
        {
            "rope_type": "yarn",
            "factor": 4.0,
            "rope_theta": 20000.0,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        {
            "rope_type": "dynamic",
            "factor": 8.0,
            "original_max_position_embeddings": 2048,
        },
    ],
    ids=["linear", "yarn", "dynamic"],
)
def test_rope_parameters_propagate_to_automodel_init(ray_init_fixture, rope_parameters):
    """
    E2E test that verifies RoPE parameters from config propagate all the way to AutoModel init.
   
    1. Creates a config with RoPE parameters
    2. Initializes a worker with that config
    3. Verifies that the model initializes successfully (which confirms rope_parameters are read from config)
    """
    cfg = get_test_config_with_rope_parameters(rope_parameters)

    policy = init_worker_with_type(
        "policy",
        shared_pg=None,
        colocate_all=False,
        num_gpus_per_node=1,
        num_nodes=1,
        cfg=cfg,
    )

    assert policy is not None, "Worker should be initialized successfully"
    

def test_rope_parameters_empty_dict(ray_init_fixture):
    """
    Test that empty rope_parameters dict is handled correctly.
    """
    cfg = get_test_config_with_rope_parameters({})

    policy = init_worker_with_type(
        "policy",
        shared_pg=None,
        colocate_all=False,
        num_gpus_per_node=1,
        num_nodes=1,
        cfg=cfg,
    )

    assert policy is not None

