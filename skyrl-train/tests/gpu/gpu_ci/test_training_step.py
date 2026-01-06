"""
Run with:
uv run --isolated --extra dev -- pytest tests/gpu/gpu_ci/test_training_step.py
"""

import ray
import pytest
import hydra
from omegaconf import DictConfig

from tests.gpu.utils import init_worker_with_type, make_dummy_experience, validate_cfg
from skyrl_train.utils.utils import print_mem
from skyrl_train.entrypoints.main_base import config_dir


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_actor_config() -> DictConfig:
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.logger = "console"
    cfg.generator.inference_engine_tensor_parallel_size = 2

    return cfg


@pytest.fixture
def cfg() -> DictConfig:
    return get_test_actor_config()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("packed", "strategy"),
    [(True, "fsdp"), (False, "fsdp"), (True, "fsdp2"), (False, "fsdp2")],
    ids=[
        "packed-fsdp",
        "unpacked-fsdp",
        "packed-fsdp2",
        "unpacked-fsdp2",
    ],
)
async def test_policy_forward_backward_and_optim_step(ray_init_fixture, cfg, packed, strategy):
    """
    Full test: initialize actor group, send dummy experience to forward_backward + optim_step, validate output.
    """
    cfg.trainer.use_sample_packing = packed
    cfg.trainer.strategy = strategy
    validate_cfg(cfg)

    try:
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        dummy_experience = make_dummy_experience()

        results = ray.get(actor_group.async_run_ray_method("pass_through", "forward_backward", dummy_experience, 1))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))

        memory = ray.get(actor_group.async_run_ray_method("pass_through", "get_cuda_memory"))
        memory = memory[0]
        print_mem("memory after forward_backward + optim_step", memory)

        for result in results:
            assert isinstance(result, dict), "Result should be a dictionary of training stats"
            assert "policy_loss" in result
            assert "ppo_clip_ratio" in result
            assert "policy_entropy" in result
            for k, v in result.items():
                assert isinstance(v, (int, float)), f"{k} should be an int or float"

    finally:
        ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("packed", "strategy"),
    [(True, "fsdp"), (False, "fsdp"), (True, "fsdp2"), (False, "fsdp2")],
    ids=[
        "packed-fsdp",
        "unpacked-fsdp",
        "packed-fsdp2",
        "unpacked-fsdp2",
    ],
)
async def test_critic_forward_backward_and_optim_step(ray_init_fixture, cfg, packed, strategy):
    """
    Full test: initialize critic actor group, send dummy experience to forward_backward + optim_step, validate output.
    """
    cfg.trainer.use_sample_packing = packed
    cfg.trainer.strategy = strategy
    validate_cfg(cfg)
    try:
        actor_group = init_worker_with_type(
            "critic",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        dummy_experience = make_dummy_experience()

        results = ray.get(actor_group.async_run_ray_method("pass_through", "forward_backward", dummy_experience, 1))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))

        for result in results:
            assert isinstance(result, dict), "Result should be a dictionary of training stats"
            assert "critic_loss" in result
            assert "values_mean" in result
            for k, v in result.items():
                assert isinstance(v, float), f"{k} should be a float"

    finally:
        ray.shutdown()
