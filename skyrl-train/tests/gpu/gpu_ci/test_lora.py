"""
# Run vllm tests (requires vllm extra):
uv run --isolated --extra dev --extra vllm pytest tests/gpu/gpu_ci/test_lora.py
"""

import pytest
import asyncio
import ray
import hydra
from omegaconf import DictConfig

from tests.gpu.utils import init_worker_with_type, get_test_prompts, init_inference_engines, run_inference
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.entrypoints.main_base import config_dir

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_actor_config(enable_lora: bool = False) -> DictConfig:
    """Get base config with test-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

        # Override specific parameters
        cfg.trainer.policy.model.path = MODEL
        cfg.trainer.critic.model.path = ""
        cfg.trainer.placement.policy_num_gpus_per_node = 2
        cfg.generator.async_engine = True
        cfg.generator.num_inference_engines = 1
        cfg.generator.run_engines_locally = True

        # LoRA configuration
        if enable_lora:
            cfg.trainer.policy.model.lora.rank = 32
            cfg.trainer.policy.model.lora.alpha = 32
            cfg.trainer.policy.model.lora.dropout = 0.1
            cfg.trainer.policy.model.lora.target_modules = "all-linear"

        return cfg


@pytest.mark.parametrize(
    ("colocate_all", "weight_sync_backend", "strategy", "backend", "tp_size"),
    [
        pytest.param(False, "nccl", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(True, "nccl", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(False, "nccl", "fsdp2", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(True, "nccl", "fsdp2", "vllm", 2, marks=pytest.mark.vllm),
    ],
    ids=[
        "no_colocate_nccl_fsdp_vllm",
        "colocate_nccl_fsdp_vllm",
        "no_colocate_nccl_fsdp2_vllm",
        "colocate_nccl_fsdp2_vllm",
    ],
)
def test_policy_local_engines_e2e(ray_init_fixture, colocate_all, weight_sync_backend, strategy, backend, tp_size):
    """
    Tests initalizing the policy actor group and inference engine, syncing weights, and performing generation.
    """
    cfg = get_test_actor_config(enable_lora=True)
    cfg.trainer.placement.colocate_all = colocate_all
    cfg.generator.weight_sync_backend = weight_sync_backend
    cfg.trainer.strategy = strategy
    cfg.generator.backend = backend
    cfg.generator.inference_engine_tensor_parallel_size = tp_size

    # If colocate is True, this will load the engine, sleep, and wake up the engine
    client, pg = init_inference_engines(
        model=MODEL,
        cfg=cfg,
        use_local=True,
        async_engine=cfg.generator.async_engine,
        tp_size=cfg.generator.inference_engine_tensor_parallel_size,
        colocate_all=cfg.trainer.placement.colocate_all,
        backend=backend,
        sleep_level=1,  # since we explicitly sync weights
        enable_lora=True,  # Enable LoRA for this test
    )

    policy = init_worker_with_type(
        "policy",
        shared_pg=pg,
        colocate_all=cfg.trainer.placement.colocate_all,
        num_gpus_per_node=cfg.generator.inference_engine_tensor_parallel_size,
        cfg=cfg,
    )
    sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
    ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
    asyncio.run(client.reset_prefix_cache())
    ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
    outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL), sampling_params))
    print(f"Example output: {outputs['responses'][0]}, {outputs['stop_reasons'][0]}")
