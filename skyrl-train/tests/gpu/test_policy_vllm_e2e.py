"""
uv run --isolated --extra dev --extra vllm pytest tests/gpu/test_policy_vllm_e2e.py
"""

import pytest
import asyncio
import ray
import hydra
from omegaconf import DictConfig

from tests.gpu.utils import init_worker_with_type, get_test_prompts
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from transformers import AutoTokenizer
from ray.util.placement_group import placement_group
from skyrl_train.utils import get_ray_pg_ready_with_timeout
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.entrypoints.main_base import config_dir

model = "Qwen/Qwen2.5-1.5B-Instruct"
tp_size = 2


def get_test_actor_config() -> DictConfig:
    """Get base config with test-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

        # Override specific parameters
        cfg.trainer.policy.model.path = model
        cfg.trainer.critic.model.path = ""
        cfg.trainer.placement.policy_num_gpus_per_node = 2
        cfg.generator.async_engine = True
        cfg.generator.num_inference_engines = 1
        cfg.generator.inference_engine_tensor_parallel_size = tp_size
        cfg.generator.run_engines_locally = True

        return cfg


async def run_vllm_inference(client, prompts):
    engine_input = InferenceEngineInput(prompts=prompts)
    await client.generate(engine_input)


def init_inference_engines(cfg, v1, use_local, async_engine, tp_size, colocate_all):
    assert use_local, "This test does not yet support remote engines."
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "NCCL_CUMEM_ENABLE": "0",
                "NCCL_P2P_DISABLE": "0",
                "CUDA_LAUNCH_BLOCKING": "1",
                "VLLM_USE_V1": "1" if v1 else "0",
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            }
        },
    )
    if colocate_all:
        pg = placement_group([{"GPU": 1, "CPU": 1}] * tp_size, strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=30)
        sleep = True
    else:
        pg, sleep = None, False

    eps = create_ray_wrapped_inference_engines(
        num_inference_engines=1,
        tensor_parallel_size=tp_size,
        model_dtype="bfloat16",
        pretrain=model,
        seed=42,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        max_model_len=1536,
        shared_pg=pg,
        gpu_memory_utilization=0.8,
        vllm_enable_sleep=sleep,
        async_engine=async_engine,
        max_num_batched_tokens=8192,
        max_num_seqs=1024,
        sampling_params=get_sampling_params_for_backend("vllm", cfg.generator.sampling_params),
        tokenizer=AutoTokenizer.from_pretrained(model),
    )
    client = InferenceEngineClient(eps)
    if sleep:
        asyncio.run(client.wake_up())
    return client, pg


@pytest.mark.parametrize(
    ("colocate_all", "weight_sync_backend", "strategy"),
    [
        (False, "nccl", "fsdp"),
        (True, "nccl", "fsdp"),
        (False, "gloo", "fsdp"),
        (True, "gloo", "fsdp"),
        (False, "nccl", "deepspeed"),
        (True, "nccl", "deepspeed"),
        (False, "nccl", "fsdp2"),
        (True, "nccl", "fsdp2"),
    ],
    ids=[
        "no_colocate_nccl_fsdp",
        "colocate_nccl_fsdp",
        "no_colocate_gloo_fsdp",
        "colocate_gloo_fsdp",
        "no_colocate_nccl_deepspeed",
        "colocate_nccl_deepspeed",
        "no_colocate_nccl_fsdp2",
        "colocate_nccl_fsdp2",
    ],
)
def test_policy_vllm_e2e(colocate_all, weight_sync_backend, strategy):
    """
    Tests initalizing the policy actor group and inference engine, syncing weights, and performing generation.
    """
    try:
        cfg = get_test_actor_config()
        cfg.trainer.placement.colocate_all = colocate_all
        cfg.generator.weight_sync_backend = weight_sync_backend
        cfg.trainer.strategy = strategy

        client, pg = init_inference_engines(
            cfg=cfg,
            v1=True,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
        )

        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=cfg.trainer.placement.colocate_all,
            num_gpus_per_node=cfg.generator.inference_engine_tensor_parallel_size,
            cfg=cfg,
        )
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        asyncio.run(client.reset_prefix_cache())
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        asyncio.run(run_vllm_inference(client, get_test_prompts(model)))
    finally:
        ray.shutdown()
