"""
# Run only vllm tests (requires vllm extra):
uv run --isolated --extra dev --extra vllm --extra deepspeed pytest tests/gpu/test_policy_local_engines_e2e.py -m "vllm"

# Run only sglang tests (requires sglang extra):
uv run --isolated --extra dev --extra sglang --extra deepspeed pytest tests/gpu/test_policy_local_engines_e2e.py -m "sglang"
"""

import pytest
import asyncio
import ray
import hydra
from omegaconf import DictConfig

from tests.gpu.utils import init_worker_with_type, get_test_prompts, init_inference_engines
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.utils.ppo_utils import PolicyLossRegistry, AdvantageEstimatorRegistry

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_actor_config() -> DictConfig:
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

        return cfg


async def run_inference(client, prompts, sampling_params):
    engine_input = InferenceEngineInput(prompts=prompts, sampling_params=sampling_params)
    return await client.generate(engine_input)


@pytest.mark.parametrize(
    ("colocate_all", "weight_sync_backend", "strategy", "backend", "tp_size"),
    [
        pytest.param(False, "nccl", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(True, "nccl", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(False, "gloo", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(True, "gloo", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(False, "nccl", "deepspeed", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(True, "nccl", "deepspeed", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(False, "nccl", "fsdp2", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(True, "nccl", "fsdp2", "vllm", 2, marks=pytest.mark.vllm),
        # TODO(Charlie): add TP > 1 tests for sglang when we support it
        pytest.param(False, "nccl", "deepspeed", "sglang", 1, marks=pytest.mark.sglang),
        pytest.param(True, "nccl", "deepspeed", "sglang", 1, marks=pytest.mark.sglang),
        pytest.param(False, "nccl", "fsdp2", "sglang", 1, marks=pytest.mark.sglang),
        pytest.param(True, "nccl", "fsdp2", "sglang", 1, marks=pytest.mark.sglang),
        pytest.param(False, "gloo", "fsdp", "sglang", 1, marks=pytest.mark.sglang),
        pytest.param(True, "gloo", "fsdp", "sglang", 1, marks=pytest.mark.sglang),
    ],
    ids=[
        "no_colocate_nccl_fsdp_vllm",
        "colocate_nccl_fsdp_vllm",
        "no_colocate_gloo_fsdp_vllm",
        "colocate_gloo_fsdp_vllm",
        "no_colocate_nccl_deepspeed_vllm",
        "colocate_nccl_deepspeed_vllm",
        "no_colocate_nccl_fsdp2_vllm",
        "colocate_nccl_fsdp2_vllm",
        "no_colocate_nccl_deepspeed_sglang",
        "colocate_nccl_deepspeed_sglang",
        "no_colocate_nccl_fsdp2_sglang",
        "colocate_nccl_fsdp2_sglang",
        "no_colocate_gloo_fsdp_sglang",
        "colocate_gloo_fsdp_sglang",
    ],
)
def test_policy_local_engines_e2e(colocate_all, weight_sync_backend, strategy, backend, tp_size):
    """
    Tests initalizing the policy actor group and inference engine, syncing weights, and performing generation.
    """
    try:
        cfg = get_test_actor_config()
        cfg.trainer.placement.colocate_all = colocate_all
        cfg.generator.weight_sync_backend = weight_sync_backend
        cfg.trainer.strategy = strategy
        cfg.generator.backend = backend
        cfg.generator.inference_engine_tensor_parallel_size = tp_size

        # If colocate is True, this will load the engine, sleep, and wake up the engine
        client, pg = init_inference_engines(
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
            backend=backend,
            model=MODEL,
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
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL), sampling_params))

        print(f"Example output: {outputs['responses'][0]}, {outputs['stop_reasons'][0]}")
    finally:
        AdvantageEstimatorRegistry.reset()
        PolicyLossRegistry.reset()
        ray.shutdown()
