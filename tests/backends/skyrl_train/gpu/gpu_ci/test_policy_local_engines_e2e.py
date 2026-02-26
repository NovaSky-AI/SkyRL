"""
# Run only vllm tests (requires vllm extra):
uv run --isolated --extra dev --extra vllm pytest tests/gpu/gpu_ci/test_policy_local_engines_e2e.py -m "vllm"

# Run only sglang tests (requires sglang extra):
uv run --isolated --extra dev --extra sglang pytest tests/gpu/gpu_ci/test_policy_local_engines_e2e.py -m "sglang"
"""

import pytest
import asyncio
import ray
from transformers import AutoTokenizer

from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    get_test_prompts,
    InferenceEngineState,
    run_inference,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_actor_config() -> SkyRLTrainConfig:
    """Get base config with test-specific overrides."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL
    cfg.trainer.critic.model.path = ""
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.generator.inference_engine.async_engine = True
    cfg.generator.inference_engine.num_engines = 1
    cfg.generator.inference_engine.run_engines_locally = True
    # NOTE: We reduce the gpu memory used by vLLM because of the colocated tests
    # that can OOM on L4s. For more details, see: https://github.com/NovaSky-AI/SkyRL/pull/1221
    cfg.generator.inference_engine.gpu_memory_utilization = 0.7
    return cfg


@pytest.mark.parametrize(
    ("colocate_all", "weight_sync_backend", "strategy", "backend", "tp_size"),
    [
        pytest.param(False, "nccl", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(True, "nccl", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(False, "gloo", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(True, "gloo", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(False, "nccl", "fsdp2", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(True, "nccl", "fsdp2", "vllm", 2, marks=pytest.mark.vllm),
        # TODO(Charlie): add TP > 1 tests for sglang when we support it
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
        "no_colocate_nccl_fsdp2_vllm",
        "colocate_nccl_fsdp2_vllm",
        "no_colocate_nccl_fsdp2_sglang",
        "colocate_nccl_fsdp2_sglang",
        "no_colocate_gloo_fsdp_sglang",
        "colocate_gloo_fsdp_sglang",
    ],
)
def test_policy_local_engines_e2e(ray_init_fixture, colocate_all, weight_sync_backend, strategy, backend, tp_size):
    """
    Tests initalizing the policy actor group and inference engine, syncing weights, and performing generation.
    """
    cfg = get_test_actor_config()
    cfg.trainer.placement.colocate_all = colocate_all
    cfg.generator.inference_engine.weight_sync_backend = weight_sync_backend
    cfg.trainer.strategy = strategy
    cfg.generator.inference_engine.backend = backend
    cfg.generator.inference_engine.tensor_parallel_size = tp_size

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # If colocate is True, this will load the engine, sleep, and wake up the engine
    with InferenceEngineState.create(
        model=MODEL,
        cfg=cfg,
        use_local=True,
        async_engine=cfg.generator.inference_engine.async_engine,
        tp_size=cfg.generator.inference_engine.tensor_parallel_size,
        colocate_all=cfg.trainer.placement.colocate_all,
        backend=backend,
        sleep_level=2,  # since we explicitly sync weights
    ) as engines:
        client, pg = engines.client, engines.pg
        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=cfg.trainer.placement.colocate_all,
            num_gpus_per_node=cfg.generator.inference_engine.tensor_parallel_size,
            cfg=cfg,
        )

        ray.get(
            policy.async_run_ray_method(
                "pass_through", "init_weight_sync_state", client, cfg.generator.inference_engine
            )
        )
        asyncio.run(client.reset_prefix_cache())
        ray.get(
            policy.async_run_ray_method(
                "pass_through", "broadcast_to_inference_engines", client, cfg.generator.inference_engine
            )
        )

        sampling_params = get_sampling_params_for_backend(
            cfg.generator.inference_engine.backend, cfg.generator.sampling_params
        )
        outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL), sampling_params, tokenizer=tokenizer))

        print(f"Example output after weight sync: {outputs['responses'][0]}, {outputs['stop_reasons'][0]}")
