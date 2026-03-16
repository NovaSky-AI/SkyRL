"""
To run:
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_policy_local_engines_e2e.py
"""

import asyncio

import pytest
import ray
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.env_vars import _SKYRL_USE_NEW_INFERENCE
from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    get_test_prompts,
    init_worker_with_type,
    run_inference,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_actor_config(model: str) -> SkyRLTrainConfig:
    """Get base config with test-specific overrides."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model
    cfg.trainer.critic.model.path = ""
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.generator.inference_engine.async_engine = True
    cfg.generator.inference_engine.num_engines = 1
    cfg.generator.inference_engine.run_engines_locally = True
    # NOTE: We reduce the gpu memory used by vLLM because of the colocated tests
    # that can OOM on L4s. For more details, see: https://github.com/NovaSky-AI/SkyRL/pull/1221
    cfg.generator.inference_engine.gpu_memory_utilization = 0.7
    return cfg


# TODO (aaron): add back tests when we support gloo
_skip_new_inference = pytest.mark.skipif(_SKYRL_USE_NEW_INFERENCE, reason="Not yet supported on new inference path")


@pytest.mark.parametrize(
    (
        "colocate_all",
        "weight_sync_backend",
        "strategy",
        "num_engines",
        "tp_size",
        "distributed_executor_backend",
    ),
    [
        pytest.param(False, "nccl", "fsdp", 1, 2, "ray"),
        pytest.param(True, "nccl", "fsdp", 1, 2, "ray"),
        pytest.param(False, "gloo", "fsdp", 1, 2, "ray", marks=_skip_new_inference),
        pytest.param(True, "gloo", "fsdp", 1, 2, "ray", marks=_skip_new_inference),
        pytest.param(False, "nccl", "fsdp2", 1, 2, "ray"),
        pytest.param(True, "nccl", "fsdp2", 2, 2, "ray"),
        pytest.param(True, "nccl", "fsdp2", 2, 2, "mp"),
        pytest.param(False, "nccl", "fsdp2", 1, 2, "mp"),
    ],
    ids=[
        "no_colocate_nccl_fsdp_vllm",
        "colocate_nccl_fsdp_vllm",
        "no_colocate_gloo_fsdp_vllm",
        "colocate_gloo_fsdp_vllm",
        "no_colocate_nccl_fsdp2_vllm",
        "colocate_nccl_fsdp2_vllm",
        "colocate_nccl_fsdp2_vllm_mp",
        "non_colocated_nccl_fsdp2_vllm_mp",
    ],
)
def test_policy_local_engines_e2e(
    ray_init_fixture,
    colocate_all,
    weight_sync_backend,
    strategy,
    num_engines,
    tp_size,
    distributed_executor_backend,
):
    """
    Tests initalizing the policy actor group and inference engine, syncing weights, and performing generation.
    """
    cfg = get_test_actor_config(MODEL)
    cfg.trainer.placement.colocate_all = colocate_all
    cfg.generator.inference_engine.weight_sync_backend = weight_sync_backend
    cfg.trainer.strategy = strategy
    cfg.generator.inference_engine.tensor_parallel_size = tp_size
    cfg.generator.inference_engine.distributed_executor_backend = distributed_executor_backend
    cfg.generator.inference_engine.num_engines = num_engines
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # If colocate is True, this will load the engine, sleep, and wake up the engine
    with InferenceEngineState.create(
        model=MODEL,
        cfg=cfg,
        use_local=True,
        async_engine=cfg.generator.inference_engine.async_engine,
        tp_size=cfg.generator.inference_engine.tensor_parallel_size,
        colocate_all=cfg.trainer.placement.colocate_all,
        sleep_level=2,  # since we explicitly sync weights
    ) as engines:
        client, pg = engines.client, engines.pg
        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=cfg.trainer.placement.colocate_all,
            num_gpus_per_node=cfg.generator.inference_engine.tensor_parallel_size
            * cfg.generator.inference_engine.num_engines
            * cfg.generator.inference_engine.data_parallel_size,
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
