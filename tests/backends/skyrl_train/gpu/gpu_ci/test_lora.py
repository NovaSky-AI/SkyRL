"""
# Run FSDP tests:
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_lora.py -k "fsdp"

# Run Megatron tests:
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/gpu/gpu_ci/test_lora.py -k "megatron"
"""

import pytest
import ray

from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    get_test_prompts,
    init_worker_with_type,
    run_inference,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_actor_config(
    strategy: str = "fsdp",
    enable_lora: bool = False,
    colocate_all: bool = False,
    weight_sync_backend: str = "nccl",
    tp_size: int = 2,
    merge_lora: bool = True,
) -> SkyRLTrainConfig:
    """Get base config with test-specific overrides."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL
    cfg.trainer.critic.model.path = ""
    cfg.trainer.strategy = strategy
    cfg.trainer.placement.colocate_all = colocate_all
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.generator.inference_engine.async_engine = True
    cfg.generator.inference_engine.num_engines = 1
    cfg.generator.inference_engine.run_engines_locally = True
    cfg.generator.inference_engine.weight_sync_backend = weight_sync_backend
    cfg.generator.inference_engine.tensor_parallel_size = tp_size

    if strategy == "megatron":
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
        cfg.trainer.policy.megatron_config.lora_config.merge_lora = merge_lora

    if enable_lora:
        cfg.trainer.policy.model.lora = SkyRLLoraConfig(
            rank=32,
            alpha=32,
            dropout=0.1,
            target_modules="all-linear",
        )

    return cfg


@pytest.mark.parametrize(
    ("colocate_all", "weight_sync_backend", "strategy", "tp_size", "merge_lora"),
    [
        pytest.param(False, "nccl", "fsdp", 2, True),
        pytest.param(True, "nccl", "fsdp", 2, True),
        pytest.param(False, "nccl", "fsdp2", 2, True),
        pytest.param(True, "nccl", "fsdp2", 2, True),
        pytest.param(False, "nccl", "megatron", 2, True, marks=pytest.mark.megatron),
        pytest.param(True, "nccl", "megatron", 2, True, marks=pytest.mark.megatron),
        pytest.param(False, "nccl", "megatron", 2, False, marks=pytest.mark.megatron),
        pytest.param(True, "nccl", "megatron", 2, False, marks=pytest.mark.megatron),
    ],
    ids=[
        "no_colocate_nccl_fsdp",
        "colocate_nccl_fsdp",
        "no_colocate_nccl_fsdp2",
        "colocate_nccl_fsdp2",
        "no_colocate_nccl_megatron_merged",
        "colocate_nccl_megatron_merged",
        "no_colocate_nccl_megatron_adapter",
        "colocate_nccl_megatron_adapter",
    ],
)
@pytest.mark.asyncio
async def test_policy_local_engines_e2e(
    ray_init_fixture, colocate_all, weight_sync_backend, strategy, tp_size, merge_lora
):
    """
    Tests initalizing the policy actor group and inference engine, syncing weights, and performing generation.
    """
    cfg = get_test_actor_config(
        strategy=strategy,
        enable_lora=True,
        colocate_all=colocate_all,
        weight_sync_backend=weight_sync_backend,
        tp_size=tp_size,
        merge_lora=merge_lora,
    )

    # Only enable LoRA on the vLLM side when adapters are loaded separately.
    # When merge_lora=True the bridge merges LoRA into the full weights, so
    # vLLM receives plain weights and must NOT have enable_lora (which wraps
    # modules and changes named_parameters(), breaking load_weights).
    needs_vllm_lora = not (strategy == "megatron" and merge_lora)

    # If colocate is True, this will load the engine, sleep, and wake up the engine
    async with InferenceEngineState.create(
        cfg=cfg,
        model=MODEL,
        use_local=True,
        async_engine=cfg.generator.inference_engine.async_engine,
        tp_size=cfg.generator.inference_engine.tensor_parallel_size,
        colocate_all=cfg.trainer.placement.colocate_all,
        sleep_level=1 if needs_vllm_lora else 2,
        enable_lora=needs_vllm_lora,
    ) as engines:
        client, pg = engines.client, engines.pg

        await client.sleep(level=1)

        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=cfg.trainer.placement.colocate_all,
            num_gpus_per_node=cfg.generator.inference_engine.tensor_parallel_size,
            cfg=cfg,
        )
        sampling_params = get_sampling_params_for_backend(
            cfg.generator.inference_engine.backend, cfg.generator.sampling_params
        )
        await client.wake_up(tags=["weights"])

        ray.get(
            policy.async_run_ray_method(
                "pass_through", "init_weight_sync_state", client, cfg.generator.inference_engine
            )
        )
        ray.get(
            policy.async_run_ray_method(
                "pass_through", "broadcast_to_inference_engines", client, cfg.generator.inference_engine
            )
        )
        policy.offload_to_cpu()
        await client.wake_up(tags=["kv_cache"])
        await client.reset_prefix_cache()
        outputs = await run_inference(client, get_test_prompts(MODEL), sampling_params)
        print(f"Example output: {outputs['responses'][0]}, {outputs['stop_reasons'][0]}")
