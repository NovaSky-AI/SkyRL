"""
To run:
uv run --isolated --extra dev --extra fsdp pytest -s -vvv tests/backends/skyrl_train/gpu/gpu_ci/test_policy_local_engines_e2e.py
"""

import pytest
import ray
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    get_test_prompts,
    init_worker_with_type,
    run_inference,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MOE_MODEL = "hf-internal-testing/tiny-qwen3-moe"
QWEN_LARGE_MOE_MODEL = "Qwen/Qwen3.5-35B-A3B"

# Opt-in marker for tests that require H100s. Auto-skipped unless `-m h100`
# is passed (see pytest_collection_modifyitems in the gpu conftest).
_h100_only = pytest.mark.h100


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "colocate_all",
        "weight_sync_backend",
        "strategy",
        "num_engines",
        "tp_size",
        "distributed_executor_backend",
        "model",
        "dp_size",
        "use_new_inference_servers",
    ),
    [
        pytest.param(False, "nccl", "fsdp", 1, 2, "ray", MODEL, 1, None),
        pytest.param(True, "nccl", "fsdp", 2, 2, "ray", MODEL, 1, None),
        pytest.param(True, "nccl", "fsdp", 2, 2, "mp", MODEL, 1, None),
        pytest.param(False, "nccl", "fsdp", 1, 2, "mp", MODEL, 1, None),
        # moe model, dp > 1
        pytest.param(True, "nccl", "fsdp", 2, 1, "ray", MOE_MODEL, 2, None),
        pytest.param(False, "nccl", "fsdp", 1, 1, "ray", MOE_MODEL, 2, None),
        # sharded_rdt (NIXL pull): non-colocated TP=1 — 1 GPU for the FSDP policy
        # (rank-0 named actor) + 1 GPU for vLLM. Requires the new inference server
        # path so the worker extension registers the engine and bakes the plan.
        pytest.param(False, "sharded_rdt", "fsdp", 1, 1, "ray", MODEL, 1, True),
        # sharded_rdt with 2 FSDP ranks + 2 vLLM engines — closer to the production
        # training run (4 ranks / 4 engines). With num_engines=2 the policy world
        # size becomes 2, so gather_layer triggers a real FSDP2 all-gather collective
        # rather than the trivial no-op that fires on a single rank.
        pytest.param(False, "sharded_rdt", "fsdp", 2, 1, "ray", MODEL, 1, True),
        # Qwen3.5-35B-A3B (~35B MoE, ~3B activated) on 4xH100-80G. "fsdp"
        # is fsdp2 in the current backend (FSDP1 was removed). Colocated
        # uses tp=4 across all 4 GPUs; non-colocated splits 2 GPUs for
        # vLLM (tp=2) and 2 for the FSDP policy.
        pytest.param(True, "nccl", "fsdp", 1, 4, "ray", QWEN_LARGE_MOE_MODEL, 1, None, marks=_h100_only),
        pytest.param(False, "nccl", "fsdp", 1, 2, "ray", QWEN_LARGE_MOE_MODEL, 1, None, marks=_h100_only),
    ],
    ids=[
        "no_colocate_nccl_fsdp_vllm",
        "colocate_nccl_fsdp_vllm",
        "colocate_nccl_fsdp_vllm_mp",
        "non_colocated_nccl_fsdp_vllm_mp",
        "colocate_nccl_fsdp_vllm_dp",
        "non_colocated_nccl_fsdp_vllm_dp",
        "no_colocate_sharded_rdt_fsdp_vllm",
        "no_colocate_sharded_rdt_fsdp_vllm_2fsdp_2vllm",
        "colocate_nccl_fsdp_vllm_qwen3_5_35b_a3b_h100",
        "no_colocate_nccl_fsdp_vllm_qwen3_5_35b_a3b_h100",
    ],
)
async def test_policy_local_engines_e2e(
    ray_init_fixture,
    colocate_all,
    weight_sync_backend,
    strategy,
    num_engines,
    tp_size,
    distributed_executor_backend,
    model,
    dp_size,
    use_new_inference_servers,
):
    """
    Tests initalizing the policy actor group and inference engine, syncing weights, and performing generation.
    """
    cfg = get_test_actor_config(model)
    # Large MoE policy on 4xH100 can't hold fp32 master weights alongside vLLM,
    # so init in bf16 here. Production keeps fp32 init (FSDP mixed precision
    # handles the bf16 cast during forward).
    if model == QWEN_LARGE_MOE_MODEL:
        cfg.trainer.policy.inference_only_init = True
    cfg.trainer.placement.colocate_all = colocate_all
    cfg.generator.inference_engine.weight_sync_backend = weight_sync_backend
    cfg.trainer.strategy = strategy
    cfg.generator.inference_engine.tensor_parallel_size = tp_size
    cfg.generator.inference_engine.distributed_executor_backend = distributed_executor_backend
    cfg.generator.inference_engine.num_engines = num_engines
    cfg.generator.inference_engine.data_parallel_size = dp_size
    tokenizer = AutoTokenizer.from_pretrained(model)

    # If colocate is True, this will load the engine, sleep, and wake up the engine
    async with InferenceEngineState.create(
        model=model,
        cfg=cfg,
        use_local=True,
        async_engine=cfg.generator.inference_engine.async_engine,
        tp_size=cfg.generator.inference_engine.tensor_parallel_size,
        colocate_all=cfg.trainer.placement.colocate_all,
        sleep_level=2,  # since we explicitly sync weights
        use_new_inference_servers=use_new_inference_servers,
    ) as engines:
        client, pg = engines.client, engines.pg

        # Sleep the inference engine before initializing the policy worker so
        # the GPU is free for FSDP shard allocation (vLLM otherwise holds the
        # bulk of HBM and FSDP init OOMs). The partial wake_up(tags=...) calls
        # below mirror WorkerDispatch.save_weights_for_sampler.
        if colocate_all:
            await client.sleep()

        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=cfg.trainer.placement.colocate_all,
            num_gpus_per_node=cfg.generator.inference_engine.tensor_parallel_size
            * cfg.generator.inference_engine.num_engines
            * cfg.generator.inference_engine.data_parallel_size,
            cfg=cfg,
            rdt_master_actor=(weight_sync_backend == "sharded_rdt"),
        )

        ray.get(
            policy.async_run_ray_method(
                "pass_through", "init_weight_sync_state", client, cfg.generator.inference_engine
            )
        )
        await client.reset_prefix_cache()
        # Partially wake just the "weights" pool so vLLM's param.data has real
        # GPU backing for the broadcast/IPC copy; KV cache is woken after FSDP
        # offloads to CPU below.
        if colocate_all:
            await client.wake_up(tags=["weights"])
        ray.get(
            policy.async_run_ray_method(
                "pass_through", "broadcast_to_inference_engines", client, cfg.generator.inference_engine
            )
        )
        if colocate_all:
            policy.offload_to_cpu()
            await client.wake_up(tags=["kv_cache"])

        sampling_params = get_sampling_params_for_backend(
            cfg.generator.inference_engine.backend, cfg.generator.sampling_params
        )
        outputs = await run_inference(client, get_test_prompts(model), sampling_params, tokenizer=tokenizer)

        print(f"Example output after weight sync: {outputs['responses'][0]}, {outputs['stop_reasons'][0]}")

