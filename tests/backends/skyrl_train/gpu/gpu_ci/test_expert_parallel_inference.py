"""
Tests for expert parallel (EP).

uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_expert_parallel_inference.py
"""

import asyncio

import pytest
import pytest_asyncio
import ray

from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.utils.tok import get_tokenizer
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    _ensure_chat_template,
    are_responses_similar,
    get_test_actor_config,
    get_test_prompts,
    init_worker_with_type,
    run_inference,
)

MODEL = "huihui-ai/Huihui-MoE-0.8B-2E"
NUM_GPUS = 4  # Should be divisible by 2


@pytest.fixture(scope="module")
def test_config() -> SkyRLTrainConfig:
    cfg = get_test_actor_config()

    # Use MoE policy model
    cfg.trainer.policy.model.path = MODEL

    # Colocated
    cfg.trainer.placement.colocate_all = True

    # vLLM generator with EP enabled
    cfg.generator.inference_engine.backend = "vllm"
    cfg.generator.inference_engine.async_engine = True
    cfg.generator.inference_engine.num_engines = NUM_GPUS // 2
    cfg.generator.inference_engine.tensor_parallel_size = 2
    cfg.generator.inference_engine.expert_parallel_size = 2
    cfg.generator.inference_engine.data_parallel_size = 1
    cfg.generator.inference_engine.gpu_memory_utilization = 0.8

    # Small lengths for faster tests
    cfg.generator.max_input_length = 2048
    cfg.generator.sampling_params.max_generate_length = 512

    # Training knobs for tests
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.train_batch_size = 128
    cfg.trainer.policy_mini_batch_size = 128
    cfg.trainer.micro_forward_batch_size_per_gpu = 8
    cfg.trainer.micro_train_batch_size_per_gpu = 8
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    # Small micro batches to fit the MoE in 2 GPUs during training.
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1

    # Deterministic sampling for stable execution
    cfg.generator.sampling_params.temperature = 0.0
    cfg.generator.sampling_params.top_p = 1.0
    cfg.generator.sampling_params.top_k = -1

    return cfg


@pytest_asyncio.fixture(scope="module")
async def inference_state(module_scoped_ray_init_fixture, test_config):
    async with InferenceEngineState.create(test_config, sleep_level=1) as engines:
        yield engines


async def _run_single_generation(client, prompts, sampling_params, tokenizer):
    tasks = [run_inference(client, [p], sampling_params, tokenizer=tokenizer) for p in prompts]
    results = await asyncio.gather(*tasks)
    responses, reasons = [], []
    for r in results:
        responses.extend(r["responses"])
        reasons.extend(r["stop_reasons"])
    return responses, reasons


@pytest.mark.asyncio
async def test_ep_generation(inference_state: InferenceEngineState, test_config: SkyRLTrainConfig):
    """
    Ensure vLLM generation with expert parallel enabled (EP=2) runs without errors.
    Validate that the number of outputs matches the number of inputs.
    """
    tokenizer = get_tokenizer(test_config.trainer.policy.model.path)
    _ensure_chat_template(tokenizer)
    inference_state.client.tokenizer = tokenizer
    prompts = get_test_prompts(MODEL, num_samples=4)
    sampling_params = get_sampling_params_for_backend(
        test_config.generator.inference_engine.backend, test_config.generator.sampling_params
    )

    responses, reasons = await _run_single_generation(inference_state.client, prompts, sampling_params, tokenizer)
    assert len(responses) == len(prompts)
    assert len(reasons) == len(prompts)


@pytest.mark.asyncio
async def test_ep_weight_sync(inference_state: InferenceEngineState, test_config: SkyRLTrainConfig):
    """
    Ensure generation works after syncing weights from training policy worker.
    """

    # Generate before weight sync
    tokenizer = get_tokenizer(test_config.trainer.policy.model.path)
    _ensure_chat_template(tokenizer)
    inference_state.client.tokenizer = tokenizer
    prompts = get_test_prompts(test_config.trainer.policy.model.path, num_samples=4)
    sampling_params = get_sampling_params_for_backend(
        test_config.generator.inference_engine.backend, test_config.generator.sampling_params
    )
    out_before = await run_inference(inference_state.client, prompts, sampling_params, tokenizer=tokenizer)
    assert len(out_before["responses"]) == len(prompts)

    await inference_state.client.sleep()

    # Initialize policy worker on the same placement group
    policy = init_worker_with_type(
        "policy",
        shared_pg=inference_state.pg,
        colocate_all=True,
        num_gpus_per_node=test_config.trainer.placement.policy_num_gpus_per_node,
        cfg=test_config,
    )

    # Sync weights to inference engines
    ray.get(
        policy.async_run_ray_method(
            "pass_through",
            "init_weight_sync_state",
            inference_state.client,
            test_config.generator.inference_engine,
        )
    )
    await inference_state.client.wake_up(tags=["weights"])
    ray.get(
        policy.async_run_ray_method(
            "pass_through",
            "broadcast_to_inference_engines",
            inference_state.client,
            test_config.generator.inference_engine,
        )
    )
    policy.offload_to_cpu()
    await inference_state.client.wake_up(tags=["kv_cache"])
    await inference_state.client.reset_prefix_cache()

    # Generate after weight sync
    out_after = await run_inference(inference_state.client, prompts, sampling_params, tokenizer=tokenizer)
    assert len(out_after["responses"]) == len(prompts)
    assert len(out_after["stop_reasons"]) == len(prompts)

    # Check that weights are not corrupted: responses should be similar pre/post sync
    for i in range(len(prompts)):
        if not are_responses_similar([out_before["responses"][i]], [out_after["responses"][i]], tolerance=0.02):
            print(
                f"Response changed significantly after weight sync: before={out_before['responses'][i][:200]} ... after={out_after['responses'][i][:200]} ..."
            )
