"""
Tests for expert parallel (EP).

uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/test_expert_parallel_inference.py

"""

import asyncio

import pytest
import ray

from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.utils.tok import get_tokenizer
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    are_responses_similar,
    get_available_gpus,
    get_test_actor_config,
    get_test_prompts,
    init_worker_with_type,
    run_inference,
)

MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
NUM_GPUS = 4  # Should be divisible by 2


def _check_gpus(num_gpus: int):
    available = get_available_gpus()
    if len(available) < num_gpus:
        pytest.skip(f"Expert parallel tests require >= {num_gpus} GPUs, found {len(available)}: {available}")


def _get_test_cfg() -> SkyRLTrainConfig:
    cfg = get_test_actor_config()

    # Use MoE policy model
    cfg.trainer.policy.model.path = MODEL

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

    return cfg


async def _run_single_generation(client, prompts, sampling_params, tokenizer):
    tasks = [run_inference(client, [p], sampling_params, tokenizer=tokenizer) for p in prompts]
    results = await asyncio.gather(*tasks)
    responses, reasons = [], []
    for r in results:
        responses.extend(r["responses"])
        reasons.extend(r["stop_reasons"])
    return responses, reasons


def test_ep_generation():
    """
    Ensure vLLM generation with expert parallel enabled (EP=2) runs without errors.
    Validate that the number of outputs matches the number of inputs.
    """
    _check_gpus(num_gpus=NUM_GPUS)

    try:
        cfg = _get_test_cfg()
        # Deterministic sampling for stable execution
        cfg.generator.sampling_params.temperature = 0.0
        cfg.generator.sampling_params.top_p = 1.0
        cfg.generator.sampling_params.top_k = -1

        with InferenceEngineState.create(cfg, sleep_level=1) as state:
            tokenizer = get_tokenizer(MODEL)
            prompts = get_test_prompts(MODEL, num_samples=4)
            sampling_params = get_sampling_params_for_backend(
                cfg.generator.inference_engine.backend, cfg.generator.sampling_params
            )

            responses, reasons = asyncio.run(_run_single_generation(state.client, prompts, sampling_params, tokenizer))
            assert len(responses) == len(prompts)
            assert len(reasons) == len(prompts)
    finally:
        ray.shutdown()


def test_ep_weight_sync(ray_init_fixture):
    """
    Ensure generation works after syncing weights from training policy worker.
    """
    _check_gpus(num_gpus=NUM_GPUS)

    cfg = _get_test_cfg()
    cfg.trainer.placement.colocate_all = True
    # Deterministic sampling for robust comparisons
    cfg.generator.sampling_params.temperature = 0.0
    cfg.generator.sampling_params.top_p = 1.0
    cfg.generator.sampling_params.top_k = -1

    with InferenceEngineState.create(cfg, colocate_all=True) as state:
        # Generate before weight sync
        tokenizer = get_tokenizer(MODEL)
        prompts = get_test_prompts(MODEL, num_samples=4)
        sampling_params = get_sampling_params_for_backend(
            cfg.generator.inference_engine.backend, cfg.generator.sampling_params
        )
        out_before = asyncio.run(run_inference(state.client, prompts, sampling_params, tokenizer=tokenizer))
        assert len(out_before["responses"]) == len(prompts)

        asyncio.run(state.client.sleep())

        # Initialize policy worker on the same placement group
        policy = init_worker_with_type(
            "policy",
            shared_pg=state.pg,
            colocate_all=True,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # Sync weights to inference engines
        ray.get(
            policy.async_run_ray_method(
                "pass_through",
                "init_weight_sync_state",
                state.client,
                cfg.generator.inference_engine,
            )
        )
        asyncio.run(state.client.wake_up(tags=["weights"]))
        ray.get(
            policy.async_run_ray_method(
                "pass_through",
                "broadcast_to_inference_engines",
                state.client,
                cfg.generator.inference_engine,
            )
        )
        policy.offload_to_cpu()
        asyncio.run(state.client.wake_up(tags=["kv_cache"]))
        asyncio.run(state.client.reset_prefix_cache())

        # Generate after weight sync
        out_after = asyncio.run(run_inference(state.client, prompts, sampling_params, tokenizer=tokenizer))
        assert len(out_after["responses"]) == len(prompts)
        assert len(out_after["stop_reasons"]) == len(prompts)

        # Check that weights are not corrupted: responses should be similar pre/post sync
        for i in range(len(prompts)):
            if not are_responses_similar([out_before["responses"][i]], [out_after["responses"][i]], tolerance=0.02):
                print(
                    f"Response changed significantly after weight sync: before={out_before['responses'][i][:200]} ... after={out_after['responses'][i][:200]} ..."
                )
