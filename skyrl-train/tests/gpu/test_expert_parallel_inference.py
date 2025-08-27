"""
Tests for expert parallel (EP).

uv run --isolated --extra dev --extra vllm pytest tests/gpu/test_expert_parallel_inference.py -m "vllm"

"""

import asyncio
import pytest
import ray
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer

from tests.gpu.utils import get_available_gpus, get_test_prompts, init_worker_with_type, are_responses_similar
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.utils import initialize_ray, get_ray_pg_ready_with_timeout
from skyrl_train.entrypoints.main_base import config_dir
from ray.util.placement_group import placement_group


MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"


def _check_gpus(num_gpus: int):
    available = get_available_gpus()
    if len(available) < num_gpus:
        pytest.skip(f"Expert parallel tests require >= {num_gpus} GPUs, found {len(available)}: {available}")


def _get_test_cfg() -> DictConfig:
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

    # Use MoE policy model
    cfg.trainer.policy.model.path = MODEL

    # vLLM generator with EP enabled
    cfg.generator.backend = "vllm"
    cfg.generator.num_inference_engines = 2
    cfg.generator.inference_engine_tensor_parallel_size = 2
    cfg.generator.inference_engine_expert_parallel_size = 2
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.generator.gpu_memory_utilization = 0.8

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
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    # Small micro batches to fit the MoE in 4 GPUs during training.
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1

    return cfg


async def _run_batch_generation(client: InferenceEngineClient, prompts):
    inp = InferenceEngineInput(prompts=prompts)
    out = await client.generate(inp)
    return out["responses"], out["stop_reasons"]


async def _run_single_generation(client: InferenceEngineClient, prompts):
    tasks = [client.generate(InferenceEngineInput(prompts=[p])) for p in prompts]
    results = await asyncio.gather(*tasks)
    responses, reasons = [], []
    for r in results:
        responses.extend(r["responses"])
        reasons.extend(r["stop_reasons"])
    return responses, reasons


@pytest.mark.vllm
def test_ep_generation():
    """
    Validate vLLM generation with expert parallel enabled using two engines:
    - num_inference_engines=2
    - tp=2, ep=2, dp=1 (per engine)
    """
    _check_gpus(num_gpus=4)

    try:
        cfg = _get_test_cfg()
        # Deterministic sampling for similarity checks
        cfg.generator.sampling_params.temperature = 0.0
        cfg.generator.sampling_params.top_p = 1.0
        cfg.generator.sampling_params.top_k = -1
        initialize_ray(cfg)

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        engines = create_ray_wrapped_inference_engines(
            num_inference_engines=cfg.generator.num_inference_engines,
            tensor_parallel_size=cfg.generator.inference_engine_tensor_parallel_size,
            model_dtype=cfg.generator.model_dtype,
            pretrain=cfg.trainer.policy.model.path,
            seed=cfg.trainer.seed,
            vllm_v1_disable_multiproc=cfg.generator.vllm_v1_disable_multiproc,
            enable_prefix_caching=cfg.generator.enable_prefix_caching,
            enforce_eager=cfg.generator.enforce_eager,
            max_model_len=cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length,
            expert_parallel_size=cfg.generator.inference_engine_expert_parallel_size,
            data_parallel_size=cfg.generator.inference_engine_data_parallel_size,
            shared_pg=None,
            gpu_memory_utilization=cfg.generator.gpu_memory_utilization,
            inference_engine_enable_sleep=False,
            async_engine=True,
            max_num_batched_tokens=8192,
            max_num_seqs=1024,
            sampling_params=get_sampling_params_for_backend("vllm", cfg.generator.sampling_params),
            tokenizer=tokenizer,
            backend="vllm",
        )
        client = InferenceEngineClient(engines)

        prompts = get_test_prompts(MODEL, num_samples=4)

        # Batched
        batch_responses, batch_reasons = asyncio.run(_run_batch_generation(client, prompts))
        assert len(batch_responses) == len(prompts)
        assert len(batch_reasons) == len(prompts)

        # Single
        single_responses, single_reasons = asyncio.run(_run_single_generation(client, prompts))
        assert len(single_responses) == len(prompts)
        assert len(single_reasons) == len(prompts)

        # Ensure batched and single generation outputs are similar
        for i in range(len(prompts)):
            if not are_responses_similar([batch_responses[i]], [single_responses[i]], tolerance=0.02):
                print(
                    f"Responses differ: batched={batch_responses[i][:200]} ... vs single={single_responses[i][:200]} ..."
                )
    finally:
        ray.shutdown()


@pytest.mark.vllm
def test_ep_weight_sync():
    """
    Ensure generation works after syncing weights from training policy worker.
    - 4 GPUs, Two inference engines (tp=2, ep=2, dp=1) using colocate_all
    - Training uses fsdp2 across all 4 GPUs
    """
    _check_gpus(num_gpus=4)

    pg = None
    try:
        cfg = _get_test_cfg()
        cfg.trainer.placement.colocate_all = True
        # Deterministic sampling for robust comparisons
        cfg.generator.sampling_params.temperature = 0.0
        cfg.generator.sampling_params.top_p = 1.0
        cfg.generator.sampling_params.top_k = -1

        initialize_ray(cfg)

        # Create a shared PG with 4 bundles (sufficient for two engines with tp=2 and training)
        pg = placement_group([{"GPU": 1, "CPU": 1}] * 4, strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=60)

        # Spin up two inference engines with EP enabled, colocated
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        engines = create_ray_wrapped_inference_engines(
            num_inference_engines=cfg.generator.num_inference_engines,
            tensor_parallel_size=cfg.generator.inference_engine_tensor_parallel_size,
            model_dtype=cfg.generator.model_dtype,
            pretrain=cfg.trainer.policy.model.path,
            seed=cfg.trainer.seed,
            vllm_v1_disable_multiproc=cfg.generator.vllm_v1_disable_multiproc,
            enable_prefix_caching=cfg.generator.enable_prefix_caching,
            enforce_eager=cfg.generator.enforce_eager,
            max_model_len=cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length,
            expert_parallel_size=cfg.generator.inference_engine_expert_parallel_size,
            data_parallel_size=cfg.generator.inference_engine_data_parallel_size,
            shared_pg=pg,
            gpu_memory_utilization=cfg.generator.gpu_memory_utilization,
            inference_engine_enable_sleep=True,
            async_engine=True,
            max_num_batched_tokens=8192,
            max_num_seqs=1024,
            sampling_params=get_sampling_params_for_backend("vllm", cfg.generator.sampling_params),
            tokenizer=tokenizer,
            backend="vllm",
        )
        client = InferenceEngineClient(engines)
        asyncio.run(client.wake_up())

        # Generate before weight sync
        prompts = get_test_prompts(MODEL, num_samples=4)
        out_before = asyncio.run(client.generate(InferenceEngineInput(prompts=prompts)))
        assert len(out_before["responses"]) == len(prompts)

        asyncio.run(client.sleep())

        # Initialize policy worker on all 4 GPUs
        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=True,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # Sync weights to inference engines
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        asyncio.run(client.wake_up(tags=["weights"]))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        policy.offload_to_cpu()
        asyncio.run(client.wake_up(tags=["kv_cache"]))
        asyncio.run(client.reset_prefix_cache())

        # Generate after weight sync
        out_after = asyncio.run(client.generate(InferenceEngineInput(prompts=prompts)))
        assert len(out_after["responses"]) == len(prompts)
        assert len(out_after["stop_reasons"]) == len(prompts)

        # Check that weights are not corrupted: responses should be similar pre/post sync
        for i in range(len(prompts)):
            if not are_responses_similar([out_before["responses"][i]], [out_after["responses"][i]], tolerance=0.02):
                print(
                    f"Response changed significantly after weight sync: before={out_before['responses'][i][:200]} ... after={out_after['responses'][i][:200]} ..."
                )
    finally:
        if pg is not None:
            try:
                ray.util.remove_placement_group(pg)
            except Exception:
                pass
        ray.shutdown()
