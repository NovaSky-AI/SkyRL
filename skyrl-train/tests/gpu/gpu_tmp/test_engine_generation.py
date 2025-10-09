"""
# Run only vllm tests (requires vllm extra):
uv run --isolated --extra dev --extra vllm pytest tests/gpu/gpu_ci/test_engine_generation.py -m "vllm"

# Run only sglang tests (requires sglang extra):
uv run --isolated --extra dev --extra sglang pytest tests/gpu/gpu_ci/test_engine_generation.py -m "sglang"
"""

import pytest
import ray
import hydra
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
import asyncio
from tests.gpu.utils import are_responses_similar, get_test_prompts, init_remote_inference_servers
from transformers import AutoTokenizer
from omegaconf import DictConfig
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import config_dir

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def get_test_actor_config() -> DictConfig:
    """Get base config with test-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

        cfg.trainer.policy.model.path = MODEL

        cfg.generator.sampling_params.temperature = 0.0
        cfg.generator.sampling_params.top_p = 1
        cfg.generator.sampling_params.top_k = -1
        cfg.generator.sampling_params.max_generate_length = 1024
        cfg.generator.sampling_params.min_p = 0.0
        cfg.generator.sampling_params.logprobs = None

        return cfg


def init_ray_inference_engines(backend: str, tp_size: int, dp_size: int, config: DictConfig) -> InferenceEngineClient:
    """Initialize ray-wrapped inference engines for the specified backend"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    engine = create_ray_wrapped_inference_engines(
        num_inference_engines=1,
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        model_dtype="bfloat16",
        pretrain=MODEL,
        seed=42,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        shared_pg=None,
        gpu_memory_utilization=0.8,
        inference_engine_enable_sleep=False,
        async_engine=True,
        max_num_batched_tokens=8192,
        max_num_seqs=1024,
        tokenizer=tokenizer,
        backend=backend,
    )
    client = InferenceEngineClient(engine, tokenizer, config)
    return client


async def run_batch_generation(client, prompts, sampling_params):
    engine_input = InferenceEngineInput(prompts=prompts, sampling_params=sampling_params)
    engine_output = await client.generate(engine_input)
    return engine_output["responses"], engine_output["stop_reasons"]


async def run_single_generation(client, prompts, sampling_params):
    tasks = []
    for prompt in prompts:
        engine_input = InferenceEngineInput(prompts=[prompt], sampling_params=sampling_params)
        task = client.generate(engine_input)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    responses = []
    finish_reasons = []
    for result in results:
        responses.extend(result["responses"])
        finish_reasons.extend(result["stop_reasons"])

    return responses, finish_reasons


async def run_batch_generation_with_tokens(client, prompt_token_ids, sampling_params):
    engine_input = InferenceEngineInput(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
    engine_output = await client.generate(engine_input)
    return engine_output["responses"], engine_output["stop_reasons"]


async def run_single_generation_with_tokens(client, prompt_token_ids, sampling_params):
    tasks = []
    for tokens in prompt_token_ids:
        engine_input = InferenceEngineInput(prompt_token_ids=[tokens], sampling_params=sampling_params)
        task = client.generate(engine_input)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    responses = []
    finish_reasons = []
    for result in results:
        responses.extend(result["responses"])
        finish_reasons.extend(result["stop_reasons"])

    return responses, finish_reasons


@pytest.mark.parametrize(
    "backend,tp_size,dp_size",
    [
        pytest.param("vllm", 2, 1, marks=pytest.mark.vllm),
        pytest.param("vllm", 2, 2, marks=pytest.mark.vllm),
        # TODO(Charlie): add TP > 1 tests for sglang when we support it
        pytest.param("sglang", 1, 1, marks=pytest.mark.sglang),
    ],
    ids=["vllm", "vllm_dp2", "sglang"],
)
def test_inference_engines_generation(backend: str, tp_size: int, dp_size: int):
    """
    Tests generation with both remote and ray-wrapped engines for the specified backend.
    """
    try:
        cfg = get_test_actor_config()
        cfg.generator.backend = backend
        initialize_ray(cfg)

        prompts = get_test_prompts(MODEL)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        try:
            llm_client, remote_server_process = init_remote_inference_servers(tp_size, backend, tokenizer, cfg, MODEL)
            sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)

            # Batched generation
            remote_batch_responses, batch_finish_reasons = asyncio.run(
                run_batch_generation(llm_client, prompts, sampling_params)
            )
            assert len(remote_batch_responses) == len(
                prompts
            ), f"Number of responses should match number of prompts, got {len(remote_batch_responses)} responses but {len(prompts)} prompts"
            assert len(batch_finish_reasons) == len(
                prompts
            ), f"Number of finish reasons should match number of prompts, got {len(batch_finish_reasons)} finish reasons but {len(prompts)} prompts"

            # Single generation (ie, submit individual requests)
            remote_single_responses, single_finish_reasons = asyncio.run(
                run_single_generation(llm_client, prompts, sampling_params)
            )
            assert len(remote_single_responses) == len(
                prompts
            ), f"Number of responses should match number of prompts, got {len(remote_single_responses)} responses but {len(prompts)} prompts"
            assert len(single_finish_reasons) == len(
                prompts
            ), f"Number of finish reasons should match number of prompts, got {len(single_finish_reasons)} finish reasons but {len(prompts)} prompts"

            # Ensure batched and single generation outputs are (roughly) the same
            for i in range(len(prompts)):
                if not are_responses_similar(remote_batch_responses[i], remote_single_responses[i], tolerance=0.01):
                    print(
                        f"Remote batch and single generation responses are not similar, got batch={remote_batch_responses[i]} and single={remote_single_responses[i]}"
                    )

        finally:
            if "remote_server_process" in locals():
                remote_server_process.terminate()
                remote_server_process.wait()

        # Get responses from Ray engine
        llm_client = init_ray_inference_engines(backend, tp_size, dp_size, cfg)
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)

        # Batched generation
        local_batch_responses, batch_finish_reasons = asyncio.run(
            run_batch_generation(llm_client, prompts, sampling_params)
        )
        assert len(local_batch_responses) == len(
            prompts
        ), f"Number of responses should match number of prompts, got {len(local_batch_responses)} responses but {len(prompts)} prompts"
        assert len(batch_finish_reasons) == len(
            prompts
        ), f"Number of finish reasons should match number of prompts, got {len(batch_finish_reasons)} finish reasons but {len(prompts)} prompts"

        # Single generation (ie, submit individual requests)
        local_single_responses, single_finish_reasons = asyncio.run(
            run_single_generation(llm_client, prompts, sampling_params)
        )
        assert len(local_single_responses) == len(
            prompts
        ), f"Number of responses should match number of prompts, got {len(local_single_responses)} responses but {len(prompts)} prompts"
        assert len(single_finish_reasons) == len(
            prompts
        ), f"Number of finish reasons should match number of prompts, got {len(single_finish_reasons)} finish reasons but {len(prompts)} prompts"

        # Ensure batched and single generation outputs are (roughly) the same
        for i in range(len(prompts)):
            if not are_responses_similar(local_batch_responses[i], local_single_responses[i], tolerance=0.01):
                print(
                    f"Local batch and single generation responses are not similar, got batch={local_batch_responses[i]} and single={local_single_responses[i]}"
                )

        # Finally, ensure that remote and local outputs are (roughly) the same
        for i in range(len(prompts)):
            if not are_responses_similar(remote_batch_responses[i], local_batch_responses[i], tolerance=0.01):
                print(
                    f"Remote and local batch generation responses are not similar, got remote={remote_batch_responses[i]} and local={local_batch_responses[i]}"
                )

    finally:
        ray.shutdown()


@pytest.mark.parametrize(
    "backend,tp_size,dp_size",
    [
        pytest.param("vllm", 2, 2, marks=pytest.mark.vllm),
        # TODO(Charlie): add TP > 1 tests for sglang when we support it
        pytest.param("sglang", 1, 1, marks=pytest.mark.sglang),
    ],
    ids=["vllm_dp2", "sglang"],
)
def test_token_based_generation(backend: str, tp_size: int, dp_size: int):
    """Test generation using prompt_token_ids for the specified backend."""

    try:
        cfg = get_test_actor_config()
        cfg.generator.backend = backend
        initialize_ray(cfg)

        prompts = get_test_prompts(MODEL, 3)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        prompt_token_ids = tokenizer.apply_chat_template(
            prompts, add_generation_prompt=True, tokenize=True, return_dict=True
        )["input_ids"]

        llm_client = init_ray_inference_engines(backend, tp_size, dp_size, cfg)
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)

        # Test batch generation with tokens
        token_batch_responses, _ = asyncio.run(
            run_batch_generation_with_tokens(llm_client, prompt_token_ids, sampling_params)
        )
        assert len(token_batch_responses) == len(prompts)

        # Test single generation with tokens
        token_single_responses, _ = asyncio.run(
            run_single_generation_with_tokens(llm_client, prompt_token_ids, sampling_params)
        )
        assert len(token_single_responses) == len(prompts)

        # Compare with prompt-based generation
        prompt_responses, _ = asyncio.run(run_batch_generation(llm_client, prompts, sampling_params))

        # Outputs should be similar since we're using the same inputs
        for i in range(len(prompts)):
            if not are_responses_similar([token_batch_responses[i]], [prompt_responses[i]], tolerance=0.01):
                print(
                    f"Token and prompt responses differ: token={token_batch_responses[i]}, prompt={prompt_responses[i]}"
                )

    finally:
        ray.shutdown()
