"""
# Run only vllm tests (requires vllm extra):
uv run --isolated --extra dev --extra vllm pytest tests/gpu/gpu_ci/test_engine_generation.py -m "vllm"

# Run only sglang tests (requires sglang extra):
uv run --isolated --extra dev --extra sglang pytest tests/gpu/gpu_ci/test_engine_generation.py -m "sglang"
"""

import pytest

from skyrl_train.env_vars import _SKYRL_USE_NEW_INFERENCE
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
import asyncio
from tests.gpu.utils import (
    are_responses_similar,
    get_test_prompts,
    init_inference_engines,
    init_remote_inference_servers,
)
from transformers import AutoTokenizer
from skyrl_train.config import SkyRLConfig
from skyrl_train.inference_engines.base import InferenceEngineInput

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_actor_config() -> SkyRLConfig:
    """Get base config with test-specific overrides."""
    cfg = SkyRLConfig()
    cfg.trainer.policy.model.path = MODEL

    cfg.generator.sampling_params.temperature = 0.0
    cfg.generator.sampling_params.top_p = 1
    cfg.generator.sampling_params.top_k = -1
    cfg.generator.sampling_params.max_generate_length = 1024
    cfg.generator.sampling_params.min_p = 0.0
    cfg.generator.sampling_params.logprobs = None

    return cfg


def init_ray_inference_engines(backend: str, tp_size: int, pp_size: int, dp_size: int, config: SkyRLConfig):
    """Initialize ray-wrapped inference engines for the specified backend.

    Returns:
        Tuple of (client, pg, router, server_group) where router and server_group
        may be None for the old inference pathway.
    """
    # Set config parameters for new inference pathway (used by build_vllm_cli_args)
    config.generator.inference_engine_tensor_parallel_size = tp_size
    config.generator.inference_engine_pipeline_parallel_size = pp_size
    config.generator.inference_engine_data_parallel_size = dp_size

    client, pg, router, server_group = init_inference_engines(
        config,
        model=config.trainer.policy.model.path,
        async_engine=True,
        use_local=True,
        tp_size=tp_size,
        colocate_all=True,
        backend=backend,
        gpu_memory_utilization=0.8,
        num_inference_engines=1,
        # SGLang always discards weights, so sleep_level is not applicable.
        sleep_level=1 if backend == "vllm" else 2,
    )
    return client, pg, router, server_group


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


@pytest.mark.skipif(_SKYRL_USE_NEW_INFERENCE, reason="New inference pathway doesn't support text based generation")
@pytest.mark.parametrize(
    "backend,tp_size,pp_size,dp_size",
    [
        pytest.param("vllm", 2, 1, 1, marks=pytest.mark.vllm),
        pytest.param("vllm", 2, 1, 2, marks=pytest.mark.vllm),
        pytest.param("vllm", 2, 2, 1, marks=pytest.mark.vllm),  # TP=2, PP=2
        # TODO(Charlie): add TP > 1 tests for sglang when we support it
        pytest.param("sglang", 1, 1, 1, marks=pytest.mark.sglang),
    ],
    ids=["vllm_tp2", "vllm_dp2", "vllm_tp2_pp2", "sglang"],
)
def test_inference_engines_generation(ray_init_fixture, backend: str, tp_size: int, pp_size: int, dp_size: int):
    """
    Tests generation with both remote and ray-wrapped engines for the specified backend.
    """
    cfg = get_test_actor_config()
    cfg.generator.backend = backend

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
    try:
        llm_client, pg, router, server_group = init_ray_inference_engines(backend, tp_size, pp_size, dp_size, cfg)
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
        if "router" in locals() and router is not None:
            router.shutdown()
        if "server_group" in locals() and server_group is not None:
            server_group.shutdown()


@pytest.mark.parametrize(
    "backend,tp_size,pp_size,dp_size",
    [
        pytest.param("vllm", 2, 1, 1, marks=pytest.mark.vllm),
        pytest.param("vllm", 2, 2, 1, marks=pytest.mark.vllm),
        pytest.param("vllm", 2, 1, 2, marks=pytest.mark.vllm),
        # TODO(Charlie): add TP > 1 tests for sglang when we support it
        pytest.param("sglang", 1, 1, 1, marks=pytest.mark.sglang),
    ],
    ids=["vllm_tp2_pp1_dp1", "vllm_tp2_pp2_dp1", "vllm_tp2_pp1_dp2", "sglang_tp1_pp1_dp1"],
)
def test_token_based_generation(ray_init_fixture, backend: str, tp_size: int, pp_size: int, dp_size: int):
    """Test generation using prompt_token_ids for the specified backend."""

    cfg = get_test_actor_config()
    cfg.generator.backend = backend

    prompts = get_test_prompts(MODEL, 3)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_token_ids = tokenizer.apply_chat_template(
        prompts, add_generation_prompt=True, tokenize=True, return_dict=True
    )["input_ids"]

    try:
        llm_client, pg, router, server_group = init_ray_inference_engines(
            backend, tp_size=tp_size, pp_size=pp_size, dp_size=dp_size, config=cfg
        )
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

        # Ensure batched and single generation outputs are (roughly) the same
        for i in range(len(prompts)):
            if not are_responses_similar(token_batch_responses[i], token_single_responses[i], tolerance=0.01):
                print(
                    f"Token batch and single generation responses are not similar, got batch={token_batch_responses[i]} and single={token_single_responses[i]}"
                )
    finally:
        if "router" in locals() and router is not None:
            router.shutdown()
        if "server_group" in locals() and server_group is not None:
            server_group.shutdown()


@pytest.mark.skipif(_SKYRL_USE_NEW_INFERENCE, reason="New inference pathway doesn't support text based generation")
@pytest.mark.parametrize(
    "backend,tp_size,pp_size,dp_size",
    [
        pytest.param("vllm", 2, 1, 1, marks=pytest.mark.vllm),
        # TODO(Charlie): add TP > 1 tests for sglang when we support it
        pytest.param("sglang", 1, 1, 1, marks=pytest.mark.sglang),
    ],
    ids=["vllm_tp2_pp1_dp1", "sglang_tp1_pp1_dp1"],
)
def test_token_based_generation_consistency(ray_init_fixture, backend: str, tp_size: int, pp_size: int, dp_size: int):
    cfg = get_test_actor_config()
    cfg.generator.backend = backend

    prompts = get_test_prompts(MODEL, 3)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_token_ids = tokenizer.apply_chat_template(
        prompts, add_generation_prompt=True, tokenize=True, return_dict=True
    )["input_ids"]

    try:
        llm_client, pg, router, server_group = init_ray_inference_engines(
            backend, tp_size=tp_size, pp_size=pp_size, dp_size=dp_size, config=cfg
        )
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)

        # Batch generation with tokens
        token_batch_responses, _ = asyncio.run(
            run_batch_generation_with_tokens(llm_client, prompt_token_ids, sampling_params)
        )
        assert len(token_batch_responses) == len(prompts)

        # Compare with prompt-based generation
        prompt_responses, _ = asyncio.run(run_batch_generation(llm_client, prompts, sampling_params))
        assert len(prompt_responses) == len(prompts)

        # Outputs should be similar since we're using the same inputs
        for i in range(len(prompts)):
            if not are_responses_similar([token_batch_responses[i]], [prompt_responses[i]], tolerance=0.01):
                print(
                    f"Token and prompt responses differ: token={token_batch_responses[i]}, prompt={prompt_responses[i]}"
                )
    finally:
        if "router" in locals() and router is not None:
            router.shutdown()
        if "server_group" in locals() and server_group is not None:
            server_group.shutdown()


# TODO: Remove this once sample API is also supported in the new inference pathway
@pytest.mark.skipif(_SKYRL_USE_NEW_INFERENCE, reason="New inference pathway doesn't support sample API yet")
@pytest.mark.parametrize(
    "backend,tp_size,dp_size",
    [
        pytest.param("vllm", 2, 1, marks=pytest.mark.vllm),
    ],
    ids=["vllm_tp2"],
)
def test_sample_api(ray_init_fixture, backend: str, tp_size: int, dp_size: int):
    """Test the Tinker-compatible sample() API for generating multiple independent samples."""
    cfg = get_test_actor_config()
    cfg.generator.backend = backend
    cfg.generator.sampling_params.temperature = 0.7

    prompts = get_test_prompts(MODEL, 1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_token_ids = tokenizer.apply_chat_template(
        prompts, add_generation_prompt=True, tokenize=True, return_dict=True
    )["input_ids"][0]

    try:
        llm_client, pg, router, server_group = init_ray_inference_engines(
            backend, tp_size=tp_size, pp_size=1, dp_size=dp_size, config=cfg
        )
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)

        num_samples = 3

        async def run_sample():
            return await llm_client.sample(
                prompt_token_ids=prompt_token_ids,
                num_samples=num_samples,
                sampling_params=sampling_params,
            )

        output = asyncio.run(run_sample())

        assert len(output["response_ids"]) == num_samples
        assert len(output["responses"]) == num_samples
        assert len(output["stop_reasons"]) == num_samples

        for i, response_ids in enumerate(output["response_ids"]):
            assert isinstance(response_ids, list)
            assert len(response_ids) > 0
            assert all(isinstance(t, int) for t in response_ids)

        unique_responses = set(output["responses"])
        print(f"Generated {len(unique_responses)} unique responses from {num_samples} samples")
        for i, resp in enumerate(output["responses"]):
            print(f"Sample {i}: {resp[:100]}..." if len(resp) > 100 else f"Sample {i}: {resp}")
    finally:
        if "router" in locals() and router is not None:
            router.shutdown()
        if "server_group" in locals() and server_group is not None:
            server_group.shutdown()
