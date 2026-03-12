"""
To run:
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_engine_generation.py
"""

import asyncio

import pytest
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.env_vars import _SKYRL_USE_NEW_INFERENCE
from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    are_responses_similar,
    get_test_prompts,
    init_remote_inference_servers,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MOE_MODEL = "Qwen/Qwen1.5-MoE-A2.7B"


def get_test_actor_config() -> SkyRLTrainConfig:
    """Get base config with test-specific overrides."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL

    cfg.generator.sampling_params.temperature = 0.0
    cfg.generator.sampling_params.top_p = 1
    cfg.generator.sampling_params.top_k = -1
    cfg.generator.sampling_params.max_generate_length = 1024
    cfg.generator.sampling_params.min_p = 0.0
    cfg.generator.sampling_params.logprobs = None

    return cfg


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
    "tp_size,pp_size,dp_size",
    [
        pytest.param(2, 1, 1),
        pytest.param(2, 1, 2),
        pytest.param(2, 2, 1),  # TP=2, PP=2
    ],
    ids=["tp2_pp1_dp1", "tp2_pp1_dp2", "tp2_pp2_dp1"],
)
def test_inference_engines_generation(ray_init_fixture, tp_size: int, pp_size: int, dp_size: int):
    """
    Tests generation with both remote and ray-wrapped engines.
    """
    cfg = get_test_actor_config()

    prompts = get_test_prompts(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    try:
        llm_client, remote_server_process = init_remote_inference_servers(tp_size, "vllm", tokenizer, cfg, MODEL)
        sampling_params = get_sampling_params_for_backend(
            cfg.generator.inference_engine.backend, cfg.generator.sampling_params
        )

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

    # Set config parameters for new inference pathway
    cfg.generator.inference_engine.tensor_parallel_size = tp_size
    cfg.generator.inference_engine.pipeline_parallel_size = pp_size
    cfg.generator.inference_engine.data_parallel_size = dp_size

    # Get responses from Ray engine
    with InferenceEngineState.create(cfg, sleep_level=1) as engines:
        llm_client = engines.client
        sampling_params = get_sampling_params_for_backend(
            cfg.generator.inference_engine.backend, cfg.generator.sampling_params
        )

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


@pytest.mark.parametrize(
    "tp_size,pp_size,dp_size,model,distributed_executor_backend",
    [
        pytest.param(2, 1, 1, MODEL, "ray"),
        pytest.param(2, 2, 1, MODEL, "ray"),
        pytest.param(2, 1, 2, MOE_MODEL, "ray"),
        pytest.param(2, 1, 2, MOE_MODEL, "mp"),
    ],
    ids=["tp2_pp1_dp1_ray", "tp2_pp2_dp1_ray", "tp2_pp1_dp2_moe_ray", "tp2_pp1_dp2_moe_mp"],
)
def test_token_based_generation(
    ray_init_fixture, tp_size: int, pp_size: int, dp_size: int, model: str, distributed_executor_backend: str
):
    """Test generation using prompt_token_ids."""

    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = model

    prompts = get_test_prompts(model, 3)
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt_token_ids = tokenizer.apply_chat_template(
        prompts, add_generation_prompt=True, tokenize=True, return_dict=True
    )["input_ids"]

    cfg.generator.inference_engine.tensor_parallel_size = tp_size
    cfg.generator.inference_engine.pipeline_parallel_size = pp_size
    cfg.generator.inference_engine.data_parallel_size = dp_size
    cfg.generator.inference_engine.distributed_executor_backend = distributed_executor_backend

    with InferenceEngineState.create(cfg, sleep_level=1) as engines:
        llm_client = engines.client
        sampling_params = get_sampling_params_for_backend(
            cfg.generator.inference_engine.backend, cfg.generator.sampling_params
        )

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


@pytest.mark.skipif(_SKYRL_USE_NEW_INFERENCE, reason="New inference pathway doesn't support text based generation")
@pytest.mark.parametrize(
    "tp_size,pp_size,dp_size",
    [
        pytest.param(2, 1, 1),
    ],
    ids=["tp2_pp1_dp1"],
)
def test_token_based_generation_consistency(ray_init_fixture, tp_size: int, pp_size: int, dp_size: int):
    cfg = get_test_actor_config()

    prompts = get_test_prompts(MODEL, 3)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_token_ids = tokenizer.apply_chat_template(
        prompts, add_generation_prompt=True, tokenize=True, return_dict=True
    )["input_ids"]

    cfg.generator.inference_engine.tensor_parallel_size = tp_size
    cfg.generator.inference_engine.pipeline_parallel_size = pp_size
    cfg.generator.inference_engine.data_parallel_size = dp_size

    with InferenceEngineState.create(cfg, sleep_level=1) as engines:
        llm_client = engines.client
        sampling_params = get_sampling_params_for_backend(
            cfg.generator.inference_engine.backend, cfg.generator.sampling_params
        )

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


@pytest.mark.skipif(_SKYRL_USE_NEW_INFERENCE, reason="Old sample API not used with new inference path")
@pytest.mark.parametrize(
    "tp_size,dp_size",
    [
        pytest.param(2, 1),
    ],
    ids=["tp2"],
)
def test_sample_api(ray_init_fixture, tp_size: int, dp_size: int):
    """Test the Tinker-compatible sample() API for generating multiple independent samples."""
    cfg = get_test_actor_config()
    cfg.generator.sampling_params.temperature = 0.7

    prompts = get_test_prompts(MODEL, 1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_token_ids = tokenizer.apply_chat_template(
        prompts, add_generation_prompt=True, tokenize=True, return_dict=True
    )["input_ids"][0]

    cfg.generator.inference_engine.tensor_parallel_size = tp_size
    cfg.generator.inference_engine.data_parallel_size = dp_size

    with InferenceEngineState.create(cfg, sleep_level=1) as engines:
        llm_client = engines.client
        sampling_params = get_sampling_params_for_backend(
            cfg.generator.inference_engine.backend, cfg.generator.sampling_params
        )

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


@pytest.mark.parametrize(
    "tp_size,dp_size",
    [
        pytest.param(2, 1),
    ],
    ids=["tp2"],
)
def test_sample_api_remote(ray_init_fixture, tp_size: int, dp_size: int):
    """Test the sample() API via RemoteInferenceClient (new inference path).

    Makes two calls to validate both output correctness and sampling behavior:
      - Call A (temp=1.0): schema checks, token decode, diversity across samples
      - Call B (temp=0.0): determinism (all samples should be identical)
    """
    cfg = get_test_actor_config()

    prompts = get_test_prompts(MODEL, 1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_token_ids = tokenizer.apply_chat_template(
        prompts, add_generation_prompt=True, tokenize=True, return_dict=True
    )["input_ids"][0]

    cfg.generator.inference_engine.tensor_parallel_size = tp_size
    cfg.generator.inference_engine.data_parallel_size = dp_size

    num_samples = 3

    with InferenceEngineState.create(cfg, sleep_level=1, use_new_inference_servers=True) as engines:
        llm_client = engines.client

        def build_payload(temperature):
            return {
                "json": {
                    "prompt": {"chunks": [{"tokens": prompt_token_ids}]},
                    "num_samples": 1,
                    "sampling_params": {
                        "temperature": temperature,
                        "max_tokens": cfg.generator.sampling_params.max_generate_length,
                        "top_k": cfg.generator.sampling_params.top_k,
                        "top_p": cfg.generator.sampling_params.top_p,
                    },
                },
                "headers": {},
            }

        async def run_samples(temperature, n):
            results = []
            for _ in range(n):
                output = await llm_client.sample(build_payload(temperature))
                assert output["type"] == "sample", f"Expected type 'sample', got {output['type']!r}"
                assert len(output["sequences"]) == 1, f"Expected 1 sequence per call, got {len(output['sequences'])}"
                results.append(output["sequences"][0])
            return results

        # --- Call A: temp=1.0, expect diverse outputs ---
        sequences = asyncio.run(run_samples(1.0, num_samples))

        decoded_texts = []
        for i, seq in enumerate(sequences):
            assert seq["stop_reason"] in ("stop", "length"), f"Unexpected stop_reason: {seq['stop_reason']}"
            assert isinstance(seq["tokens"], list), f"tokens should be a list, got {type(seq['tokens'])}"
            assert len(seq["tokens"]) > 0, f"Sequence {i} has no tokens"
            assert all(isinstance(t, int) for t in seq["tokens"]), f"Sequence {i} contains non-int tokens"
            if seq.get("logprobs") is not None:
                assert isinstance(
                    seq["logprobs"], list
                ), f"Sequence {i} logprobs should be a list, got {type(seq['logprobs'])}"

            text = tokenizer.decode(seq["tokens"], skip_special_tokens=True)
            assert len(text.strip()) > 0, f"Sequence {i} decoded to empty text from {len(seq['tokens'])} tokens"
            decoded_texts.append(text)

        unique_texts = set(decoded_texts)
        assert len(unique_texts) > 1, (
            f"All {num_samples} samples at temp=1.0 are identical — sampling params may be ignored. "
            f"Text: {decoded_texts[0][:120]!r}"
        )

        print(f"Call A (temp=1.0): {len(unique_texts)}/{num_samples} unique samples")
        for i, text in enumerate(decoded_texts):
            print(f"  Sample {i}: {text[:100]}..." if len(text) > 100 else f"  Sample {i}: {text}")

        # --- Call B: temp=0.0, expect deterministic outputs ---
        det_sequences = asyncio.run(run_samples(0.0, num_samples))

        det_token_seqs = []
        det_texts = []
        for i, seq in enumerate(det_sequences):
            assert seq["stop_reason"] in ("stop", "length"), f"Unexpected stop_reason: {seq['stop_reason']}"
            assert isinstance(
                seq["tokens"], list
            ), f"Deterministic sequence {i} tokens should be a list, got {type(seq['tokens'])}"
            assert len(seq["tokens"]) > 0, f"Deterministic sequence {i} has no tokens"
            det_token_seqs.append(tuple(seq["tokens"]))

            text = tokenizer.decode(seq["tokens"], skip_special_tokens=True)
            assert len(text.strip()) > 0, f"Deterministic sequence {i} decoded to empty text"
            det_texts.append(text)

        unique_det = set(det_token_seqs)
        assert len(unique_det) == 1, (
            f"temp=0.0 produced {len(unique_det)} distinct token sequences — expected deterministic output. "
            f"Lengths: {[len(s) for s in det_token_seqs]}"
        )

        print(f"Call B (temp=0.0): all {num_samples} samples identical (deterministic)")
        print(f"  Text: {det_texts[0][:120]}")
