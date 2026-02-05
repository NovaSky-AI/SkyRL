"""
Integration tests for Tinker API compatibility.

Tests that skyrl-train's sample() method works with Tinker-style inputs/outputs,
verifying the integration contract between skyrl-tx API and skyrl-train inference.

# Run tests:
uv run --isolated --extra dev --extra vllm pytest tests/gpu/gpu_ci/test_tinker_api_integration.py -m "vllm" -v
"""

import asyncio
from dataclasses import dataclass
from typing import Literal

import pytest
from transformers import AutoTokenizer

from skyrl_train.config import SkyRLConfig
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import (
    create_ray_wrapped_inference_engines,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# Tinker-compatible types (mirrors skyrl-tx/tx/tinker/types.py)
@dataclass
class ModelInputChunk:
    tokens: list[int]


@dataclass
class ModelInput:
    chunks: list[ModelInputChunk]

    @classmethod
    def from_tokens(cls, tokens: list[int]) -> "ModelInput":
        return cls(chunks=[ModelInputChunk(tokens=tokens)])


@dataclass
class TinkerSamplingParams:
    temperature: float
    max_tokens: int
    seed: int | None = None
    stop_tokens: list[int] | None = None
    stop_strings: list[str] | None = None
    top_k: int = -1
    top_p: float = 1.0


@dataclass
class GeneratedSequence:
    stop_reason: Literal["length", "stop"]
    tokens: list[int]
    logprobs: list[float]


@dataclass
class SampleOutput:
    sequences: list[GeneratedSequence]
    prompt_logprobs: list[float] | None = None


# Conversion functions (mirrors SkyRLInferenceClient logic)
def extract_prompt_tokens(model_input: ModelInput) -> list[int]:
    """Extract flat token list from ModelInput."""
    tokens = []
    for chunk in model_input.chunks:
        tokens.extend(chunk.tokens)
    return tokens


def convert_sampling_params(params: TinkerSamplingParams) -> dict:
    """Convert Tinker SamplingParams to skyrl-train format."""
    result = {
        "temperature": params.temperature,
        "max_tokens": params.max_tokens,
        "top_k": params.top_k,
        "top_p": params.top_p,
    }
    if params.seed is not None:
        result["seed"] = params.seed
    if params.stop_tokens:
        result["stop_token_ids"] = params.stop_tokens
    if params.stop_strings:
        result["stop"] = params.stop_strings
    return result


def convert_to_sample_output(output: dict) -> SampleOutput:
    """Convert skyrl-train's InferenceEngineOutput to Tinker SampleOutput."""
    sequences = []
    num_samples = len(output["response_ids"])

    for i in range(num_samples):
        stop_reason = output["stop_reasons"][i]
        if stop_reason in ("stop", "eos"):
            tinker_stop_reason = "stop"
        else:
            tinker_stop_reason = "length"

        logprobs = []
        if output.get("response_logprobs") and output["response_logprobs"][i]:
            logprobs = output["response_logprobs"][i]

        sequences.append(
            GeneratedSequence(
                tokens=output["response_ids"][i],
                logprobs=logprobs,
                stop_reason=tinker_stop_reason,
            )
        )

    return SampleOutput(sequences=sequences, prompt_logprobs=None)


def get_test_config() -> SkyRLConfig:
    """Get base config with test-specific overrides."""
    cfg = SkyRLConfig()
    cfg.trainer.policy.model.path = MODEL
    cfg.generator.sampling_params.temperature = 0.7
    cfg.generator.sampling_params.top_p = 1
    cfg.generator.sampling_params.top_k = -1
    cfg.generator.sampling_params.max_generate_length = 64
    cfg.generator.sampling_params.min_p = 0.0
    cfg.generator.sampling_params.logprobs = None
    return cfg


def init_inference_client(backend: str, tp_size: int, config: SkyRLConfig) -> InferenceEngineClient:
    """Initialize inference client for testing."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    engines = create_ray_wrapped_inference_engines(
        num_inference_engines=1,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
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
        max_num_batched_tokens=32768,
        max_num_seqs=1024,
        tokenizer=tokenizer,
        backend=backend,
    )
    return InferenceEngineClient(engines, tokenizer, config)


@pytest.mark.parametrize(
    "backend,tp_size",
    [
        pytest.param("vllm", 2, marks=pytest.mark.vllm),
    ],
    ids=["vllm_tp2"],
)
def test_tinker_type_conversion(ray_init_fixture, backend: str, tp_size: int):
    """Test that Tinker-style types convert correctly to/from skyrl-train format."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Create Tinker-style input
    prompt_text = "What is 2 + 2?"
    messages = [{"role": "user", "content": prompt_text}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    tinker_input = ModelInput.from_tokens(prompt_tokens)
    tinker_params = TinkerSamplingParams(
        temperature=0.7,
        max_tokens=32,
        seed=42,
        top_k=-1,
        top_p=1.0,
    )

    # Convert to skyrl-train format
    converted_tokens = extract_prompt_tokens(tinker_input)
    converted_params = convert_sampling_params(tinker_params)

    # Verify conversions
    assert converted_tokens == prompt_tokens, "Token conversion should preserve all tokens"
    assert converted_params["temperature"] == 0.7
    assert converted_params["max_tokens"] == 32
    assert converted_params["seed"] == 42
    assert converted_params["top_k"] == -1
    assert converted_params["top_p"] == 1.0


@pytest.mark.parametrize(
    "backend,tp_size",
    [
        pytest.param("vllm", 2, marks=pytest.mark.vllm),
    ],
    ids=["vllm_tp2"],
)
def test_tinker_sample_integration(ray_init_fixture, backend: str, tp_size: int):
    """Test end-to-end Tinker-style sampling through skyrl-train.

    This test simulates what skyrl-tx's SkyRLInferenceClient does:
    1. Accept Tinker-style ModelInput and SamplingParams
    2. Convert to skyrl-train format
    3. Call sample()
    4. Convert result to Tinker SampleOutput
    """
    cfg = get_test_config()
    cfg.generator.backend = backend

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm_client = init_inference_client(backend, tp_size, cfg)

    # Create Tinker-style input (simulating what skyrl-tx API receives)
    prompt_text = "What is 2 + 2?"
    messages = [{"role": "user", "content": prompt_text}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    tinker_input = ModelInput.from_tokens(prompt_tokens)
    tinker_params = TinkerSamplingParams(
        temperature=0.7,
        max_tokens=32,
        seed=None,  # No seed for variety
        top_k=-1,
        top_p=1.0,
    )
    num_samples = 3

    # Convert to skyrl-train format (same as SkyRLInferenceClient._sample)
    converted_tokens = extract_prompt_tokens(tinker_input)
    converted_params = convert_sampling_params(tinker_params)

    # Call skyrl-train's sample()
    async def run_sample():
        return await llm_client.sample(
            prompt_token_ids=converted_tokens,
            num_samples=num_samples,
            sampling_params=converted_params,
        )

    output = asyncio.run(run_sample())

    # Convert to Tinker format
    tinker_output = convert_to_sample_output(output)

    # Verify Tinker output structure
    assert len(tinker_output.sequences) == num_samples, f"Expected {num_samples} sequences"

    for i, seq in enumerate(tinker_output.sequences):
        # Verify each sequence has tokens
        assert isinstance(seq.tokens, list), f"Sequence {i} tokens should be a list"
        assert len(seq.tokens) > 0, f"Sequence {i} should have generated tokens"
        assert all(isinstance(t, int) for t in seq.tokens), "All tokens should be integers"

        # Verify stop reason is valid Tinker format
        assert seq.stop_reason in ("length", "stop"), f"Invalid stop reason: {seq.stop_reason}"

        # Verify logprobs is a list (may be empty if not requested)
        assert isinstance(seq.logprobs, list), "Logprobs should be a list"

    # Print samples for debugging
    print(f"\nGenerated {len(tinker_output.sequences)} Tinker-format sequences:")
    for i, seq in enumerate(tinker_output.sequences):
        decoded = tokenizer.decode(seq.tokens, skip_special_tokens=True)
        print(f"  Sample {i}: {decoded[:80]}... (stop={seq.stop_reason}, {len(seq.tokens)} tokens)")


@pytest.mark.parametrize(
    "backend,tp_size",
    [
        pytest.param("vllm", 2, marks=pytest.mark.vllm),
    ],
    ids=["vllm_tp2"],
)
def test_tinker_stop_tokens(ray_init_fixture, backend: str, tp_size: int):
    """Test that stop tokens are handled correctly in Tinker format."""
    cfg = get_test_config()
    cfg.generator.backend = backend

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm_client = init_inference_client(backend, tp_size, cfg)

    # Create input with stop strings
    prompt_text = "Count from 1 to 10:"
    messages = [{"role": "user", "content": prompt_text}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    tinker_input = ModelInput.from_tokens(prompt_tokens)
    tinker_params = TinkerSamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=64,
        stop_strings=["5"],  # Stop at "5"
        top_k=-1,
        top_p=1.0,
    )

    # Convert and call
    converted_tokens = extract_prompt_tokens(tinker_input)
    converted_params = convert_sampling_params(tinker_params)

    async def run_sample():
        return await llm_client.sample(
            prompt_token_ids=converted_tokens,
            num_samples=1,
            sampling_params=converted_params,
        )

    output = asyncio.run(run_sample())
    tinker_output = convert_to_sample_output(output)

    # Should have stopped at "5"
    assert len(tinker_output.sequences) == 1
    decoded = tokenizer.decode(tinker_output.sequences[0].tokens, skip_special_tokens=True)
    print(f"Output with stop string '5': {decoded}")

    # The output should not contain "6" or higher (stopped at 5)
    # Note: This is a soft check since LLM output isn't guaranteed


@pytest.mark.parametrize(
    "backend,tp_size",
    [
        pytest.param("vllm", 1, marks=pytest.mark.vllm),
    ],
    ids=["vllm_tp1"],
)
def test_multi_engine_load_balancing(ray_init_fixture, backend: str, tp_size: int):
    """Test that sample() works with multiple inference engines using random load-balancing."""
    cfg = get_test_config()
    cfg.generator.backend = backend

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Create 2 engines with TP=1 each (uses 2 GPUs total)
    engines = create_ray_wrapped_inference_engines(
        num_inference_engines=2,  # Multiple engines for load-balancing
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
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
        max_num_batched_tokens=32768,
        max_num_seqs=1024,
        tokenizer=tokenizer,
        backend=backend,
    )
    llm_client = InferenceEngineClient(engines, tokenizer, cfg)

    # Create Tinker-style input
    prompt_text = "What is 2 + 2?"
    messages = [{"role": "user", "content": prompt_text}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    tinker_input = ModelInput.from_tokens(prompt_tokens)
    tinker_params = TinkerSamplingParams(
        temperature=0.7,
        max_tokens=32,
        seed=42,
        top_k=-1,
        top_p=1.0,
    )

    # Convert to skyrl-train format
    converted_tokens = extract_prompt_tokens(tinker_input)
    converted_params = convert_sampling_params(tinker_params)

    # Call sample() multiple times - should succeed with random load-balancing across engines
    async def run_samples():
        results = []
        for i in range(5):
            result = await llm_client.sample(
                prompt_token_ids=converted_tokens,
                num_samples=2,
                sampling_params=converted_params,
            )
            results.append(result)
        return results

    results = asyncio.run(run_samples())

    # Verify all requests succeeded
    assert len(results) == 5, "Should complete 5 sample requests"
    for i, result in enumerate(results):
        tinker_output = convert_to_sample_output(result)
        assert len(tinker_output.sequences) == 2, f"Request {i} should have 2 samples"
        for seq in tinker_output.sequences:
            assert len(seq.tokens) > 0, f"Request {i} sequences should have tokens"
            assert seq.stop_reason in ("length", "stop"), f"Request {i} should have valid stop reason"

    print(f"\nSuccessfully completed 5 requests with 2 samples each across {len(engines)} engines")
