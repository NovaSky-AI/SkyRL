"""
Integration tests for Tinker API compatibility.

Tests that skyrl-train's sample() method works with Tinker-style inputs/outputs,
verifying the integration contract between skyrl-tx API and skyrl-train inference.

# Run tests:
uv run --isolated --extra dev --extra vllm pytest tests/gpu/gpu_ci/test_tinker_api_integration.py -m "vllm" -v
"""

import pytest
import asyncio
from unittest.mock import MagicMock
from transformers import AutoTokenizer
import hydra

from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.entrypoints.main_base import config_dir
from omegaconf import DictConfig

# Import actual Tinker types from skyrl-tx
from tx.tinker.types import (
    ModelInput,
    ModelInputChunk,
    SamplingParams as TinkerSamplingParams,
    GeneratedSequence,
    SampleOutput,
)

# Import the actual SkyRLInferenceClient adapter
from tx.tinker.extra.skyrl_inference import SkyRLInferenceClient


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_config() -> DictConfig:
    """Get base config with test-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")
        cfg.trainer.policy.model.path = MODEL
        cfg.generator.sampling_params.temperature = 0.7
        cfg.generator.sampling_params.top_p = 1
        cfg.generator.sampling_params.top_k = -1
        cfg.generator.sampling_params.max_generate_length = 64
        cfg.generator.sampling_params.min_p = 0.0
        cfg.generator.sampling_params.logprobs = None
        return cfg


def init_inference_client(backend: str, tp_size: int, config: DictConfig) -> InferenceEngineClient:
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


def create_skyrl_inference_client(inference_client: InferenceEngineClient) -> SkyRLInferenceClient:
    """Create SkyRLInferenceClient with a mock db_engine for testing."""
    mock_db_engine = MagicMock()
    return SkyRLInferenceClient(inference_client, mock_db_engine)


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

    # Create Tinker-style input using actual types
    prompt_text = "What is 2 + 2?"
    messages = [{"role": "user", "content": prompt_text}]
    prompt_tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )

    tinker_input = ModelInput(chunks=[ModelInputChunk(tokens=prompt_tokens)])
    tinker_params = TinkerSamplingParams(
        temperature=0.7,
        max_tokens=32,
        seed=42,
        top_k=-1,
        top_p=1.0,
    )

    # Create a mock inference client to test conversion methods
    cfg = get_test_config()
    cfg.generator.backend = backend
    llm_client = init_inference_client(backend, tp_size, cfg)
    skyrl_client = create_skyrl_inference_client(llm_client)

    # Test actual conversion methods from SkyRLInferenceClient
    converted_tokens = skyrl_client._extract_prompt_tokens(tinker_input)
    converted_params = skyrl_client._convert_sampling_params(tinker_params)

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

    This test uses the actual SkyRLInferenceClient to verify:
    1. Accept Tinker-style ModelInput and SamplingParams
    2. Convert to skyrl-train format using actual adapter methods
    3. Call sample()
    4. Convert result to Tinker SampleOutput
    """
    cfg = get_test_config()
    cfg.generator.backend = backend

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm_client = init_inference_client(backend, tp_size, cfg)
    skyrl_client = create_skyrl_inference_client(llm_client)

    # Create Tinker-style input using actual types
    prompt_text = "What is 2 + 2?"
    messages = [{"role": "user", "content": prompt_text}]
    prompt_tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )

    tinker_input = ModelInput(chunks=[ModelInputChunk(tokens=prompt_tokens)])
    tinker_params = TinkerSamplingParams(
        temperature=0.7,
        max_tokens=32,
        seed=42,  # Use seed for reproducibility
        top_k=-1,
        top_p=1.0,
    )
    num_samples = 3

    # Convert to skyrl-train format using actual SkyRLInferenceClient methods
    converted_tokens = skyrl_client._extract_prompt_tokens(tinker_input)
    converted_params = skyrl_client._convert_sampling_params(tinker_params)

    # Call skyrl-train's sample()
    async def run_sample():
        return await llm_client.sample(
            prompt_token_ids=converted_tokens,
            num_samples=num_samples,
            sampling_params=converted_params,
        )

    output = asyncio.run(run_sample())

    # Convert to Tinker format using actual SkyRLInferenceClient method
    tinker_output = skyrl_client._convert_to_sample_output(output)

    # Verify Tinker output structure
    assert isinstance(tinker_output, SampleOutput), "Should return SampleOutput type"
    assert len(tinker_output.sequences) == num_samples, f"Expected {num_samples} sequences"

    for i, seq in enumerate(tinker_output.sequences):
        # Verify each sequence has tokens
        assert isinstance(seq, GeneratedSequence), f"Sequence {i} should be GeneratedSequence"
        assert isinstance(seq.tokens, list), f"Sequence {i} tokens should be a list"
        assert len(seq.tokens) > 0, f"Sequence {i} should have generated tokens"
        assert all(isinstance(t, int) for t in seq.tokens), f"All tokens should be integers"

        # Verify stop reason is valid Tinker format
        assert seq.stop_reason in ("length", "stop"), f"Invalid stop reason: {seq.stop_reason}"

        # Verify logprobs is a list (may be empty if not requested)
        assert isinstance(seq.logprobs, list), f"Logprobs should be a list"

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
    skyrl_client = create_skyrl_inference_client(llm_client)

    # Create input with stop strings
    prompt_text = "Count from 1 to 10:"
    messages = [{"role": "user", "content": prompt_text}]
    prompt_tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )

    tinker_input = ModelInput(chunks=[ModelInputChunk(tokens=prompt_tokens)])
    tinker_params = TinkerSamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=64,
        seed=42,
        stop_strings=["5"],  # Stop at "5"
        top_k=-1,
        top_p=1.0,
    )

    # Convert and call using actual SkyRLInferenceClient methods
    converted_tokens = skyrl_client._extract_prompt_tokens(tinker_input)
    converted_params = skyrl_client._convert_sampling_params(tinker_params)

    async def run_sample():
        return await llm_client.sample(
            prompt_token_ids=converted_tokens,
            num_samples=1,
            sampling_params=converted_params,
        )

    output = asyncio.run(run_sample())
    tinker_output = skyrl_client._convert_to_sample_output(output)

    # Should have stopped at "5"
    assert len(tinker_output.sequences) == 1
    decoded = tokenizer.decode(tinker_output.sequences[0].tokens, skip_special_tokens=True)
    print(f"Output with stop string '5': {decoded}")

    # Verify we got a valid stop reason
    assert tinker_output.sequences[0].stop_reason in ("length", "stop"), \
        f"Stop reason should be valid Tinker format, got: {tinker_output.sequences[0].stop_reason}"
