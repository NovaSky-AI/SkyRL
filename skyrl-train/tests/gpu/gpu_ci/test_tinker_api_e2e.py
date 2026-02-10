"""
End-to-end test for Tinker API integration.

Tests the full flow: HTTP client → skyrl-tx API → SkyRLInferenceClient → skyrl-train sample()

# Run tests:
uv run --extra dev --extra vllm pytest tests/gpu/gpu_ci/test_tinker_api_e2e.py -m "vllm" -v
"""

import pytest
import asyncio
from dataclasses import dataclass
from transformers import AutoTokenizer

from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.config import SkyRLConfig


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


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


@dataclass
class MockSkyRLTxApp:
    """A mock skyrl-tx app that tests the SkyRLInferenceClient directly.

    This simulates what the real skyrl-tx /api/v1/asample endpoint does,
    but without needing the full FastAPI app and database.
    """

    inference_client: InferenceEngineClient

    async def asample(self, request: dict) -> dict:
        """Simulate the /api/v1/asample endpoint behavior.

        Takes a Tinker-style request, converts it, calls sample(), converts response.
        """
        # Import the conversion functions from our adapter
        # (In real deployment, these would be in skyrl-tx)
        from tests.gpu.gpu_ci.test_tinker_api_integration import (
            ModelInput,
            ModelInputChunk,
            TinkerSamplingParams,
            extract_prompt_tokens,
            convert_sampling_params,
            convert_to_sample_output,
        )

        # Parse request (simulating SampleRequest from API)
        prompt_chunks = [ModelInputChunk(tokens=chunk["tokens"]) for chunk in request["prompt"]["chunks"]]
        tinker_input = ModelInput(chunks=prompt_chunks)

        tinker_params = TinkerSamplingParams(
            temperature=request["sampling_params"]["temperature"],
            max_tokens=request["sampling_params"]["max_tokens"],
            seed=request["sampling_params"].get("seed"),
            top_k=request["sampling_params"].get("top_k", -1),
            top_p=request["sampling_params"].get("top_p", 1.0),
        )

        num_samples = request.get("num_samples", 1)

        # Convert to skyrl-train format (same as SkyRLInferenceClient._sample)
        converted_tokens = extract_prompt_tokens(tinker_input)
        converted_params = convert_sampling_params(tinker_params)

        # Call skyrl-train's sample()
        output = await self.inference_client.sample(
            prompt_token_ids=converted_tokens,
            num_samples=num_samples,
            sampling_params=converted_params,
        )

        # Convert to Tinker format
        tinker_output = convert_to_sample_output(output)

        # Return as dict (simulating JSON response)
        return {
            "sequences": [
                {
                    "tokens": seq.tokens,
                    "logprobs": seq.logprobs,
                    "stop_reason": seq.stop_reason,
                }
                for seq in tinker_output.sequences
            ],
            "prompt_logprobs": tinker_output.prompt_logprobs,
        }


@pytest.mark.parametrize(
    "backend,tp_size",
    [
        pytest.param("vllm", 2, marks=pytest.mark.vllm),
    ],
    ids=["vllm_tp2"],
)
def test_e2e_tinker_sample_flow(ray_init_fixture, backend: str, tp_size: int):
    """End-to-end test of Tinker sampling through skyrl-train.

    This test simulates the full flow:
    1. Client creates Tinker-style request
    2. Request goes through API (simulated)
    3. SkyRLInferenceClient converts and calls sample()
    4. Response is converted back to Tinker format
    5. Client receives and validates response
    """
    cfg = get_test_config()
    cfg.generator.backend = backend

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Initialize skyrl-train inference client
    llm_client = init_inference_client(backend, tp_size, cfg)

    # Create mock app (simulates skyrl-tx API server)
    app = MockSkyRLTxApp(inference_client=llm_client)

    # Create Tinker-style request (as would come from tinker-cookbook client)
    prompt_text = "What is the capital of France?"
    messages = [{"role": "user", "content": prompt_text}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    tinker_request = {
        "prompt": {"chunks": [{"tokens": prompt_tokens}]},
        "sampling_params": {
            "temperature": 0.7,
            "max_tokens": 32,
            "top_k": -1,
            "top_p": 1.0,
        },
        "num_samples": 2,
    }

    # Call the API (simulated)
    async def run_request():
        return await app.asample(tinker_request)

    response = asyncio.run(run_request())

    # Validate response structure matches Tinker SampleOutput
    assert "sequences" in response, "Response should have 'sequences'"
    assert len(response["sequences"]) == 2, "Should have 2 samples"

    for i, seq in enumerate(response["sequences"]):
        assert "tokens" in seq, f"Sequence {i} should have 'tokens'"
        assert "stop_reason" in seq, f"Sequence {i} should have 'stop_reason'"
        assert isinstance(seq["tokens"], list), "Tokens should be a list"
        assert len(seq["tokens"]) > 0, "Should have generated tokens"
        assert seq["stop_reason"] in ("length", "stop"), "Invalid stop_reason"

    # Decode and print samples
    print("\n=== E2E Test Results ===")
    print(f"Prompt: {prompt_text}")
    print(f"Generated {len(response['sequences'])} samples:")
    for i, seq in enumerate(response["sequences"]):
        decoded = tokenizer.decode(seq["tokens"], skip_special_tokens=True)
        print(f"  Sample {i}: {decoded[:100]}..." if len(decoded) > 100 else f"  Sample {i}: {decoded}")


@pytest.mark.parametrize(
    "backend,tp_size",
    [
        pytest.param("vllm", 2, marks=pytest.mark.vllm),
    ],
    ids=["vllm_tp2"],
)
def test_e2e_multiple_requests(ray_init_fixture, backend: str, tp_size: int):
    """Test multiple concurrent Tinker requests through skyrl-train."""
    cfg = get_test_config()
    cfg.generator.backend = backend

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm_client = init_inference_client(backend, tp_size, cfg)
    app = MockSkyRLTxApp(inference_client=llm_client)

    prompts = [
        "What is 2 + 2?",
        "Name the largest planet.",
        "What color is the sky?",
    ]

    requests = []
    for prompt_text in prompts:
        messages = [{"role": "user", "content": prompt_text}]
        prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        requests.append(
            {
                "prompt": {"chunks": [{"tokens": prompt_tokens}]},
                "sampling_params": {"temperature": 0.0, "max_tokens": 32, "top_k": -1, "top_p": 1.0},
                "num_samples": 1,
            }
        )

    async def run_all_requests():
        tasks = [app.asample(req) for req in requests]
        return await asyncio.gather(*tasks)

    responses = asyncio.run(run_all_requests())

    assert len(responses) == len(prompts), "Should have response for each prompt"

    print("\n=== E2E Multiple Requests Test ===")
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        assert len(response["sequences"]) == 1
        decoded = tokenizer.decode(response["sequences"][0]["tokens"], skip_special_tokens=True)
        print(f"Q: {prompt}")
        print(f"A: {decoded[:100]}...")
        print()
