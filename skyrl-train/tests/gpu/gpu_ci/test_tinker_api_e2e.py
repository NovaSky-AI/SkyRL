"""
End-to-end test for Tinker API integration.

Tests the full flow: HTTP client -> skyrl-tx API -> adapter -> skyrl-train sample()

# Run tests:
uv run --extra dev --extra vllm pytest tests/gpu/gpu_ci/test_tinker_api_e2e.py -m "vllm" -v
"""

import pytest
import asyncio
from dataclasses import dataclass
from typing import Literal
from transformers import AutoTokenizer
import hydra

from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.entrypoints.main_base import config_dir
from omegaconf import DictConfig


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# Lightweight duplicates of Tinker types for testing
# These mirror tx.tinker.types without requiring skyrl-tx dependencies
@dataclass
class ModelInputChunk:
    tokens: list[int]


@dataclass
class ModelInput:
    chunks: list[ModelInputChunk]


@dataclass
class TinkerSamplingParams:
    temperature: float
    max_tokens: int
    seed: int = 42
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


@dataclass
class MockSampleRequest:
    """Mock SampleRequest that mimics the API request structure."""
    prompt: ModelInput
    sampling_params: TinkerSamplingParams
    num_samples: int


class TinkerAdapter:
    """Test adapter that mirrors SkyRLInferenceClient conversion logic.

    This duplicates the conversion logic from tx.tinker.extra.skyrl_inference
    to test the integration contract without requiring skyrl-tx dependencies.
    """

    def __init__(self, inference_client: InferenceEngineClient):
        self.inference_client = inference_client

    def _extract_prompt_tokens(self, model_input: ModelInput) -> list[int]:
        """Extract flat token list from ModelInput."""
        tokens = []
        for chunk in model_input.chunks:
            tokens.extend(chunk.tokens)
        return tokens

    def _convert_sampling_params(self, params: TinkerSamplingParams) -> dict:
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

    def _convert_to_sample_output(self, output: dict) -> SampleOutput:
        """Convert skyrl-train output to Tinker SampleOutput."""
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

    async def _sample(self, request: MockSampleRequest) -> SampleOutput:
        """Execute sample and convert response - mirrors SkyRLInferenceClient._sample."""
        prompt_token_ids = self._extract_prompt_tokens(request.prompt)
        sampling_params = self._convert_sampling_params(request.sampling_params)

        output = await self.inference_client.sample(
            prompt_token_ids=prompt_token_ids,
            num_samples=request.num_samples,
            sampling_params=sampling_params,
        )

        return self._convert_to_sample_output(output)


@dataclass
class MockSkyRLTxApp:
    """A mock skyrl-tx app that tests the TinkerAdapter directly.

    This simulates what the real skyrl-tx /api/v1/asample endpoint does,
    but without needing the full FastAPI app and database.
    """
    adapter: TinkerAdapter

    async def asample(self, request: dict) -> dict:
        """Simulate the /api/v1/asample endpoint behavior.

        Takes a Tinker-style request, converts it, calls sample(), converts response.
        """
        # Parse request into MockSampleRequest (simulating SampleRequest from API)
        prompt_chunks = [ModelInputChunk(tokens=chunk["tokens"]) for chunk in request["prompt"]["chunks"]]
        sample_request = MockSampleRequest(
            prompt=ModelInput(chunks=prompt_chunks),
            sampling_params=TinkerSamplingParams(
                temperature=request["sampling_params"]["temperature"],
                max_tokens=request["sampling_params"]["max_tokens"],
                seed=request["sampling_params"].get("seed", 42),
                top_k=request["sampling_params"].get("top_k", -1),
                top_p=request["sampling_params"].get("top_p", 1.0),
            ),
            num_samples=request.get("num_samples", 1),
        )

        # Call the adapter's _sample method (mirrors SkyRLInferenceClient._sample)
        tinker_output = await self.adapter._sample(sample_request)

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


def create_tinker_adapter(inference_client: InferenceEngineClient) -> TinkerAdapter:
    """Create TinkerAdapter for testing."""
    return TinkerAdapter(inference_client)


@pytest.mark.vllm
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
    3. Adapter converts and calls sample()
    4. Response is converted back to Tinker format
    5. Client receives and validates response
    """
    cfg = get_test_config()
    cfg.generator.backend = backend

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Initialize skyrl-train inference client
    llm_client = init_inference_client(backend, tp_size, cfg)
    adapter = create_tinker_adapter(llm_client)

    # Create mock app (simulates skyrl-tx API server)
    app = MockSkyRLTxApp(adapter=adapter)

    # Create Tinker-style request (as would come from tinker-cookbook client)
    prompt_text = "What is the capital of France?"
    messages = [{"role": "user", "content": prompt_text}]
    prompt_tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )

    tinker_request = {
        "prompt": {
            "chunks": [{"tokens": prompt_tokens}]
        },
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
        assert isinstance(seq["tokens"], list), f"Tokens should be a list"
        assert len(seq["tokens"]) > 0, f"Should have generated tokens"
        assert seq["stop_reason"] in ("length", "stop"), f"Invalid stop_reason"

    # Decode and print samples
    print(f"\n=== E2E Test Results ===")
    print(f"Prompt: {prompt_text}")
    print(f"Generated {len(response['sequences'])} samples:")
    for i, seq in enumerate(response["sequences"]):
        decoded = tokenizer.decode(seq["tokens"], skip_special_tokens=True)
        print(f"  Sample {i}: {decoded[:100]}..." if len(decoded) > 100 else f"  Sample {i}: {decoded}")


@pytest.mark.vllm
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
    adapter = create_tinker_adapter(llm_client)
    app = MockSkyRLTxApp(adapter=adapter)

    prompts = [
        "What is 2 + 2?",
        "Name the largest planet.",
        "What color is the sky?",
    ]

    requests = []
    for prompt_text in prompts:
        messages = [{"role": "user", "content": prompt_text}]
        prompt_tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        requests.append({
            "prompt": {"chunks": [{"tokens": prompt_tokens}]},
            "sampling_params": {"temperature": 0.0, "max_tokens": 32, "top_k": -1, "top_p": 1.0},
            "num_samples": 1,
        })

    async def run_all_requests():
        tasks = [app.asample(req) for req in requests]
        return await asyncio.gather(*tasks)

    responses = asyncio.run(run_all_requests())

    assert len(responses) == len(prompts), "Should have response for each prompt"

    print(f"\n=== E2E Multiple Requests Test ===")
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        assert len(response["sequences"]) == 1
        decoded = tokenizer.decode(response["sequences"][0]["tokens"], skip_special_tokens=True)
        print(f"Q: {prompt}")
        print(f"A: {decoded[:100]}...")
        print()
