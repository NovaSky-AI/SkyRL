"""Tests for the Tinker API mock server using the real tinker client."""

import asyncio
import os
import subprocess
import tempfile
import urllib.request
from contextlib import contextmanager
from urllib.parse import urlparse

import pytest
import tinker
from tinker import types
from transformers import AutoTokenizer

from tests.tinker.conftest import wait_for_condition


BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


TEST_SERVER_PORT = 8000

# Configs for the fast cleanup test
TEST_SERVER_PORT_FAST_CLEANUP = 8001
FAST_CLEANUP_INTERVAL_SEC = 1  # How often to check for stale sessions
FAST_CLEANUP_TIMEOUT_SEC = 3  # Seconds without heartbeat before session is stale

TINKER_API_KEY = "tml-dummy"


def verify_training_client(training_client: tinker.TrainingClient):
    """Verify a training client works with a forward pass."""
    tokenizer = training_client.get_tokenizer()
    data = [make_datum(tokenizer, "Hello", " world")]
    result = training_client.forward(data, "cross_entropy").result()
    assert result is not None


def create_service_and_training_client(base_url: str, skip_verify: bool = False):
    """Create a service client and a training client, verifying it works."""
    service_client = tinker.ServiceClient(base_url=base_url, api_key=TINKER_API_KEY)
    training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)
    if not skip_verify:
        verify_training_client(training_client)
    return service_client, training_client


@contextmanager
def start_api_server(overrides: dict[str, str] | None = None):
    """Start the FastAPI server with optional config overrides. Prints log on failure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "server.log")
        db_path = os.path.join(tmp_dir, "server.db")

        with open(log_path, "w") as log_file:
            defaults = {
                "host": "0.0.0.0",
                "port": str(TEST_SERVER_PORT),
                "base-model": BASE_MODEL,
                "backend-config": '{"max_lora_adapters": 4}',
                "database-url": f"sqlite:///{db_path}",
            }
            if overrides:
                defaults.update(overrides)
            cmd = ["uv", "run", "--extra", "tinker", "-m", "tx.tinker.api"]
            for key, value in defaults.items():
                cmd.extend([f"--{key}", value])
            process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            print(f"Starting API server: {' '.join(cmd)}")
            try:
                yield process, log_path
            except Exception:
                with open(log_path) as f:
                    print(f"=== Test failed. Server log ({log_path}) ===\n{f.read()}")
                raise
            finally:
                process.terminate()
                process.wait(timeout=5)


@pytest.fixture(scope="module")
def api_server():
    """Start the FastAPI server for testing."""
    with start_api_server() as server:
        yield server


@pytest.fixture
def service_client(api_server):
    """Create a service client connected to the test server."""
    return tinker.ServiceClient(base_url=f"http://0.0.0.0:{TEST_SERVER_PORT}/", api_key=TINKER_API_KEY)


def make_datum(tokenizer, prompt: str, completion: str, weight: tuple[float, float] | None = (0.0, 1.0)):
    """Helper to create a Datum from prompt and completion with configurable weights."""
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(f"{completion}\n\n", add_special_tokens=False)
    all_tokens = prompt_tokens + completion_tokens
    target_tokens = all_tokens[1:] + [tokenizer.eos_token_id]

    loss_fn_inputs = {"target_tokens": target_tokens}
    if weight is not None:
        prompt_weight, completion_weight = weight
        all_weights = [prompt_weight] * len(prompt_tokens) + [completion_weight] * len(completion_tokens)
        loss_fn_inputs["weights"] = all_weights[1:] + [completion_weight]

    return types.Datum(
        model_input=types.ModelInput.from_ints(all_tokens),
        loss_fn_inputs=loss_fn_inputs,
    )


def test_capabilities(service_client):
    """Test the get_server_capabilities endpoint."""
    capabilities = service_client.get_server_capabilities()
    model_names = [item.model_name for item in capabilities.supported_models]
    assert BASE_MODEL in model_names


def custom_cross_entropy_loss(data, model_logprobs):
    total_loss = 0.0
    for datum, log_p in zip(data, model_logprobs):
        weights = datum.loss_fn_inputs.get("weights")
        weights = weights.to_torch() if weights else 1.0
        total_loss += -(log_p * weights).sum()
    return total_loss, {}


def test_training_workflow(service_client):
    """Test a complete training workflow."""
    training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)

    tokenizer = training_client.get_tokenizer()

    # Create training examples
    processed_examples = [
        make_datum(tokenizer, "Question: What is 2+2?\nAnswer:", " 4", weight=(0.0, 0.0)),
        make_datum(tokenizer, "Question: What color is the sky?\nAnswer:", " Blue"),
        make_datum(tokenizer, "Question: What is 3+3?\nAnswer:", " 6", weight=None),
    ]

    # Save the optimizer state
    resume_path = training_client.save_state(name="0000").result().path
    # Make sure if we save the sampler weights it will not override training weights
    training_client.save_weights_for_sampler(name="0000").result()
    # Get the training run ID from the first save
    parsed_resume = urlparse(resume_path)
    original_training_run_id = parsed_resume.netloc

    # Run training step
    fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")
    optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

    # Get results
    fwdbwd_result = fwdbwd_future.result()
    optim_result = optim_future.result()

    assert fwdbwd_result is not None
    assert optim_result is not None
    assert fwdbwd_result.loss_fn_output_type == "scalar"
    assert len(fwdbwd_result.loss_fn_outputs) == 3

    # The first example has all 0 weights, so all losses should be 0
    assert all(v == 0.0 for v in fwdbwd_result.loss_fn_outputs[0]["elementwise_loss"].data)

    # The second example has default weights (0 for prompt, 1 for completion), so should have non-zero losses
    assert any(v != 0.0 for v in fwdbwd_result.loss_fn_outputs[1]["elementwise_loss"].data)

    # The third example omits weights (auto-filled with 1s), so all losses should be non-zero
    assert all(v != 0.0 for v in fwdbwd_result.loss_fn_outputs[2]["elementwise_loss"].data)

    # Load the optimizer state and verify another forward_backward pass has the same loss
    training_client.load_state(resume_path)
    fwdbwd_result2 = training_client.forward_backward(processed_examples, "cross_entropy").result()
    assert fwdbwd_result2.loss_fn_outputs == fwdbwd_result.loss_fn_outputs
    # Also check that custom loss function produces the same loss
    fwdbwd_result_custom = training_client.forward_backward_custom(
        processed_examples, loss_fn=custom_cross_entropy_loss
    ).result()
    assert fwdbwd_result_custom.loss_fn_outputs == fwdbwd_result.loss_fn_outputs

    # Test that we can restore the training run
    training_client = service_client.create_training_client_from_state(resume_path)
    # Verify the restored client has the same state by running forward_backward again
    fwdbwd_result3 = training_client.forward_backward(processed_examples, "cross_entropy").result()
    assert fwdbwd_result3.loss_fn_outputs == fwdbwd_result.loss_fn_outputs

    sampling_path = training_client.save_weights_for_sampler(name="final").result().path
    parsed = urlparse(sampling_path)
    training_run_id = parsed.netloc
    checkpoint_id = parsed.path.lstrip("/")
    rest_client = service_client.create_rest_client()
    # Download the checkpoint
    checkpoint_response = rest_client.get_checkpoint_archive_url(training_run_id, checkpoint_id).result()
    with tempfile.NamedTemporaryFile() as tmp_archive:
        urllib.request.urlretrieve(checkpoint_response.url, tmp_archive.name)
        assert os.path.getsize(tmp_archive.name) > 0

    # List all checkpoints for the original training run
    checkpoints_response = rest_client.list_checkpoints(original_training_run_id).result()
    assert checkpoints_response is not None
    assert len(checkpoints_response.checkpoints) > 0
    # Verify that the checkpoint we created is in the list
    checkpoint_ids = [ckpt.checkpoint_id for ckpt in checkpoints_response.checkpoints]
    assert "0000" in checkpoint_ids

    # Verify the training run appears in list_training_runs with correct fields
    training_runs = rest_client.list_training_runs().result()
    assert training_runs.cursor.total_count == len(training_runs.training_runs)
    training_run = next(tr for tr in training_runs.training_runs if tr.training_run_id == original_training_run_id)
    assert training_run.base_model == BASE_MODEL
    assert training_run.is_lora is True
    assert training_run.lora_rank == 32
    assert training_run.corrupted is False


@pytest.mark.parametrize("use_lora", [False, True], ids=["base_model", "lora_model"])
def test_sample(service_client, use_lora):
    """Test the sample endpoint with base model or LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if use_lora:
        training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)
        sampling_client = training_client.save_weights_and_get_sampling_client(name="test_sample")
    else:
        sampling_client = service_client.create_sampling_client(base_model=BASE_MODEL)

    # Sample from the model (base or LoRA)
    prompt = types.ModelInput.from_ints(tokenizer.encode("Hello, how are you doing today? ", add_special_tokens=True))
    num_samples_per_request = [1, 2]
    max_tokens_per_request = [20, 10]
    requests = []
    for num_samples, max_tokens in zip(num_samples_per_request, max_tokens_per_request):
        request = sampling_client.sample(
            prompt=prompt,
            sampling_params=types.SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=42),
            num_samples=num_samples,
        )
        requests.append(request)

    # Verify we got the right number of sequences and tokens back
    for request, num_samples, max_tokens in zip(requests, num_samples_per_request, max_tokens_per_request):
        sample_result = request.result()
        assert sample_result is not None
        assert len(sample_result.sequences) == num_samples
        assert len(sample_result.sequences[0].tokens) == max_tokens

    # Test stop tokens: generate once, then use the 5th token as a stop token
    initial_result = sampling_client.sample(
        prompt=prompt,
        sampling_params=types.SamplingParams(temperature=0.0, max_tokens=10, seed=42),
        num_samples=1,
    ).result()

    stop_token = initial_result.sequences[0].tokens[4]
    stopped_result = sampling_client.sample(
        prompt=prompt,
        sampling_params=types.SamplingParams(temperature=0.0, max_tokens=50, seed=42, stop=[stop_token]),
        num_samples=1,
    ).result()

    assert len(stopped_result.sequences[0].tokens) == 5
    assert stopped_result.sequences[0].stop_reason == "stop"
    assert stopped_result.sequences[0].tokens[-1] == stop_token


def test_sample_top_k(service_client):
    """Test that top_k sampling restricts token selection."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    sampling_client = service_client.create_sampling_client(base_model=BASE_MODEL)
    prompt = types.ModelInput.from_ints(tokenizer.encode("Hello, how are you doing today? ", add_special_tokens=True))

    def sample_with_top_k(top_k, num_runs=3):
        return [
            sampling_client.sample(
                prompt=prompt,
                sampling_params=types.SamplingParams(temperature=1.0, max_tokens=5, seed=42 + i, top_k=top_k),
                num_samples=1,
            )
            .result()
            .sequences[0]
            .tokens
            for i in range(num_runs)
        ]

    # top_k=1 should produce identical outputs regardless of seed
    results_top_1 = sample_with_top_k(top_k=1)
    assert all(r == results_top_1[0] for r in results_top_1), "top_k=1 should produce identical outputs"

    # top_k=-1 (disabled) should vary with different seeds
    results_no_top_k = sample_with_top_k(top_k=-1)
    assert not all(seq == results_no_top_k[0] for seq in results_no_top_k), "Without top_k, outputs should vary"


def test_sample_with_stop_strings(service_client):
    """Test the sample endpoint with string stop sequences."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    sampling_client = service_client.create_sampling_client(base_model=BASE_MODEL)

    prompt = types.ModelInput.from_ints(tokenizer.encode("Hello, how are you doing today? ", add_special_tokens=True))

    # Generate a baseline sample without stop strings
    baseline_result = sampling_client.sample(
        prompt=prompt,
        sampling_params=types.SamplingParams(temperature=0.0, max_tokens=50, seed=42),
        num_samples=1,
    ).result()

    baseline_tokens = baseline_result.sequences[0].tokens
    baseline_text = tokenizer.decode(baseline_tokens)

    # Find a substring that appears in the generated text to use as stop string
    # Use a short substring from the middle of the text
    mid_point = len(baseline_text) // 2
    stop_string = baseline_text[mid_point : mid_point + 5]

    # Skip test if we couldn't find a good stop string
    if not stop_string or stop_string not in baseline_text:
        pytest.skip("Could not find a suitable stop string in generated text")

    # Generate with the stop string
    stopped_result = sampling_client.sample(
        prompt=prompt,
        sampling_params=types.SamplingParams(temperature=0.0, max_tokens=50, seed=42, stop=[stop_string]),
        num_samples=1,
    ).result()

    stopped_tokens = stopped_result.sequences[0].tokens
    stopped_text = tokenizer.decode(stopped_tokens)

    # Verify the stop string handling
    assert stopped_result.sequences[0].stop_reason == "stop"
    # The output should be shorter than or equal to baseline
    assert len(stopped_tokens) <= len(baseline_tokens)
    # The stop string should appear at or near the end of the decoded text
    assert stop_string in stopped_text


def test_sample_num_samples_diversity(service_client):
    """Test that num_samples > 1 produces diverse sequences with a fixed seed, and results are reproducible."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    sampling_client = service_client.create_sampling_client(base_model=BASE_MODEL)
    prompt = types.ModelInput.from_ints(tokenizer.encode("Hello, how are you doing today? ", add_special_tokens=True))

    num_samples = 3
    params = types.SamplingParams(temperature=1.0, max_tokens=10, seed=42)

    # Request 1: num_samples > 1 should produce diverse sequences
    result1 = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=num_samples).result()
    assert len(result1.sequences) == num_samples
    tokens1 = [seq.tokens for seq in result1.sequences]
    assert len(set(tuple(t) for t in tokens1)) > 1, "num_samples > 1 with seed should produce diverse sequences"

    # Request 2: same seed should reproduce the exact same sequences
    result2 = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=num_samples).result()
    tokens2 = [seq.tokens for seq in result2.sequences]
    assert tokens1 == tokens2, "Same seed should produce identical results across calls"

    # Request 3: different seed should produce different sequences
    params_different = types.SamplingParams(temperature=1.0, max_tokens=10, seed=999)
    result3 = sampling_client.sample(prompt=prompt, sampling_params=params_different, num_samples=num_samples).result()
    tokens3 = [seq.tokens for seq in result3.sequences]
    assert tokens1 != tokens3, "Different seeds should produce different results"


@pytest.fixture(scope="function")
def api_server_fast_cleanup():
    """Start the FastAPI server with fast cleanup settings for testing."""
    with start_api_server(
        overrides={
            "port": str(TEST_SERVER_PORT_FAST_CLEANUP),
            "session-cleanup-interval-sec": str(FAST_CLEANUP_INTERVAL_SEC),
            "session-timeout-sec": str(FAST_CLEANUP_TIMEOUT_SEC),
        },
    ) as server:
        yield server


def test_unload_model(api_server):
    """Test that unload_model properly unloads a model."""
    # Create a training client (which creates a model) and verify it works
    _, training_client = create_service_and_training_client(base_url=f"http://0.0.0.0:{TEST_SERVER_PORT}/")

    async def unload_model():
        async with tinker._client.AsyncTinker(
            api_key=TINKER_API_KEY, base_url=f"http://0.0.0.0:{TEST_SERVER_PORT}/"
        ) as client:
            future = await client.models.unload(request=types.UnloadModelRequest(model_id=training_client.model_id))
            while True:
                result = await client.futures.retrieve(
                    request=types.FutureRetrieveRequest(request_id=future.request_id)
                )
                if isinstance(result, types.UnloadModelResponse):
                    return result
                await asyncio.sleep(0.1)

    assert isinstance(asyncio.run(unload_model()), types.UnloadModelResponse)

    # Verify model no longer works after unload
    with pytest.raises(Exception):
        verify_training_client(training_client)


def test_stale_session_cleanup(api_server_fast_cleanup):
    """Test that stale sessions are automatically cleaned up and models are unloaded.

    This test only checks server logs for cleanup messages rather than verifying
    adapter slot reuse, since that behavior is already covered by unit tests in
    test_jax_backend.py and test_engine.py.
    """
    _, log_path = api_server_fast_cleanup
    base_url = f"http://0.0.0.0:{TEST_SERVER_PORT_FAST_CLEANUP}/"
    # Skip verification because we've reduced cleanup timeout in this test.
    # When running on a slow machine, the model may have already be cleaned
    # up at verification time.
    service_client, training_client = create_service_and_training_client(
        base_url=base_url,
        skip_verify=True,
    )

    # Stop heartbeating by deleting the clients
    del training_client
    del service_client

    # Poll for cleanup log messages
    def cleanup_logs_found():
        with open(log_path) as f:
            log_output = f.read()
        return "Auto-unloaded stale model" in log_output and "Deleted model" in log_output

    assert wait_for_condition(cleanup_logs_found, timeout_sec=10, poll_interval_sec=1), "Cleanup logs not found"
