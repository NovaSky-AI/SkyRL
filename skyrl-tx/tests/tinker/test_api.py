"""Tests for the Tinker API mock server using the real tinker client."""

import os
import subprocess
import tempfile
import urllib.request
from urllib.parse import urlparse

import pytest
import tinker
from tinker import types
from transformers import AutoTokenizer


BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


@pytest.fixture(scope="module")
def api_server():
    """Start the FastAPI server for testing."""
    process = subprocess.Popen(
        [
            "uv",
            "run",
            "--extra",
            "tinker",
            "-m",
            "tx.tinker.api",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--base-model",
            BASE_MODEL,
            # Set number of LoRA adapters lower to avoid OOMs in the CI
            "--max-lora-adapters",
            "4",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    yield process

    # Cleanup
    process.terminate()
    process.wait(timeout=5)


@pytest.fixture
def service_client(api_server):
    """Create a service client connected to the test server."""
    return tinker.ServiceClient(base_url="http://0.0.0.0:8000/", api_key="dummy")


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


@pytest.mark.parametrize("use_lora", [False, True], ids=["base_model", "lora_model"])
def test_sample(service_client, use_lora):
    """Test the sample endpoint with base model or LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if use_lora:
        training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)
        sampling_path = training_client.save_weights_for_sampler(name="test_sample").result().path
        sampling_client = service_client.create_sampling_client(sampling_path)
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
