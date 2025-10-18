"""Tests for the Tinker API mock server using the real tinker client."""

import os
import subprocess
import tempfile
import urllib.request
from urllib.parse import urlparse

import pytest
import tinker
from tinker import types


BASE_MODEL = "Qwen/Qwen3-0.6B"


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
            "--enable-dummy-sample",
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


def test_capabilities(service_client):
    """Test the get_server_capabilities endpoint."""
    capabilities = service_client.get_server_capabilities()
    model_names = [item.model_name for item in capabilities.supported_models]
    assert BASE_MODEL in model_names


def test_training_workflow(service_client):
    """Test a complete training workflow."""
    training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)

    tokenizer = training_client.get_tokenizer()

    # Create training examples
    examples = [
        {"prompt": "Question: What is 2+2?\nAnswer:", "completion": " 4"},
        {"prompt": "Question: What color is the sky?\nAnswer:", "completion": " Blue"},
    ]

    # Process examples into Datum objects
    processed_examples = []
    for i, example in enumerate(examples):
        prompt_tokens = tokenizer.encode(example["prompt"], add_special_tokens=True)
        completion_tokens = tokenizer.encode(f'{example["completion"]}\n\n', add_special_tokens=False)

        # Combine tokens
        all_tokens = prompt_tokens + completion_tokens

        if i == 0:
            # First example has all 0 weights
            weights = [0.0] * len(all_tokens)
        else:
            # All other examples have weight of 0 for prompt, 1 for completion
            weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

        # Target tokens are shifted by 1
        target_tokens = all_tokens[1:] + [tokenizer.eos_token_id]

        # Create Datum
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(all_tokens[:-1]),
            loss_fn_inputs={
                "weights": weights[:-1],
                "target_tokens": target_tokens[:-1],
            },
        )
        processed_examples.append(datum)

    # Save the optimizer state
    resume_path = training_client.save_state(name="0000").result().path
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
    assert len(fwdbwd_result.loss_fn_outputs) > 0

    # The first example has all 0 weights, so all losses should be 0
    assert all(v == 0.0 for v in fwdbwd_result.loss_fn_outputs[0]["elementwise_loss"].data)

    # Load the optimizer state and verify another forward_backward pass has the same loss
    training_client.load_state(resume_path)
    fwdbwd_result2 = training_client.forward_backward(processed_examples, "cross_entropy").result()
    assert fwdbwd_result2.loss_fn_outputs == fwdbwd_result.loss_fn_outputs

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


def test_sample(service_client):
    """Test the sample endpoint."""
    # Create a training client and save weights to get a valid model
    training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)
    tokenizer = training_client.get_tokenizer()

    # Save weights to get a valid model path
    save_future = training_client.save_weights_for_sampler(name="test_sample")
    model_path = save_future.result().path

    # Create a sampling client from the saved model path and get a sample
    sampling_client = service_client.create_sampling_client(model_path)
    prompt = types.ModelInput.from_ints(tokenizer.encode("Hello", add_special_tokens=True))
    sample_result = sampling_client.sample(
        prompt=prompt,
        sampling_params=types.SamplingParams(temperature=1.0, top_k=50, max_tokens=10),
        num_samples=1,
    ).result()

    # Verify we got sequences back
    assert sample_result is not None
    assert len(sample_result.sequences) == 1
    assert len(sample_result.sequences[0].tokens) > 0
