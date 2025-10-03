"""Tests for the Tinker API mock server using the real tinker client."""
import pytest
import subprocess
import tinker
from tinker import types


@pytest.fixture(scope="module")
def api_server():
    """Start the FastAPI server for testing."""
    process = subprocess.Popen(
        ["uv", "run", "--extra", "tinker", "uvicorn", "tx.tinker.api:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
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
    assert "Qwen/Qwen3-0.6B" in model_names


def test_training_workflow(service_client):
    """Test a complete training workflow."""
    base_model = "Qwen/Qwen3-0.6B"
    training_client = service_client.create_lora_training_client(
        base_model=base_model
    )

    tokenizer = training_client.get_tokenizer()

    # Create training examples
    examples = [
        {"prompt": "Question: What is 2+2?\nAnswer:", "completion": " 4"},
        {"prompt": "Question: What color is the sky?\nAnswer:", "completion": " Blue"},
    ]

    # Process examples into Datum objects
    processed_examples = []
    for example in examples:
        prompt_tokens = tokenizer.encode(example["prompt"])
        completion_tokens = tokenizer.encode(example["completion"])

        # Combine tokens
        all_tokens = prompt_tokens + completion_tokens

        # Create weights: 0 for prompt, 1 for completion
        weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

        # Target tokens are shifted by 1
        target_tokens = all_tokens[1:] + [tokenizer.eos_token_id]

        # Create Datum
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(all_tokens[:-1]),
            loss_fn_inputs={
                "weights": weights[:-1],
                "target_tokens": target_tokens[:-1],
            }
        )
        processed_examples.append(datum)

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

    # Get a checkpoint
    sampling_path = training_client.save_weights_for_sampler(name="0000").result().path
    assert sampling_path is not None
