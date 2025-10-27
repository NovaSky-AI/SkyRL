"""Tests for the Tinker API mock server using the real tinker client."""

import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import pytest
import tinker
from tinker import types
from transformers import AutoTokenizer
from sqlalchemy import create_engine, inspect, text


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


def test_database_schema_with_sqlite():
    """Test that SQLModel creates the correct database schema with SQLite."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db_path = Path(tmpdir) / "test_tinker.db"
        test_db_url = f"sqlite:///{test_db_path}"
        
        # Set environment variable for test database
        os.environ["TX_DATABASE_URL"] = test_db_url
        
        # Import models to register them with SQLModel metadata
        from tx.tinker.db_models import ModelDB, FutureDB, CheckpointDB
        from sqlmodel import SQLModel
        
        # Create engine and tables
        engine = create_engine(test_db_url)
        SQLModel.metadata.create_all(engine)
        
        # Verify database was created
        assert test_db_path.exists()
        assert test_db_path.stat().st_size > 0
        
        # Verify tables were created
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        expected_tables = ['models', 'futures', 'checkpoints']
        
        for table in expected_tables:
            assert table in tables, f"Table '{table}' is missing"
            
            # Verify columns exist
            columns = inspector.get_columns(table)
            column_names = [col['name'] for col in columns]
            assert len(column_names) > 0, f"Table '{table}' has no columns"


def test_database_schema_idempotency():
    """Test that running create_all multiple times is idempotent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db_path = Path(tmpdir) / "test_idempotent.db"
        test_db_url = f"sqlite:///{test_db_path}"
        os.environ["TX_DATABASE_URL"] = test_db_url
        
        # Import models to register them with SQLModel metadata
        from sqlmodel import SQLModel
        from tx.tinker.db_models import ModelDB, FutureDB, CheckpointDB
        
        engine = create_engine(test_db_url)
        
        # Create tables multiple times - should not raise errors
        SQLModel.metadata.create_all(engine)
        SQLModel.metadata.create_all(engine)
        SQLModel.metadata.create_all(engine)
        
        # Verify tables still exist and are correct
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        assert 'models' in tables
        assert 'futures' in tables
        assert 'checkpoints' in tables


def test_alembic_migration_generation():
    """Test that Alembic can generate migrations from SQLModel definitions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db_path = Path(tmpdir) / "test_alembic.db"
        test_db_url = f"sqlite:///{test_db_path}"
        
        tinker_dir = Path(__file__).parent.parent.parent / "tx" / "tinker"
        
        # Test: alembic upgrade head creates tables
        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            cwd=tinker_dir,
            capture_output=True,
            text=True,
            env={**os.environ, "TX_DATABASE_URL": test_db_url}
        )
        
        # Should succeed (even if no migrations exist, it shouldn't error)
        assert result.returncode == 0, f"Alembic upgrade failed: {result.stderr}"
        
        # Test: alembic current shows version
        result = subprocess.run(
            ["uv", "run", "alembic", "current"],
            cwd=tinker_dir,
            capture_output=True,
            text=True,
            env={**os.environ, "TX_DATABASE_URL": test_db_url}
        )
        
        assert result.returncode == 0, f"Alembic current failed: {result.stderr}"


def test_alembic_history():
    """Test that Alembic history command works."""
    tinker_dir = Path(__file__).parent.parent.parent / "tx" / "tinker"
    
    # Test: alembic history
    result = subprocess.run(
        ["uv", "run", "alembic", "history"],
        cwd=tinker_dir,
        capture_output=True,
        text=True,
    )
    
    # Should work even with no migrations
    assert result.returncode == 0, f"Alembic history failed: {result.stderr}"
