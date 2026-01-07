from cloudpathlib import AnyPath
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, SQLModel

from tx.tinker.engine import TinkerEngine
from tx.tinker.config import EngineConfig
from tx.tinker import api
from tx.tinker import types
from tx.tinker.db_models import SessionDB, ModelDB


BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def test_sample_max_num_sequences():
    """
    Verify sampling with sample_max_num_sequences constraint.
    """
    cfg = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        backend_config={
            "max_lora_adapters": 2,
            "max_lora_rank": 32,
            "sample_max_num_sequences": 2,  # Set max sample batch size to 2
        },
    )
    engine = TinkerEngine(cfg)

    # Five prompts, resulting in 3 batches (2 of size 2, 1 of size 1)
    prompts = [
        [1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17],
    ]

    sampling_params = api.SamplingParams(temperature=0.0, max_tokens=16, seed=42).to_types()

    def make_sample_input(tokens: list[int]) -> types.SampleInput:
        return types.SampleInput(
            base_model=BASE_MODEL,  # Sample from base model (no LoRA)
            prompt=types.ModelInput(chunks=[types.ModelInputChunk(tokens=tokens)]),
            sampling_params=sampling_params,
            num_samples=1,
            checkpoint_id="",  # Empty for base model sampling
            prompt_logprobs=False,
        )

    # Build a batch of 5 sample requests
    reqs = {str(request_id): ("", make_sample_input(tokens)) for request_id, tokens in enumerate(prompts)}

    # Process sample requests.
    results = engine.process_sample(reqs)

    # Verify results
    assert len(results) == len(prompts), f"Expected {len(prompts)} results, got {len(results)}"
    for request_id in reqs:
        result = results[request_id]

        assert len(result.sequences) == 1, f"Request {request_id}: expected 1 sequence, got {len(result.sequences)}"
        seq = result.sequences[0]
        tokens = seq.tokens

        # Should have generated some tokens (max_tokens=16)
        assert len(tokens) > 0, f"Request {request_id}: no tokens generated"
        assert len(tokens) <= 16, f"Request {request_id}: generated {len(tokens)} tokens, max was 16"

        # Stop reason should be valid
        assert seq.stop_reason in ["length", "stop"], f"Request {request_id}: invalid stop_reason '{seq.stop_reason}'"

        # If we have logprobs, they should match the number of tokens
        if seq.logprobs:
            assert len(seq.logprobs) == len(
                tokens
            ), f"Request {request_id}: {len(tokens)} tokens but {len(seq.logprobs)} logprobs"


def test_sample_with_prompt_logprobs():
    """Test correct handling of prompt_logprobs in sampling requests."""
    cfg = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        backend_config={"max_lora_adapters": 2, "max_lora_rank": 32},
    )
    engine = TinkerEngine(cfg)

    prompts = [
        [1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12],
    ]

    sampling_params = api.SamplingParams(temperature=0.0, max_tokens=8, seed=42).to_types()

    # Test with prompt_logprobs enabled
    reqs_with_logprobs = {
        f"req_{i}": (
            "",
            types.SampleInput(
                base_model=BASE_MODEL,
                prompt=types.ModelInput(chunks=[types.ModelInputChunk(tokens=tokens)]),
                sampling_params=sampling_params,
                num_samples=1,
                checkpoint_id="",
                prompt_logprobs=True,
            ),
        )
        for i, tokens in enumerate(prompts)
    }

    results_with = engine.process_sample(reqs_with_logprobs)

    for i, tokens in enumerate(prompts):
        request_id = f"req_{i}"
        result = results_with[request_id]

        # Verify prompt_logprobs are returned
        assert result.prompt_logprobs is not None, f"Request {request_id}: prompt_logprobs should not be None"
        # Prompt logprobs should have length = prompt_length - 1
        expected_length = len(tokens) - 1
        assert (
            len(result.prompt_logprobs) == expected_length
        ), f"Request {request_id}: expected {expected_length} prompt_logprobs, got {len(result.prompt_logprobs)}"

    # Test mixed batch: one request with prompt_logprobs=True and one with =False
    reqs_mixed = {
        "req_with_0": (
            "",
            types.SampleInput(
                base_model=BASE_MODEL,
                prompt=types.ModelInput(chunks=[types.ModelInputChunk(tokens=prompts[0])]),
                sampling_params=sampling_params,
                num_samples=1,
                checkpoint_id="",
                prompt_logprobs=True,
            ),
        ),
        "req_without_1": (
            "",
            types.SampleInput(
                base_model=BASE_MODEL,
                prompt=types.ModelInput(chunks=[types.ModelInputChunk(tokens=prompts[1])]),
                sampling_params=sampling_params,
                num_samples=1,
                checkpoint_id="",
                prompt_logprobs=False,
            ),
        ),
    }

    results_mixed = engine.process_sample(reqs_mixed)

    # Verify request with prompt_logprobs=True has logprobs
    assert results_mixed["req_with_0"].prompt_logprobs is not None
    assert len(results_mixed["req_with_0"].prompt_logprobs) == len(prompts[0]) - 1

    # Verify request with prompt_logprobs=False has None
    assert results_mixed["req_without_1"].prompt_logprobs is None


def test_sample_prompt_logprobs_with_microbatching():
    """Test that prompt_logprobs work correctly with micro-batching."""
    cfg = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        backend_config={
            "max_lora_adapters": 2,
            "max_lora_rank": 32,
            "sample_max_num_sequences": 2,  # Force micro-batching with batch size of 2
        },
    )
    engine = TinkerEngine(cfg)

    # Create 5 prompts, which will be split into 3 micro-batches (2, 2, 1)
    prompts = [
        [1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10],
        [11, 12, 13, 14],
        [15, 16],
    ]

    sampling_params = api.SamplingParams(temperature=0.0, max_tokens=8, seed=42).to_types()

    # All requests ask for prompt_logprobs
    reqs = {
        f"req_{i}": (
            "",
            types.SampleInput(
                base_model=BASE_MODEL,
                prompt=types.ModelInput(chunks=[types.ModelInputChunk(tokens=tokens)]),
                sampling_params=sampling_params,
                num_samples=1,
                checkpoint_id="",
                prompt_logprobs=True,
            ),
        )
        for i, tokens in enumerate(prompts)
    }

    results = engine.process_sample(reqs)

    # Verify that each request got its correct prompt_logprobs
    for i, tokens in enumerate(prompts):
        request_id = f"req_{i}"
        result = results[request_id]

        # Verify prompt_logprobs are returned
        assert result.prompt_logprobs is not None, f"Request {request_id}: prompt_logprobs should not be None"

        # Verify correct length
        expected_length = len(tokens) - 1
        assert (
            len(result.prompt_logprobs) == expected_length
        ), f"Request {request_id}: expected {expected_length} prompt_logprobs, got {len(result.prompt_logprobs)}"


def test_process_unload_model():
    """Test that process_unload_model removes model from backend."""
    config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        backend_config={"max_lora_adapters": 4, "max_lora_rank": 32},
    )
    engine = TinkerEngine(config)
    SQLModel.metadata.create_all(engine.db_engine)

    model_id = "test_model"
    _ = engine.process_single_request(
        types.RequestType.CREATE_MODEL, model_id, {"lora_config": {"rank": 8, "alpha": 16}}
    )
    assert engine.backend.has_model(model_id)

    result = engine.process_unload_model(model_id, types.UnloadModelInput())
    assert result.status == "unloaded"
    assert not engine.backend.has_model(model_id)


def test_cleanup_stale_sessions():
    """Test that cleanup_stale_sessions unloads models from expired sessions."""
    config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        backend_config={"max_lora_adapters": 4, "max_lora_rank": 32},
        session_timeout_sec=60,
        database_url="sqlite:///:memory:",  # Use in-memory DB for test isolation
    )
    engine = TinkerEngine(config)
    SQLModel.metadata.create_all(engine.db_engine)

    model_id = "stale_model"
    session_id = "stale_session"

    # Create model in backend
    _ = engine.process_single_request(
        types.RequestType.CREATE_MODEL, model_id, {"lora_config": {"rank": 8, "alpha": 16}}
    )
    assert engine.backend.has_model(model_id)

    # Insert stale session and model into DB
    stale_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=120)
    with Session(engine.db_engine) as session:
        session.add(
            SessionDB(
                session_id=session_id,
                sdk_version="test",
                status="active",
                last_heartbeat_at=stale_heartbeat,
            )
        )
        session.add(
            ModelDB(
                model_id=model_id,
                base_model=BASE_MODEL,
                lora_config=types.LoraConfig(rank=8, alpha=16).model_dump(),
                status="ready",
                request_id=1,
                session_id=session_id,
            )
        )
        session.commit()

    # Run cleanup and assert one model was unloaded
    assert engine.cleanup_stale_sessions() == 1
    assert not engine.backend.has_model(model_id)
