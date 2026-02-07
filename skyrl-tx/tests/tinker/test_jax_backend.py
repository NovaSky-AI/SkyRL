"""Unit tests for JaxBackend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import optax

from tx.tinker.backends.jax import JaxBackend, JaxBackendConfig
from tx.tinker.engine import prepare_model_pass_batch, prepare_sample_batch
from tx.tinker.types import LoraConfig, OptimStepInput
from tx.tinker import api
from tx.tinker import types
from tx.layers.lora import LoRALinear


BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
MAX_LORA_ADAPTERS = 4
LORA_RANK = 8


def create_backend(max_lora_adapters: int = MAX_LORA_ADAPTERS):
    """Create a JaxBackend."""
    config = JaxBackendConfig(max_lora_adapters=max_lora_adapters, max_lora_rank=32)
    return JaxBackend(BASE_MODEL, config)


def create_model(backend: JaxBackend, model_id: str) -> int:
    """Create a model and return its adapter index."""
    lora_config = LoraConfig(rank=LORA_RANK, alpha=16, seed=0)
    backend.create_model(model_id, lora_config)
    return backend.models[model_id].adapter_index


def test_delete_model_basic():
    """Test basic model deletion."""
    backend = create_backend()
    model_id = "test_model"

    # Create model
    _ = create_model(backend, model_id)
    assert backend.has_model(model_id)

    # Delete model
    backend.delete_model(model_id)
    assert not backend.has_model(model_id)


def test_delete_non_existent_model():
    """Test deleting a non-existent model raises ValueError."""
    backend = create_backend()
    with pytest.raises(ValueError, match="not found"):
        backend.delete_model("nonexistent_model")


def test_adapter_slot_reuse():
    """Test that deleted adapter slots are reused."""
    backend = create_backend()

    # Create 3 models and check adapter indices
    assert create_model(backend, "model_1") == 1
    assert create_model(backend, "model_2") == 2
    assert create_model(backend, "model_3") == 3

    # Delete first model, new model should reuse index 1
    backend.delete_model("model_1")
    assert create_model(backend, "model_4") == 1

    # Delete middle model, new model should fill gap at index 1
    backend.delete_model("model_2")
    assert create_model(backend, "model_5") == 2


def test_max_adapters_limit():
    """Test that creating more than available adapters raises ValueError."""
    backend = create_backend()

    # Index 0 is reserved for base model, so we have max_lora_adapters - 1 slots
    num_available = MAX_LORA_ADAPTERS - 1
    for i in range(num_available):
        _ = create_model(backend, f"model_{i}")

    # Try to create one more - should fail
    with pytest.raises(ValueError, match="Maximum number of LoRA adapters"):
        _ = create_model(backend, "model_overflow")


def test_max_adapters_after_delete():
    """Test that deleting a model frees a slot for new models."""
    backend = create_backend()
    # Index 0 is reserved for base model, so we have max_lora_adapters - 1 slots
    num_available = MAX_LORA_ADAPTERS - 1
    for i in range(num_available):
        _ = create_model(backend, f"model_{i}")

    # Delete one model
    backend.delete_model("model_0")

    # Now we should be able to create a new model which should reuse the freed slot
    assert create_model(backend, "model_new") == 1


def test_clear_lora_adapter():
    """Test that clear_lora_adapter zeros out adapter state."""
    backend = create_backend()
    model_id = "test_model"
    adapter_idx = create_model(backend, model_id)

    # Verify adapter has non-zero rank after creation
    model = backend.model
    lora_layer: LoRALinear = model.model.layers[0].self_attn.q_proj
    assert lora_layer.lora_ranks[adapter_idx] > 0

    # Delete the model (calls clear_lora_adapter internally)
    backend.delete_model(model_id)

    # Verify adapter state is zeroed
    assert lora_layer.lora_ranks[adapter_idx] == 0
    assert lora_layer.lora_scaling[adapter_idx] == 0.0
    assert (lora_layer.lora_A[adapter_idx] == 0.0).all()
    assert (lora_layer.lora_B[adapter_idx] == 0.0).all()


def make_fwd_bwd_input(token_lists: list[list[int]]) -> types.ForwardBackwardInput:
    """Build a ForwardBackwardInput for testing."""
    samples = []
    for tokens in token_lists:
        targets = tokens[1:] + [0]
        weights = [1] * len(tokens)
        samples.append(
            types.Datum(
                model_input=types.ModelInput(chunks=[types.ModelInputChunk(tokens=tokens)]),
                loss_fn_inputs=types.LossFnInputs(
                    target_tokens=types.TensorData(data=targets),
                    weights=types.TensorData(data=weights),
                    advantages=types.TensorData(data=[]),
                    logprobs=types.TensorData(data=[]),
                ),
            )
        )
    return types.ForwardBackwardInput(data=samples, loss_fn="cross_entropy")


def _assert_tree_allclose(t1, t2, rtol=1e-3, atol=1e-3, min_match_pct=99.0):
    """Assert that at least min_match_pct% of elements in two trees are close."""
    leaves1 = jax.tree.leaves(t1)
    leaves2 = jax.tree.leaves(t2)
    assert len(leaves1) == len(leaves2), "Gradient trees differ in structure/leaf count"
    for a, b in zip(leaves1, leaves2):
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)

        # Check how many elements are close
        matches = np.isclose(a_arr, b_arr, rtol=rtol, atol=atol)
        match_pct = 100.0 * np.sum(matches) / a_arr.size
        if match_pct < min_match_pct:
            # Show statistics about mismatches
            diff = np.abs(a_arr - b_arr)
            rel_diff = np.abs((a_arr - b_arr) / (np.abs(b_arr) + 1e-10))
            failing = ~matches
            raise AssertionError(
                f"Only {match_pct:.2f}% of elements match (required: {min_match_pct}%)\n"
                f"  Max absolute diff: {np.max(diff[failing])}\n"
                f"  Max relative diff: {np.max(rel_diff[failing])}\n"
                f"  Mean of mismatches: {np.mean(diff[failing])}"
            )


def test_adapter_gradient_calculation():
    """Test that gradients for one adapter are not affected by another adapter's batch size."""
    config = JaxBackendConfig(max_lora_adapters=8, max_lora_rank=32)
    backend = JaxBackend(BASE_MODEL, config)

    adapter1_id = "adapter1"
    adapter2_id = "adapter2"
    backend.create_model(adapter1_id, LoraConfig(rank=32, alpha=32, seed=0))
    backend.create_model(adapter2_id, LoraConfig(rank=32, alpha=32, seed=0))

    # Adapter1 samples (fixed across both rounds)
    a1_input = make_fwd_bwd_input([[1, 2, 3, 4], [5, 6, 7, 8]])
    # Adapter2 samples (round 1: 2 samples; round 2: 4 samples)
    a2_input1 = make_fwd_bwd_input(
        [
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
    )
    reqs_round1 = {
        "101": (adapter1_id, a1_input),
        "102": (adapter2_id, a2_input1),
    }

    # Process round 1 batch
    backend.forward_backward(prepare_model_pass_batch(reqs_round1))

    adapter1_idx = backend.models[adapter1_id].adapter_index
    adapter2_idx = backend.models[adapter2_id].adapter_index

    # Extract gradients for adapter 1
    grads_A1_round1 = jax.tree.map(lambda x: x[adapter1_idx], backend.accumulated_grads.grad_sum)

    # Clear stored grads so we can run another fwd/bwd without optimizer update.
    backend.accumulated_grads = backend.accumulated_grads.reset_adapter(adapter1_idx)
    backend.accumulated_grads = backend.accumulated_grads.reset_adapter(adapter2_idx)

    a1_input = make_fwd_bwd_input([[1, 2, 3, 4], [5, 6, 7, 8]])
    a2_input2 = make_fwd_bwd_input([[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]])
    reqs_round2 = {
        "201": (adapter1_id, a1_input),
        "202": (adapter2_id, a2_input2),
    }

    # Process round 2 batch
    backend.forward_backward(prepare_model_pass_batch(reqs_round2))

    grads_A1_round2 = jax.tree.map(lambda x: x[adapter1_idx], backend.accumulated_grads.grad_sum)

    # Compare gradients using 99% match threshold
    _assert_tree_allclose(grads_A1_round1, grads_A1_round2, rtol=1e-3, atol=1e-2, min_match_pct=99.0)


def test_micro_batch_grad_accumulation():
    """
    Verifies that fwd-bwd with micro-batching produces the same
    per-adapter mean gradients as without micro-batching.
    """
    adapter1_id = "adapter1"
    adapter2_id = "adapter2"

    # Build backend with micro-batching enabled (batch size 4)
    config_micro = JaxBackendConfig(max_lora_adapters=8, max_lora_rank=32, train_micro_batch_size=4)
    backend_micro = JaxBackend(BASE_MODEL, config_micro)
    backend_micro.create_model(adapter1_id, LoraConfig(rank=32, alpha=32, seed=0))
    backend_micro.create_model(adapter2_id, LoraConfig(rank=32, alpha=32, seed=0))

    # Fused batch with 6 total examples: 2 for adapter1, 4 for adapter2.
    a1_input = make_fwd_bwd_input([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2 samples
    a2_input = make_fwd_bwd_input(
        [
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
        ]
    )

    reqs = {
        "1001": (adapter1_id, a1_input),
        "1002": (adapter2_id, a2_input),
    }

    # Run 1: micro-batching enabled
    backend_micro.forward_backward(prepare_model_pass_batch(reqs))

    adapter1_idx = backend_micro.models[adapter1_id].adapter_index
    adapter2_idx = backend_micro.models[adapter2_id].adapter_index

    mean_micro_a1 = backend_micro.accumulated_grads.get_mean(adapter1_idx)
    mean_micro_a2 = backend_micro.accumulated_grads.get_mean(adapter2_idx)

    # Sanity check gradient sum denominators with micro-batching
    assert backend_micro.accumulated_grads.counts[adapter1_idx] == 2
    assert backend_micro.accumulated_grads.counts[adapter2_idx] == 4

    # Build a second backend without micro-batching
    config_full = JaxBackendConfig(max_lora_adapters=8, max_lora_rank=32, train_micro_batch_size=0)
    backend_full = JaxBackend(BASE_MODEL, config_full)
    backend_full.create_model(adapter1_id, LoraConfig(rank=32, alpha=32, seed=0))
    backend_full.create_model(adapter2_id, LoraConfig(rank=32, alpha=32, seed=0))

    # Run 2: micro-batching disabled
    backend_full.forward_backward(prepare_model_pass_batch(reqs))

    # Note: adapter indices might be different in new backend instance if logic changed,
    # but here we create them in same order so it should be fine.
    # Better to fetch them again to be safe.
    adapter1_idx_full = backend_full.models[adapter1_id].adapter_index
    adapter2_idx_full = backend_full.models[adapter2_id].adapter_index

    mean_full_a1 = backend_full.accumulated_grads.get_mean(adapter1_idx_full)
    mean_full_a2 = backend_full.accumulated_grads.get_mean(adapter2_idx_full)

    # Sanity check gradient sum denominators without micro-batching
    assert backend_full.accumulated_grads.counts[adapter1_idx_full] == 2
    assert backend_full.accumulated_grads.counts[adapter2_idx_full] == 4

    # Compare MEAN gradients with and without micro-batching
    _assert_tree_allclose(mean_micro_a1, mean_full_a1, rtol=1e-3, atol=5e-3)
    _assert_tree_allclose(mean_micro_a2, mean_full_a2, rtol=1e-3, atol=5e-3)


def test_process_optim_step_hyperparams_behavior():
    """Request-scoped overrides apply for the step, base hyperparameters stay unchanged, and update size shifts."""
    config = JaxBackendConfig(max_lora_adapters=8, max_lora_rank=32)
    backend = JaxBackend(BASE_MODEL, config)

    low_adapter = "adapter_low"
    default_adapter = "adapter_default"
    for model_id in (low_adapter, default_adapter):
        backend.create_model(model_id, LoraConfig(rank=32, alpha=32, seed=0))

    tokens = [[1, 2, 3, 4], [5, 6, 7, 8]]

    def apply_step(request_id: int, model_id: str, optim_input: OptimStepInput) -> float:
        reqs = {str(request_id): (model_id, make_fwd_bwd_input(tokens))}
        backend.forward_backward(prepare_model_pass_batch(reqs))
        params_before = jax.tree.map(jnp.copy, backend.lora_params)
        backend.optim_step(model_id, optim_input)
        delta = jax.tree.map(
            lambda old, new: (new - old).astype(jnp.float32),
            params_before,
            backend.lora_params,
        )
        return float(optax.global_norm(delta))

    tiny_input = OptimStepInput(
        adam_params=types.AdamParams(learning_rate=1e-8, beta1=1e-8, beta2=1e-8, eps=1e-9, weight_decay=0.0)
    )
    default_input = OptimStepInput(adam_params=api.AdamParams().to_types())
    # Apply override step on the first adapter.
    tiny_norm = apply_step(1, low_adapter, tiny_input)

    # Apply fallback/default step on the second adapter (same engine).
    default_norm = apply_step(2, default_adapter, default_input)

    # Expect a large gap in update magnitude between the two adapters.
    assert tiny_norm > 0
    assert default_norm / tiny_norm == pytest.approx(1e4, rel=5e-3)


def test_gradient_checkpointing():
    """
    Verify gradient checkpointing doesn't affect loss values.
    """
    losses = []
    for use_gradient_checkpointing in (False, True):
        config = JaxBackendConfig(
            max_lora_adapters=1,
            max_lora_rank=4,
            train_micro_batch_size=1,
            gradient_checkpointing=use_gradient_checkpointing,
        )
        backend = JaxBackend(BASE_MODEL, config)

        # Create batch
        B, T = 2, 8
        vocab = backend.model.config.vocab_size
        input_ids = jnp.arange(B * T, dtype=jnp.int32).reshape(B, T) % vocab
        attention_mask = jnp.ones((B, T), dtype=jnp.int32)
        adapter_indices = jnp.zeros((B,), dtype=jnp.int32)
        target_ids = input_ids
        loss_mask = jnp.ones((B, T), dtype=jnp.float32)
        loss_fn_types = jnp.zeros((B,), dtype=jnp.int32)
        sampling_logprobs = jnp.zeros((B, T), dtype=jnp.float32)
        advantages = jnp.zeros((B, T), dtype=jnp.float32)

        # Compute loss, using gradient checkpointing if enabled
        _, per_token_losses, _ = backend._forward_backward_and_accumulate(
            backend.accumulated_grads,
            backend.lora_params,
            backend.non_lora_params,
            input_ids,
            attention_mask,
            adapter_indices,
            target_ids,
            loss_mask,
            loss_fn_types,
            sampling_logprobs,
            advantages,
        )
        losses.append(float(per_token_losses.mean()))

    # Check relative difference between losses is small
    assert abs(losses[0] - losses[1]) / abs(losses[0]) < 5e-3


def make_sample_input(tokens: list[int], prompt_logprobs: bool = False, max_tokens: int = 16) -> types.SampleInput:
    """Build a SampleInput for testing."""
    return types.SampleInput(
        base_model=BASE_MODEL,  # Sample from base model (no LoRA)
        prompt=types.ModelInput(chunks=[types.ModelInputChunk(tokens=tokens)]),
        sampling_params=api.SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=42).to_types(),
        num_samples=1,
        checkpoint_id="",  # Empty for base model sampling
        prompt_logprobs=prompt_logprobs,
    )


def test_sample_max_num_sequences():
    """
    Verify sampling with sample_max_num_sequences constraint.
    """
    config = JaxBackendConfig(
        max_lora_adapters=2,
        max_lora_rank=32,
        sample_max_num_sequences=2,  # Set max sample batch size to 2
    )
    backend = JaxBackend(BASE_MODEL, config)

    # Five prompts, resulting in 3 batches (2 of size 2, 1 of size 1)
    prompts = [
        [1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17],
    ]

    # Build a batch of 5 sample requests
    reqs = {str(request_id): ("", make_sample_input(tokens)) for request_id, tokens in enumerate(prompts)}

    # Process sample requests.
    results = backend.sample(prepare_sample_batch(reqs))

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
    config = JaxBackendConfig(max_lora_adapters=2, max_lora_rank=32)
    backend = JaxBackend(BASE_MODEL, config)

    prompts = [
        [1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12],
    ]

    # Test with prompt_logprobs enabled
    reqs_with_logprobs = {
        f"req_{i}": ("", make_sample_input(tokens, prompt_logprobs=True, max_tokens=8))
        for i, tokens in enumerate(prompts)
    }

    results_with = backend.sample(prepare_sample_batch(reqs_with_logprobs))

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
        "req_with_0": ("", make_sample_input(prompts[0], prompt_logprobs=True, max_tokens=8)),
        "req_without_1": ("", make_sample_input(prompts[1], prompt_logprobs=False, max_tokens=8)),
    }

    results_mixed = backend.sample(prepare_sample_batch(reqs_mixed))

    # Verify request with prompt_logprobs=True has logprobs
    assert results_mixed["req_with_0"].prompt_logprobs is not None
    assert len(results_mixed["req_with_0"].prompt_logprobs) == len(prompts[0]) - 1

    # Verify request with prompt_logprobs=False has None
    assert results_mixed["req_without_1"].prompt_logprobs is None


def test_sample_prompt_logprobs_with_microbatching():
    """Test that prompt_logprobs work correctly with micro-batching."""
    config = JaxBackendConfig(
        max_lora_adapters=2,
        max_lora_rank=32,
        sample_max_num_sequences=2,  # Force micro-batching with batch size of 2
    )
    backend = JaxBackend(BASE_MODEL, config)

    # Create 5 prompts, which will be split into 3 micro-batches (2, 2, 1)
    prompts = [
        [1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10],
        [11, 12, 13, 14],
        [15, 16],
    ]

    # All requests ask for prompt_logprobs
    reqs = {
        f"req_{i}": ("", make_sample_input(tokens, prompt_logprobs=True, max_tokens=8))
        for i, tokens in enumerate(prompts)
    }

    results = backend.sample(prepare_sample_batch(reqs))

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


def test_adapter_reuse_initializes_lora_adapter():
    """Test that reusing an adapter slot initializes the lora adapter properly."""
    # Use max_lora_adapters=2 so only slot 1 is available
    # (slot 0 is reserved for base model)
    backend = create_backend(max_lora_adapters=2)
    model = backend.model
    lora_layer: LoRALinear = model.model.layers[0].self_attn.q_proj

    # Create first model
    model_id_1 = "model_1"
    adapter_idx = create_model(backend, model_id_1)

    # Verify lora_A is non-zero after creation
    assert not (
        lora_layer.lora_A[adapter_idx, ..., :LORA_RANK] == 0.0
    ).all(), "lora_A should be initialized with he_uniform (non-zero)"

    # Delete the model (clears both lora_A and lora_B to zeros)
    backend.delete_model(model_id_1)
    assert (lora_layer.lora_A[adapter_idx] == 0.0).all(), "lora_A should be zeroed after clear_lora_adapter"

    # Create a new model that reuses the same adapter slot
    model_id_2 = "model_2"
    new_adapter_idx = create_model(backend, model_id_2)
    assert new_adapter_idx == adapter_idx, "Should reuse the same adapter slot"

    # Verify lora_A is initialized (non-zero)
    assert not (
        lora_layer.lora_A[adapter_idx, ..., :LORA_RANK] == 0.0
    ).all(), "lora_A should be initialized with he_uniform after adapter reuse"

    # Verify lora_B is zeros
    assert (lora_layer.lora_B[adapter_idx] == 0.0).all(), "lora_B should be zeros"


def test_mixed_train_unembed_adapters():
    """Test that chunked and non-chunked paths produce same results with train_unembed adapters."""

    def create_backend_and_models(loss_chunk_size):
        config = JaxBackendConfig(max_lora_adapters=3, max_lora_rank=32, loss_chunk_size=loss_chunk_size)
        backend = JaxBackend(BASE_MODEL, config)
        backend.create_model("model_normal", LoraConfig(rank=8, alpha=16, seed=0, train_unembed=False))
        backend.create_model("model_unembed", LoraConfig(rank=8, alpha=16, seed=1, train_unembed=True))
        return backend

    def run_forward(backend):
        normal_idx = backend.models["model_normal"].adapter_index
        unembed_idx = backend.models["model_unembed"].adapter_index

        batch_size, seq_len = 2, 16
        vocab = backend.model.config.vocab_size
        input_ids = jnp.arange(batch_size * seq_len, dtype=jnp.int32).reshape(batch_size, seq_len) % vocab
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        target_ids = (input_ids + 1) % vocab
        loss_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
        loss_fn_types = jnp.zeros((batch_size,), dtype=jnp.int32)
        sampling_logprobs = jnp.zeros((batch_size, seq_len), dtype=jnp.float32)
        advantages = jnp.zeros((batch_size, seq_len), dtype=jnp.float32)
        adapter_indices = jnp.array([normal_idx, unembed_idx], dtype=jnp.int32)

        _, losses, logprobs = backend._forward(
            backend.accumulated_grads,
            backend.lora_params,
            backend.non_lora_params,
            input_ids,
            attention_mask,
            adapter_indices,
            target_ids,
            loss_mask,
            loss_fn_types,
            sampling_logprobs,
            advantages,
        )
        return np.asarray(losses), np.asarray(logprobs)

    # Run non-chunked backend first, then delete
    backend = create_backend_and_models(loss_chunk_size=0)
    losses_nonchunked, logprobs_nonchunked = run_forward(backend)
    del backend

    # Run chunked backend
    backend = create_backend_and_models(loss_chunk_size=1024)
    losses_chunked, logprobs_chunked = run_forward(backend)

    np.testing.assert_allclose(
        logprobs_chunked,
        logprobs_nonchunked,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Chunked vs non-chunked logprobs mismatch with mixed train_unembed adapters",
    )
    np.testing.assert_allclose(
        losses_chunked,
        losses_nonchunked,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Chunked vs non-chunked losses mismatch with mixed train_unembed adapters",
    )
