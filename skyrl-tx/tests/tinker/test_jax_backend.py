"""Unit tests for JaxBackend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import optax

from tx.tinker.backends.jax import JaxBackend, JaxBackendConfig
from tx.tinker.engine import prepare_model_pass_batch
from tx.tinker.types import LoraConfig, OptimStepInput, AdamParams
from tx.tinker import types
from tx.layers.lora import LoRALinear


BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
MAX_LORA_ADAPTERS = 4
LORA_RANK = 8


def create_backend():
    """Create a JaxBackend."""
    config = JaxBackendConfig(max_lora_adapters=MAX_LORA_ADAPTERS, max_lora_rank=32)
    return JaxBackend(BASE_MODEL, config)


def create_model(backend: JaxBackend, model_id: str) -> int:
    """Create a model and return its adapter index."""
    lora_config = LoraConfig(rank=LORA_RANK, alpha=16)
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


def test_clear_adapter_config():
    """Test that clear_adapter_config zeros out adapter state."""
    backend = create_backend()
    model_id = "test_model"
    adapter_idx = create_model(backend, model_id)

    # Verify adapter has non-zero rank after creation
    model = backend.model
    lora_layer: LoRALinear = model.model.layers[0].self_attn.q_proj
    assert lora_layer.lora_ranks[adapter_idx] > 0

    # Delete the model (calls clear_adapter_config internally)
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
    backend.create_model(adapter1_id, LoraConfig(rank=32, alpha=32))
    backend.create_model(adapter2_id, LoraConfig(rank=32, alpha=32))

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
    backend_micro.create_model(adapter1_id, LoraConfig(rank=32, alpha=32))
    backend_micro.create_model(adapter2_id, LoraConfig(rank=32, alpha=32))

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
    backend_full.create_model(adapter1_id, LoraConfig(rank=32, alpha=32))
    backend_full.create_model(adapter2_id, LoraConfig(rank=32, alpha=32))

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
        backend.create_model(model_id, LoraConfig(rank=32, alpha=32))

    tokens = [[1, 2, 3, 4], [5, 6, 7, 8]]

    def apply_step(request_id: int, model_id: str, adam_params: AdamParams) -> float:
        reqs = {str(request_id): (model_id, make_fwd_bwd_input(tokens))}
        backend.forward_backward(prepare_model_pass_batch(reqs))
        params_before = jax.tree.map(jnp.copy, backend.lora_params)
        backend.optim_step(model_id, OptimStepInput(adam_params=adam_params))
        delta = jax.tree.map(
            lambda old, new: (new - old).astype(jnp.float32),
            params_before,
            backend.lora_params,
        )
        return float(optax.global_norm(delta))

    tiny_request = types.OptimStepInput(
        adam_params=types.AdamParams(learning_rate=1e-8, beta1=1e-8, beta2=1e-8, eps=1e-9, weight_decay=0.0)
    )
    default_request = types.OptimStepInput(adam_params=api.AdamParams().to_types())
    # Apply override step on the first adapter.
    tiny_norm = apply_step(1, low_adapter, tiny_request)

    # Apply fallback/default step on the second adapter (same engine).
    default_norm = apply_step(2, default_adapter, default_request)

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
