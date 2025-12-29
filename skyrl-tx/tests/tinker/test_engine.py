from cloudpathlib import AnyPath

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import optax
from tx.tinker.engine import TinkerEngine
from tx.tinker.config import EngineConfig
from tx.tinker import api
from tx.tinker import types


BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def make_fwd_bwd_input(token_lists: list[list[int]]) -> types.ForwardBackwardInput:
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
    config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        max_lora_adapters=8,
        max_lora_rank=32,
    )
    engine = TinkerEngine(config)

    adapter1_id = "adapter1"
    adapter2_id = "adapter2"

    # Create two LoRA adapters
    engine.process_single_request(
        types.RequestType.CREATE_MODEL, adapter1_id, {"lora_config": {"rank": 32, "alpha": 32}}
    )
    engine.process_single_request(
        types.RequestType.CREATE_MODEL, adapter2_id, {"lora_config": {"rank": 32, "alpha": 32}}
    )

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
    engine.process_forward_backward_batch(reqs_round1)

    adapter1_idx = engine.models[adapter1_id].adapter_index
    adapter2_idx = engine.models[adapter2_id].adapter_index

    # Extract gradients for adapter 1
    grads_A1_round1 = jax.tree.map(lambda x: x[adapter1_idx], engine.accumulated_grads.grad_sum)

    # Clear stored grads so we can run another fwd/bwd without optimizer update.
    engine.accumulated_grads = engine.accumulated_grads.reset_adapter(adapter1_idx)
    engine.accumulated_grads = engine.accumulated_grads.reset_adapter(adapter2_idx)

    a1_input = make_fwd_bwd_input([[1, 2, 3, 4], [5, 6, 7, 8]])
    a2_input2 = make_fwd_bwd_input([[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]])
    reqs_round2 = {
        "201": (adapter1_id, a1_input),
        "202": (adapter2_id, a2_input2),
    }

    # Process round 2 batch
    engine.process_forward_backward_batch(reqs_round2)

    grads_A1_round2 = jax.tree.map(lambda x: x[adapter1_idx], engine.accumulated_grads.grad_sum)

    # Compare gradients using 99% match threshold
    _assert_tree_allclose(grads_A1_round1, grads_A1_round2, rtol=1e-3, atol=1e-2, min_match_pct=99.0)


def test_micro_batch_grad_accumulation():
    """
    Verifies that fwd-bwd with micro-batching produces the same
    per-adapter mean gradients as without micro-batching.
    """
    # Build engine and two adapters.
    config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        max_lora_adapters=8,
        max_lora_rank=32,
        train_micro_batch_size=4,
    )
    engine = TinkerEngine(config)

    adapter1_id = "adapter1"
    adapter2_id = "adapter2"

    engine.process_single_request(
        types.RequestType.CREATE_MODEL, adapter1_id, {"lora_config": {"rank": 32, "alpha": 32}}
    )
    engine.process_single_request(
        types.RequestType.CREATE_MODEL, adapter2_id, {"lora_config": {"rank": 32, "alpha": 32}}
    )

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
    engine.process_forward_backward_batch(reqs)

    adapter1_idx = engine.models[adapter1_id].adapter_index
    adapter2_idx = engine.models[adapter2_id].adapter_index

    mean_micro_a1 = engine.accumulated_grads.get_mean(adapter1_idx)
    mean_micro_a2 = engine.accumulated_grads.get_mean(adapter2_idx)

    # Sanity check gradient sum denominators with micro-batching
    assert engine.accumulated_grads.counts[adapter1_idx] == 2
    assert engine.accumulated_grads.counts[adapter2_idx] == 4

    # Build a second engine without micro-batching
    config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        max_lora_adapters=8,
        max_lora_rank=32,
        train_micro_batch_size=0,
    )
    engine = TinkerEngine(config)

    engine.process_single_request(
        types.RequestType.CREATE_MODEL, adapter1_id, {"lora_config": {"rank": 32, "alpha": 32}}
    )
    engine.process_single_request(
        types.RequestType.CREATE_MODEL, adapter2_id, {"lora_config": {"rank": 32, "alpha": 32}}
    )

    # Run 2: micro-batching disabled
    engine.process_forward_backward_batch(reqs)

    # Note: adapter indices might be different in new engine instance if logic changed,
    # but here we create them in same order so it should be fine.
    # Better to fetch them again to be safe.
    adapter1_idx_full = engine.models[adapter1_id].adapter_index
    adapter2_idx_full = engine.models[adapter2_id].adapter_index

    mean_full_a1 = engine.accumulated_grads.get_mean(adapter1_idx_full)
    mean_full_a2 = engine.accumulated_grads.get_mean(adapter2_idx_full)

    # Sanity check gradient sum denominators without micro-batching
    assert engine.accumulated_grads.counts[adapter1_idx_full] == 2
    assert engine.accumulated_grads.counts[adapter2_idx_full] == 4

    # Compare MEAN gradients with and without micro-batching
    _assert_tree_allclose(mean_micro_a1, mean_full_a1, rtol=1e-3, atol=5e-3)
    _assert_tree_allclose(mean_micro_a2, mean_full_a2, rtol=1e-3, atol=5e-3)


def test_process_optim_step_hyperparams_behavior():
    """Request-scoped overrides apply for the step, base hyperparameters stay unchanged, and update size shifts."""
    config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        max_lora_adapters=8,
        max_lora_rank=32,
    )

    engine = TinkerEngine(config)

    low_adapter = "adapter_low"
    default_adapter = "adapter_default"

    for model_id in (low_adapter, default_adapter):
        engine.process_single_request(
            types.RequestType.CREATE_MODEL,
            model_id,
            {"lora_config": {"rank": 32, "alpha": 32}},
        )

    tokens = [[1, 2, 3, 4], [5, 6, 7, 8]]

    def apply_step(request_id: int, model_id: str, request: types.OptimStepInput) -> float:
        engine.process_forward_backward_batch({str(request_id): (model_id, make_fwd_bwd_input(tokens))})
        params_before = jax.tree.map(jnp.copy, engine.lora_params)
        engine.process_optim_step(model_id, request)
        delta = jax.tree.map(lambda old, new: (new - old).astype(jnp.float32), params_before, engine.lora_params)
        return float(optax.global_norm(delta))

    tiny_request = types.OptimStepInput(
        adam_params=types.AdamParams(learning_rate=1e-8, beta1=1e-8, beta2=1e-8, eps=1e-9)
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
        cfg = EngineConfig(
            base_model=BASE_MODEL,
            enforce_eager=False,
            train_batch_size=2,
            train_micro_batch_size=1,
            max_lora_adapters=1,
            max_lora_rank=4,
            gradient_checkpointing=use_gradient_checkpointing,
        )
        engine = TinkerEngine(cfg)

        # Create batch
        B, T = 2, 8
        vocab = engine.model.config.vocab_size
        input_ids = jnp.arange(B * T, dtype=jnp.int32).reshape(B, T) % vocab
        attention_mask = jnp.ones((B, T), dtype=jnp.int32)
        adapter_indices = jnp.zeros((B,), dtype=jnp.int32)
        target_ids = input_ids
        loss_mask = jnp.ones((B, T), dtype=jnp.float32)
        loss_fn_types = jnp.zeros((B,), dtype=jnp.int32)
        sampling_logprobs = jnp.zeros((B, T), dtype=jnp.float32)
        advantages = jnp.zeros((B, T), dtype=jnp.float32)

        # Compute loss, using gradient checkpointing if enabled
        _, per_token_losses, _ = engine._forward_backward_and_accumulate(
            engine.accumulated_grads,
            engine.lora_params,
            engine.non_lora_params,
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


def test_sample_max_num_sequences():
    """
    Verify sampling with sample_max_num_sequences constraint.
    """
    cfg = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        max_lora_adapters=2,
        max_lora_rank=32,
        sample_max_num_sequences=2,  # Set max sample batch size to 2
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
    results = engine.process_sample_batch(reqs)

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
        max_lora_adapters=2,
        max_lora_rank=32,
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

    results_with = engine.process_sample_batch(reqs_with_logprobs)

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

    results_mixed = engine.process_sample_batch(reqs_mixed)

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
        max_lora_adapters=2,
        max_lora_rank=32,
        sample_max_num_sequences=2,  # Force micro-batching with batch size of 2
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

    results = engine.process_sample_batch(reqs)

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
