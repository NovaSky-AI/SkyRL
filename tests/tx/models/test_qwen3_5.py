import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from skyrl.tx.models.configs import Qwen3_5Config
from skyrl.tx.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)
from tests.tx.models.conftest import load_model


@pytest.mark.parametrize("tp", [1, 2])
def test_qwen3_5(tp: int):
    if tp > 1 and os.getenv("CI"):
        pytest.skip("TP > 1 currently runs out of memory in the CI")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", attn_implementation="eager", use_safetensors=True, torch_dtype=torch.float32
    )

    inputs = ["The capital of France is", "The most popular programming language is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    with torch.no_grad():
        hf_outputs = hf_model(
            batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True, return_dict=True
        )
    del hf_model

    _, model = load_model(
        "Qwen/Qwen3.5-0.8B",
        Qwen3_5Config,
        Qwen3_5ForCausalLM,
        ("fsdp", "tp"),
        mesh_shape=(1, tp),
        max_lora_adapters=32,
        max_lora_rank=32,
    )

    outputs = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy(), output_hidden_states=True)
    assert outputs.hidden_states is not None
    assert np.allclose(hf_outputs.hidden_states[0].numpy(), outputs.hidden_states[0], rtol=1e-6)
    assert np.allclose(hf_outputs.hidden_states[1].numpy(), outputs.hidden_states[1], rtol=1e-3, atol=1e-3)
    assert np.allclose(hf_outputs.hidden_states[-1].numpy(), outputs.hidden_states[-1], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [32, 64, 100])
@pytest.mark.parametrize("chunk_size", [16, 32])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_chunk_gated_delta_rule_matches_recurrent(batch_size: int, seq_len: int, chunk_size: int, dtype: jnp.dtype):
    """Test that chunk_gated_delta_rule produces the same results as recurrent_gated_delta_rule."""
    num_heads = 4
    k_head_dim = 16
    v_head_dim = 32

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    # Generate random inputs
    query = jax.random.normal(k1, (batch_size, seq_len, num_heads, k_head_dim), dtype=dtype)
    keys = jax.random.normal(k2, (batch_size, seq_len, num_heads, k_head_dim), dtype=dtype)
    value = jax.random.normal(k3, (batch_size, seq_len, num_heads, v_head_dim), dtype=dtype)
    # g should be negative (decay)
    g = -jax.random.uniform(k4, (batch_size, seq_len, num_heads), dtype=dtype, minval=0.01, maxval=0.5)
    # beta should be in [0, 1]
    beta = jax.random.uniform(k5, (batch_size, seq_len, num_heads), dtype=dtype)

    # Test without initial state
    recurrent_out, recurrent_state = recurrent_gated_delta_rule(query, keys, value, g, beta)
    chunk_out, chunk_state = chunk_gated_delta_rule(query, keys, value, g, beta, chunk_size=chunk_size)

    # Use relaxed tolerances for bfloat16 (no float32 upcast in chunked impl)
    rtol = 2e-2 if dtype == jnp.bfloat16 else 1e-4
    atol = 2e-2 if dtype == jnp.bfloat16 else 1e-4

    assert (
        recurrent_out.shape == chunk_out.shape
    ), f"Output shapes don't match: {recurrent_out.shape} vs {chunk_out.shape}"
    assert (
        recurrent_state.shape == chunk_state.shape
    ), f"State shapes don't match: {recurrent_state.shape} vs {chunk_state.shape}"

    np.testing.assert_allclose(
        np.array(recurrent_out),
        np.array(chunk_out),
        rtol=rtol,
        atol=atol,
        err_msg="Outputs don't match between recurrent and chunked implementations",
    )
    np.testing.assert_allclose(
        np.array(recurrent_state),
        np.array(chunk_state),
        rtol=rtol,
        atol=atol,
        err_msg="Final states don't match between recurrent and chunked implementations",
    )

    # Test with initial state
    initial_state = jax.random.normal(k6, (batch_size, num_heads, k_head_dim, v_head_dim), dtype=dtype)

    recurrent_out2, recurrent_state2 = recurrent_gated_delta_rule(
        query, keys, value, g, beta, initial_state=initial_state
    )
    chunk_out2, chunk_state2 = chunk_gated_delta_rule(
        query, keys, value, g, beta, chunk_size=chunk_size, initial_state=initial_state
    )

    np.testing.assert_allclose(
        np.array(recurrent_out2),
        np.array(chunk_out2),
        rtol=rtol,
        atol=atol,
        err_msg="Outputs don't match with initial state",
    )
    np.testing.assert_allclose(
        np.array(recurrent_state2),
        np.array(chunk_state2),
        rtol=rtol,
        atol=atol,
        err_msg="Final states don't match with initial state",
    )
