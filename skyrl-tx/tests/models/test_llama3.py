import os
import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tx.models import Llama3ForCausalLM
from tx.utils.models import load_safetensors


@pytest.mark.parametrize("tp", [1, 2])
def test_llama3(tp: int):
    """Test LLama3 model against HuggingFace reference implementation."""
    if not jax._src.xla_bridge.backends_are_initialized():  # type: ignore
        jax.config.update("jax_num_cpu_devices", 2)

    if tp > 1 and os.getenv("CI"):
        pytest.skip("TP > 1 currently runs out of memory in the CI")

    # Use a small LLama model for testing
    model_name = "meta-llama/Llama-3.2-1B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # LLaMA tokenizers don't have a pad token by default, set it to eos_token
    tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)

    inputs = ["The capital of France is", "The most popular programming language is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    with torch.no_grad():
        hf_outputs = hf_model(
            batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True, return_dict=True
        )

    # Save the HF model checkpoint so we can load our model from it
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)

        config = AutoConfig.from_pretrained(model_name)
        mesh = jax.make_mesh((1, tp), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Llama3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        outputs = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy(), output_hidden_states=True)

        assert outputs.hidden_states is not None
        assert np.allclose(hf_outputs.hidden_states[0], outputs.hidden_states[0], rtol=1e-6)
        assert np.allclose(hf_outputs.hidden_states[1], outputs.hidden_states[1], rtol=1e-3, atol=1e-3)
        # Higher tolerance for final layer due to accumulated numerical differences between PyTorch and JAX
        assert np.allclose(hf_outputs.hidden_states[-1], outputs.hidden_states[-1], rtol=5e-2, atol=5e-2)


def test_llama3_forward():
    """Test that LLama3 model can be instantiated and forward pass works."""
    if not jax._src.xla_bridge.backends_are_initialized():  # type: ignore
        jax.config.update("jax_num_cpu_devices", 2)

    # Create a minimal config for testing
    from transformers import LlamaConfig

    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_theta=500000.0,
    )

    mesh = jax.make_mesh((1, 1), ("dp", "tp"))
    with jax.set_mesh(mesh):
        model = Llama3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))

    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask)

    # Check output shapes
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
    assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)
    assert outputs.kv_cache is not None
    assert len(outputs.kv_cache.keys) == config.num_hidden_layers
    assert len(outputs.kv_cache.values) == config.num_hidden_layers


def test_llama3_generation():
    """Test that LLama3 model can be used for simple multi-step generation."""
    if not jax._src.xla_bridge.backends_are_initialized():  # type: ignore
        jax.config.update("jax_num_cpu_devices", 2)

    from transformers import LlamaConfig

    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_theta=500000.0,
    )

    mesh = jax.make_mesh((1, 1), ("dp", "tp"))
    with jax.set_mesh(mesh):
        model = Llama3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))

    # Step 1: Prefill
    batch_size = 2
    seq_len = 10
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    outputs = model(input_ids, attention_mask=attention_mask)
    assert outputs.kv_cache.cache_position == seq_len

    # Step 2: Generate one token by extending the sequence
    next_input_ids = jnp.concatenate([input_ids, jnp.ones((batch_size, 1), dtype=jnp.int32)], axis=1)
    next_attention_mask = jnp.ones((batch_size, seq_len + 1), dtype=jnp.int32)

    next_outputs = model(next_input_ids, attention_mask=next_attention_mask)

    # Check output shapes
    assert next_outputs.logits.shape == (batch_size, seq_len + 1, config.vocab_size)
    assert next_outputs.kv_cache.cache_position == seq_len + 1
