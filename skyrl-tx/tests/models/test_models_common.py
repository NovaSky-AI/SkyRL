"""Common tests for gradient checkpointing."""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformers import AutoConfig, PretrainedConfig

from tx.models.configs import Llama3Config, Qwen3Config
from tx.models.llama3 import Llama3ForCausalLM
from tx.models.qwen3 import Qwen3ForCausalLM


QWEN3_MODEL = "Qwen/Qwen3-0.6B"
LLAMA3_MODEL = "unsloth/Llama-3.2-1B"


def create_qwen3_model():
    """Create Qwen3 model for testing."""
    base_config = PretrainedConfig.from_pretrained(QWEN3_MODEL)
    config = Qwen3Config(base_config, max_lora_adapters=1, max_lora_rank=1, shard_attention_heads=True)
    mesh = jax.make_mesh((1, 1), ("fsdp", "tp"))
    with jax.set_mesh(mesh):
        model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(42))
    return model, config


def create_llama3_model():
    """Create Llama3 model for testing."""
    base_config = AutoConfig.from_pretrained(LLAMA3_MODEL)
    config = Llama3Config(base_config, max_lora_adapters=1, max_lora_rank=1, shard_attention_heads=True)
    mesh = jax.make_mesh((1, 1), ("dp", "tp"))
    with jax.set_mesh(mesh):
        model = Llama3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(42))
    return model, config


@pytest.mark.parametrize("create_model", [create_qwen3_model, create_llama3_model], ids=["qwen3", "llama3"])
class TestGradientCheckpointing:

    def test_output_matches_non_checkpointed(self, create_model):
        """Forward pass should produce identical outputs with/without checkpointing."""
        model, config = create_model()

        batch_size, seq_len = 2, 8
        input_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, config.vocab_size)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        # Run without checkpointing
        config.gradient_checkpointing = False
        out_no_ckpt = model(input_ids, attention_mask=attention_mask, is_training=True)

        # Run with checkpointing
        config.gradient_checkpointing = True
        out_ckpt = model(input_ids, attention_mask=attention_mask, is_training=True)

        np.testing.assert_allclose(out_no_ckpt.logits, out_ckpt.logits, rtol=1e-4, atol=1e-6)

    def test_hidden_states_length_matches(self, create_model):
        """Both paths should return same number of hidden states."""
        model, config = create_model()

        batch_size, seq_len = 2, 8
        input_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, config.vocab_size)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        config.gradient_checkpointing = False
        out_no_ckpt = model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True, is_training=True
        )

        config.gradient_checkpointing = True
        out_ckpt = model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True, is_training=True
        )

        assert len(out_no_ckpt.hidden_states) == len(out_ckpt.hidden_states)
        assert len(out_ckpt.hidden_states) == config.num_hidden_layers + 1

        for i, (hs_no_ckpt, hs_ckpt) in enumerate(
            zip(out_no_ckpt.hidden_states, out_ckpt.hidden_states)
        ):
            np.testing.assert_allclose(
                hs_no_ckpt, hs_ckpt, rtol=1e-4, atol=1e-6, err_msg=f"Mismatch at hidden state {i}"
            )

    def test_is_training_false_uses_standard_path(self, create_model):
        """is_training=False should use standard path with KV cache support."""
        model, config = create_model()
        config.gradient_checkpointing = True

        batch_size, seq_len = 2, 8
        input_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, config.vocab_size)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        out = model(input_ids, attention_mask=attention_mask, is_training=False)

        # KV cache should be populated (checkpointed path returns empty)
        assert len(out.kv_cache.keys) == config.num_hidden_layers
