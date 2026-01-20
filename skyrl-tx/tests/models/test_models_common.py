"""Common tests for Llama3 and Qwen3 models."""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformers import PretrainedConfig

from tx.models.configs import Llama3Config, Qwen3Config
from tx.models.llama3 import Llama3ForCausalLM
from tx.models.qwen3 import Qwen3ForCausalLM


def make_small_config(config_class, gradient_checkpointing=False, num_hidden_layers=2):
    """Create a minimal config for fast testing."""
    base_config = PretrainedConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=1000,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
    )
    return config_class(
        base_config,
        max_lora_adapters=1,
        max_lora_rank=1,
        shard_attention_heads=False,
        gradient_checkpointing=gradient_checkpointing,
    )


@pytest.fixture
def input_batch():
    """Common test inputs."""
    batch_size, seq_len = 2, 16
    input_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, 1000)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    return input_ids, attention_mask


@pytest.mark.parametrize("model_class,config_class", [
    (Llama3ForCausalLM, Llama3Config),
    (Qwen3ForCausalLM, Qwen3Config),
])
class TestGradientCheckpointing:

    def test_output_matches_non_checkpointed(self, model_class, config_class, input_batch):
        """Forward pass should produce identical outputs with/without checkpointing."""
        input_ids, attention_mask = input_batch

        # Create model without checkpointing
        config = make_small_config(config_class, gradient_checkpointing=False)
        model = model_class(config, dtype=jnp.float32, rngs=nnx.Rngs(42))
        out_no_ckpt = model(input_ids, attention_mask=attention_mask, is_training=True)

        # Enable checkpointing
        config.gradient_checkpointing = True
        out_ckpt = model(input_ids, attention_mask=attention_mask, is_training=True)

        np.testing.assert_allclose(out_no_ckpt.logits, out_ckpt.logits, rtol=1e-5)

    def test_hidden_states_length_matches(self, model_class, config_class, input_batch):
        """Both paths should return same number of hidden states."""
        input_ids, attention_mask = input_batch
        config = make_small_config(config_class, gradient_checkpointing=False)
        model = model_class(config, dtype=jnp.float32, rngs=nnx.Rngs(42))

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
                hs_no_ckpt, hs_ckpt, rtol=1e-5, err_msg=f"Mismatch at hidden state {i}"
            )

    def test_is_training_false_uses_standard_path(self, model_class, config_class, input_batch):
        """is_training=False should use standard path with KV cache support."""
        input_ids, attention_mask = input_batch
        config = make_small_config(config_class, gradient_checkpointing=True)
        model = model_class(config, dtype=jnp.float32, rngs=nnx.Rngs(42))

        out = model(input_ids, attention_mask=attention_mask, is_training=False)

        # KV cache should be populated (checkpointed path returns empty)
        assert len(out.kv_cache.keys) == config.num_hidden_layers

    def test_single_layer_model(self, model_class, config_class, input_batch):
        """Checkpointing should work with single layer."""
        input_ids, attention_mask = input_batch

        config = make_small_config(config_class, gradient_checkpointing=True, num_hidden_layers=1)
        model = model_class(config, dtype=jnp.float32, rngs=nnx.Rngs(42))

        out = model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True, is_training=True
        )

        # [embed, normed_output]
        assert len(out.hidden_states) == 2

    def test_single_layer_output_matches(self, model_class, config_class, input_batch):
        """Single layer model outputs should match with/without checkpointing."""
        input_ids, attention_mask = input_batch

        config = make_small_config(config_class, gradient_checkpointing=False, num_hidden_layers=1)
        model = model_class(config, dtype=jnp.float32, rngs=nnx.Rngs(42))

        out_no_ckpt = model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True, is_training=True
        )

        config.gradient_checkpointing = True
        out_ckpt = model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True, is_training=True
        )

        np.testing.assert_allclose(out_no_ckpt.logits, out_ckpt.logits, rtol=1e-5)
        assert len(out_no_ckpt.hidden_states) == len(out_ckpt.hidden_states)
