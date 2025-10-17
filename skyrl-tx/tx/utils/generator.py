"""Generator mixin for autoregressive text generation with KV caching.

This module provides a GeneratorMixin class that can be inherited by causal language models
to add efficient text generation capabilities with KV cache for faster decoding.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class KVCache:
    """Container for key-value cache across all layers.

    Attributes:
        keys: List of key tensors, one per layer [batch_size, num_kv_heads, seq_len, head_dim]
        values: List of value tensors, one per layer [batch_size, num_kv_heads, seq_len, head_dim]
    """

    keys: list[jax.Array]
    values: list[jax.Array]


def sample_token(logits: jax.Array, *, temperature: float = 1.0, key: jax.Array) -> jax.Array:
    """Sample next token from logits using temperature sampling.

    Args:
        logits: Logits tensor [batch_size, vocab_size]
        temperature: Sampling temperature
        key: JAX random key for sampling

    Returns:
        Sampled token indices [batch_size]
    """
    return jax.random.categorical(key, logits / temperature, axis=-1)


class GeneratorMixin:
    """Mixin class that adds text generation capabilities to causal language models.

    This mixin can be inherited by any model class (like Qwen3ForCausalLM) to add
    a generate() method that efficiently produces text using KV caching.

    The model class must implement:
    - __call__(input_ids, attention_mask, adapter_indices=None, kv_cache=None)
      that returns dict with 'logits' and 'kv_cache'

    Example:
        class Qwen3ForCausalLM(nnx.Module, GeneratorMixin):
            ...

        model = Qwen3ForCausalLM(config, dtype=jnp.bfloat16, rngs=rngs)
        generated = model.generate(input_ids, attention_mask, max_new_tokens=100, temperature=0.8)
    """

    def generate(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        *,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> jax.Array:
        """Generate text autoregressively with KV caching.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len], 1 for real tokens, 0 for padding
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            seed: Random seed for sampling

        Returns:
            Generated token sequences [batch_size, seq_len + max_new_tokens]
        """
        rng = jax.random.PRNGKey(seed)
        generated_ids = input_ids

        # Prefill: process full prompt
        outputs = self(input_ids, attention_mask=attention_mask)

        # Decode: generate tokens one at a time
        for step in range(max_new_tokens):
            rng, sample_key = jax.random.split(rng)
            next_token = sample_token(outputs["logits"][:, -1, :], temperature=temperature, key=sample_key)
            generated_ids = jnp.concatenate([generated_ids, next_token[:, None]], axis=1)

            if step < max_new_tokens - 1:
                attention_mask = jnp.concatenate([attention_mask, jnp.ones_like(next_token)[:, None]], axis=1)
                outputs = self(next_token[:, None], attention_mask=attention_mask, kv_cache=outputs["kv_cache"])

        return generated_ids
