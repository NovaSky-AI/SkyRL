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
        temperature: Sampling temperature (higher = more random)
        key: JAX random key for sampling

    Returns:
        Sampled token indices [batch_size]
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Sample from the distribution
    return jax.random.categorical(key, logits, axis=-1)


class GeneratorMixin:
    """Mixin class that adds text generation capabilities to causal language models.

    This mixin can be inherited by any model class (like Qwen3ForCausalLM) to add
    a generate() method that efficiently produces text using KV caching.

    The model class must implement:
    - __call__(input_ids, attention_mask=None, adapter_indices=None, kv_cache=None)
      that returns dict with 'logits' and optionally 'kv_cache'

    Example:
        class Qwen3ForCausalLM(nnx.Module, GeneratorMixin):
            ...

        model = Qwen3ForCausalLM(config, dtype=jnp.bfloat16, rngs=rngs)
        generated = model.generate(input_ids, max_new_tokens=100, temperature=0.8)
    """

    def generate(
        self,
        input_ids: jax.Array,
        *,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> jax.Array:
        """Generate text autoregressively with KV caching.

        This method implements a two-stage generation process:
        1. Prefill: Process the full input prompt and populate KV cache
        2. Decode: Generate tokens one at a time, updating the cache

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (default: 1.0)
            seed: Random seed for sampling

        Returns:
            Generated token sequences [batch_size, seq_len + max_new_tokens]
        """
        # Initialize random key
        rng = jax.random.PRNGKey(seed)

        # PREFILL STAGE: Process the full prompt
        # This computes attention over the entire input sequence and populates the KV cache
        outputs = self(input_ids)

        # Get the logits for the last token in the prompt
        logits = outputs["logits"][:, -1, :]  # [batch_size, vocab_size]

        # Initialize KV cache if the model supports it
        kv_cache = outputs.get("kv_cache", None)

        # Sample the first generated token
        rng, sample_key = jax.random.split(rng)
        next_token = sample_token(logits, temperature=temperature, key=sample_key)  # [batch_size]

        # Initialize generated sequence with input + first generated token
        generated_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)

        # DECODE STAGE: Generate remaining tokens one at a time
        for step in range(max_new_tokens - 1):
            # Forward pass with the full sequence so far (or just new token if KV cache is available)
            if kv_cache is not None:
                # Use KV cache: only process the last generated token
                current_input = next_token[:, None]  # [batch_size, 1]
                outputs = self(current_input, kv_cache=kv_cache)
            else:
                # No KV cache: process the full sequence
                outputs = self(generated_ids)

            logits = outputs["logits"][:, -1, :]  # [batch_size, vocab_size]
            kv_cache = outputs.get("kv_cache", None)

            # Sample next token
            rng, sample_key = jax.random.split(rng)
            next_token = sample_token(logits, temperature=temperature, key=sample_key)

            # Append to generated sequence
            generated_ids = jnp.concatenate([generated_ids, next_token[:, None]], axis=1)

        return generated_ids
