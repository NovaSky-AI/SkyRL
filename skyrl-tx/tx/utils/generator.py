"""Generator mixin for autoregressive text generation with KV caching."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class KVCache:
    """Key-value cache for all layers, each entry in the list corresponds to one layer."""

    keys: list[jax.Array]
    values: list[jax.Array]


def sample_token(logits: jax.Array, *, temperature: float, key: jax.Array) -> jax.Array:
    """Sample next token from logits using temperature."""
    if temperature == 0.0:
        return jnp.argmax(logits, axis=-1)[:, None]
    return jax.random.categorical(key, logits / temperature, axis=-1)[:, None]


class GeneratorMixin:
    """Adds autoregressive generation with KV caching to causal language models."""

    def generate(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int,
    ) -> jax.Array:
        """Generate text autoregressively with KV caching."""
        rng = jax.random.PRNGKey(seed)
        generated_ids = input_ids

        # Prefill: process full prompt
        outputs = self(input_ids, attention_mask=attention_mask)

        # Decode: generate tokens one at a time
        for step in range(max_new_tokens):
            rng, sample_key = jax.random.split(rng)
            next_token = sample_token(outputs["logits"][:, -1, :], temperature=temperature, key=sample_key)
            generated_ids = jnp.concatenate([generated_ids, next_token], axis=1)

            if step < max_new_tokens - 1:
                attention_mask = jnp.concatenate([attention_mask, jnp.ones_like(next_token)], axis=1)
                outputs = self(next_token, attention_mask=attention_mask, kv_cache=outputs["kv_cache"])

        return generated_ids
