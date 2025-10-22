"""Generator mixin for autoregressive text generation with KV caching."""

from __future__ import annotations
from dataclasses import dataclass

import jax
from jax import lax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass
class KVCache:
    """Key-value cache for all layers, each entry in the list corresponds to one layer."""

    keys: list[jax.Array]
    values: list[jax.Array]
    cache_position: int

    def pad_to_length(self, max_length: int) -> KVCache:
        """Pad KV cache to a specified maximum length.

        Args:
            max_length: Target length to pad the cache to.

        Returns:
            New KVCache with padded keys and values.
        """
        # k and v have shape [B, T, num_heads, head_dim]
        cache_pad_length = max_length - self.keys[0].shape[1]
        pad_spec = ((0, 0), (0, cache_pad_length), (0, 0), (0, 0))
        return KVCache(
            keys=[jnp.pad(k, pad_spec) for k in self.keys],
            values=[jnp.pad(v, pad_spec) for v in self.values],
            cache_position=self.cache_position,
        )


@dataclass
class GenerateResult:
    """Result from autoregressive text generation.

    Attributes:
        generated_ids: Token IDs of the generated text including the prompt.
        stop_reasons: Reason for stopping generation for each sequence ('stop' or 'length').
        scores: Logits for each generated token (only if return_scores=True).
    """

    generated_ids: jax.Array
    stop_reasons: list[str]
    scores: list[jax.Array] | None = None


def sample_token(logits: jax.Array, *, temperature: float, key: jax.Array) -> jax.Array:
    """Sample next token from logits using temperature."""
    if temperature == 0.0:
        return jnp.argmax(logits, axis=-1)[:, None]
    return jax.random.categorical(key, logits / temperature, axis=-1)[:, None]


def compute_positions(attention_mask: jax.Array) -> jax.Array:
    """Compute positions from attention mask.

    Positions start at 0 from the first non-zero value in the attention mask
    and increment sequentially.
    """
    first_token_idx = jnp.argmax(attention_mask, axis=1, keepdims=True)
    return jnp.arange(attention_mask.shape[1])[None, :] - first_token_idx


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
        return_scores: bool = False,
        adapter_indices: jax.Array | None = None,
        max_length: int = 512,
    ) -> GenerateResult:
        """Generate text autoregressively with KV caching.

        Args:
            max_length: Maximum sequence length for fixed-size buffers (default: 512).

        Returns:
            GenerateResult containing generated_ids, stop_reasons, and optionally scores.
        """
        batch_size, prompt_length = input_ids.shape

        # Prefill: process full prompt
        positions = compute_positions(attention_mask)
        outputs = self(input_ids, attention_mask=attention_mask, positions=positions, adapter_indices=adapter_indices)
        kv_cache = outputs["kv_cache"].pad_to_length(max_length)

        def scan_fn(carry, _):
            kv_cache, rng, generated_ids, attention_mask, last_positions, logits = carry
            rng, sample_key = jax.random.split(rng)
            next_token = sample_token(logits, temperature=temperature, key=sample_key)

            # Update generated_ids and attention mask
            generated_ids = lax.dynamic_update_slice(generated_ids, next_token, (0, kv_cache.cache_position))
            attention_mask = lax.dynamic_update_slice(
                attention_mask, jnp.ones((batch_size, 1), dtype=attention_mask.dtype), (0, kv_cache.cache_position)
            )
            last_positions = last_positions + 1

            # Run decoder step
            outputs = self(
                next_token,
                attention_mask=attention_mask,
                positions=last_positions,
                kv_cache=kv_cache,
                adapter_indices=adapter_indices,
            )

            new_logits = outputs["logits"][:, -1, :]
            new_carry = (outputs["kv_cache"], rng, generated_ids, attention_mask, last_positions, new_logits)
            return new_carry, logits if return_scores else None

        # Pad inputs to max_length
        pad_length = max_length - prompt_length
        attention_mask = jnp.pad(attention_mask, ((0, 0), (0, pad_length)))
        generated_ids = jnp.pad(input_ids, ((0, 0), (0, pad_length)))

        initial_carry = (
            kv_cache,
            jax.random.PRNGKey(seed),
            generated_ids,
            attention_mask,
            positions[:, -1:],
            outputs["logits"][:, -1, :],
        )
        (_, _, generated_ids, _, _, _), logits_seq = jax.lax.scan(
            scan_fn, initial_carry, xs=None, length=max_new_tokens
        )

        return GenerateResult(
            generated_ids=generated_ids[:, : prompt_length + max_new_tokens],
            stop_reasons=["length"] * batch_size,
            scores=list(logits_seq) if return_scores else None,
        )
