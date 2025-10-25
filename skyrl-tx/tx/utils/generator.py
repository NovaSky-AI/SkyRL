"""Generator mixin for autoregressive text generation with KV caching."""

from __future__ import annotations
from dataclasses import dataclass

import jax
from jax import lax
import jax.numpy as jnp

import tx.utils.models
from tx.tinker import types


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
        generated_ids: List of token ID lists, one for each request (excluding the prompt).
        stop_reasons: Reason for stopping generation for each sequence ('stop' or 'length').
        scores: Logits for each generated token (only if return_scores=True).
    """

    generated_ids: list[list[int]]
    stop_reasons: list[str]
    scores: list[jax.Array] | None = None


def sample_token(logits: jax.Array, *, temperatures: jax.Array, key: jax.Array) -> jax.Array:
    """Sample next token from logits using temperatures."""
    temperatures = temperatures[:, None]
    zero_temp_mask = temperatures == 0.0
    scaled_logits = logits / jnp.where(zero_temp_mask, 1.0, temperatures)
    sampled = jax.random.categorical(key, scaled_logits, axis=-1)[:, None]
    greedy = jnp.argmax(logits, axis=-1)[:, None]
    next_token = jnp.where(zero_temp_mask, greedy, sampled)
    return next_token


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
        sampling_params: list[types.SamplingParams],
        return_scores: bool = False,
        adapter_indices: jax.Array | None = None,
    ) -> GenerateResult:
        """Generate text autoregressively with KV caching.

        Args:
            max_length: Maximum sequence length for fixed-size buffers (default: 512).

        Returns:
            GenerateResult containing generated_ids, stop_reasons, and optionally scores.
        """
        batch_size, prompt_length = input_ids.shape
        assert len(sampling_params) == batch_size
        max_new_tokens = max(sampling_param.max_tokens for sampling_param in sampling_params)
        max_length = tx.utils.models.round_up_seq_len(prompt_length + max_new_tokens)
        temperatures = jnp.array([sampling_param.temperature for sampling_param in sampling_params])
        seeds = [sampling_param.seed for sampling_param in sampling_params]
        # TODO: Implement per-request seeds
        assert all(seed == seeds[0] for seed in seeds), "All seeds must be the same"
        rng = jax.random.PRNGKey(seeds[0])

        # Prefill: process full prompt
        positions = compute_positions(attention_mask)
        outputs = self(input_ids, attention_mask=attention_mask, positions=positions, adapter_indices=adapter_indices)
        kv_cache = outputs["kv_cache"].pad_to_length(max_length)

        def scan_fn(carry, _):
            kv_cache, rng, generated_ids, attention_mask, last_positions, logits = carry
            rng, sample_key = jax.random.split(rng)
            next_token = sample_token(logits, temperatures=temperatures, key=sample_key)

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

        initial_carry = (kv_cache, rng, generated_ids, attention_mask, positions[:, -1:], outputs["logits"][:, -1, :])
        (kv_cache, rng, generated_ids, attention_mask, last_positions, logits), logits_seq = jax.lax.scan(
            scan_fn, initial_carry, xs=None, length=max_new_tokens - 1
        )

        # Sample final token
        rng, sample_key = jax.random.split(rng)
        next_token = sample_token(logits, temperatures=temperatures, key=sample_key)
        generated_ids = lax.dynamic_update_slice(generated_ids, next_token, (0, kv_cache.cache_position))

        # Extract only the newly generated tokens (excluding prompt) per request
        ids = []
        for i, sampling_param in enumerate(sampling_params):
            ids.append(generated_ids[i, prompt_length : prompt_length + sampling_param.max_tokens].tolist())

        return GenerateResult(
            generated_ids=ids,
            stop_reasons=["length"] * batch_size,
            scores=list(logits_seq) + [logits] if return_scores else None,
        )
