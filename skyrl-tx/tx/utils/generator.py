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
class GenerateOutput:
    """Result from autoregressive text generation.

    Attributes:
        generated_ids: List of token ID lists, one for each request (excluding the prompt).
        stop_reasons: Reason for stopping generation for each sequence ('stop' or 'length').
        logprobs: Log probabilities for each sampled token.
    """

    generated_ids: list[list[int]]
    stop_reasons: list[str]
    logprobs: list[list[float]]


def batched_sample_token(logits: jax.Array, *, temperatures: jax.Array, sample_keys: jax.Array) -> jax.Array:
    """Sample next token per-example using a per-example PRNGKey."""
    temperatures = temperatures[:, None]
    zero_temp_mask = temperatures == 0.0
    scaled_logits = logits / jnp.where(zero_temp_mask, 1.0, temperatures)
    # Draw one sample per example
    sampled = jax.vmap(lambda key, logit: jax.random.categorical(key, logit, axis=-1))(sample_keys, scaled_logits)
    greedy = jnp.argmax(logits, axis=-1)
    next_token = jnp.where(zero_temp_mask, greedy[:, None], sampled[:, None])
    return next_token


def compute_positions(attention_mask: jax.Array) -> jax.Array:
    """Compute positions from attention mask.

    Positions start at 0 from the first non-zero value in the attention mask
    and increment sequentially.
    """
    first_token_idx = jnp.argmax(attention_mask, axis=1, keepdims=True)
    return jnp.arange(attention_mask.shape[1])[None, :] - first_token_idx


def next_token_and_logprobs(
    logits: jax.Array,
    temperatures: jax.Array,
    rngs: jax.Array,  # Shape [B, key_dim]
    all_logprobs: jax.Array,
    cache_position: int,
    stop_tokens: jax.Array,
    stop_pos: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sample next token and compute logprobs, updating the logprobs array."""
    split_keys = jax.vmap(jax.random.split)(rngs)
    next_rngs, sample_keys = split_keys[:, 0], split_keys[:, 1]
    next_token = batched_sample_token(logits, temperatures=temperatures, sample_keys=sample_keys)

    logprobs = jax.nn.log_softmax(logits, axis=-1)
    sampled_logprobs = jnp.take_along_axis(logprobs, next_token, axis=-1)  # [batch_size, 1]
    all_logprobs = lax.dynamic_update_slice(all_logprobs, sampled_logprobs, (0, cache_position))

    # Check if sampled token is in stop tokens and update stop position
    is_stop = jnp.any(next_token == stop_tokens, axis=1, keepdims=True)
    # Only update stop_pos if not already stopped (stop_pos == -1)
    stop_pos = jnp.where((stop_pos == -1) & is_stop, cache_position, stop_pos)

    return next_rngs, next_token, all_logprobs, stop_pos


def decode_fn(carry, _):
    """Decode one token step for use with jax.lax.scan.

    Args:
        carry: Tuple of (kv_cache, rngs, generated_ids, attention_mask, last_positions, logits,
                        all_logprobs, stop_pos, model, temperatures, stop_tokens, adapter_indices)
        _: Unused scan input

    Returns:
        Tuple of (new_carry, None) where new_carry has the same structure as carry
    """
    (kv_cache, rngs, generated_ids, attention_mask, last_positions, logits, all_logprobs, stop_pos,
     model, temperatures, stop_tokens, adapter_indices) = carry
    batch_size = generated_ids.shape[0]

    rngs, next_token, all_logprobs, stop_pos = next_token_and_logprobs(
        logits, temperatures, rngs, all_logprobs, kv_cache.cache_position, stop_tokens, stop_pos
    )

    # Update generated_ids and attention mask
    generated_ids = lax.dynamic_update_slice(generated_ids, next_token, (0, kv_cache.cache_position))
    attention_mask = lax.dynamic_update_slice(
        attention_mask, jnp.ones((batch_size, 1), dtype=attention_mask.dtype), (0, kv_cache.cache_position)
    )
    last_positions = last_positions + 1

    # Run decoder step
    outputs = model(
        next_token,
        attention_mask=attention_mask,
        positions=last_positions,
        kv_cache=kv_cache,
        adapter_indices=adapter_indices,
    )

    new_logits = outputs.logits[:, -1, :]
    new_carry = (
        outputs.kv_cache,
        rngs,
        generated_ids,
        attention_mask,
        last_positions,
        new_logits,
        all_logprobs,
        stop_pos,
        model,
        temperatures,
        stop_tokens,
        adapter_indices,
    )
    return new_carry, None


class GeneratorMixin:
    """Adds autoregressive generation with KV caching to causal language models."""

    @staticmethod
    @jax.jit
    def _prefill_fn(model, input_ids, attention_mask, positions, adapter_indices):
        return model(input_ids, attention_mask=attention_mask, positions=positions, adapter_indices=adapter_indices)
    
    def generate_padded(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        *,
        sampling_params: list[types.SamplingParams],
        adapter_indices: jax.Array | None = None,
    ) -> GenerateOutput:
        batch_size, prompt_length = input_ids.shape
        rounded_batch_size = tx.utils.models.round_up_seq_len(batch_size)

        # Pad batch dimension
        batch_pad_size = rounded_batch_size - batch_size
        input_ids_padded = jnp.pad(input_ids, ((0, batch_pad_size), (0, 0)))
        attention_mask_padded = jnp.pad(attention_mask, ((0, batch_pad_size), (0, 0)))

        # Pad sampling_params by duplicating the last element
        sampling_params_padded = sampling_params + [sampling_params[-1]] * batch_pad_size

        # Pad adapter_indices if provided
        if adapter_indices is not None:
            adapter_indices_padded = jnp.pad(adapter_indices, ((0, batch_pad_size),))
        else:
            adapter_indices_padded = None

        # Generate with padded batch
        output = self.generate(
            input_ids_padded,
            attention_mask_padded,
            sampling_params=sampling_params_padded,
            adapter_indices=adapter_indices_padded,
        )

        # Extract only the original batch results
        return GenerateOutput(
            generated_ids=output.generated_ids[:batch_size],
            stop_reasons=output.stop_reasons[:batch_size],
            logprobs=output.logprobs[:batch_size],
        )

    def generate(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        *,
        sampling_params: list[types.SamplingParams],
        adapter_indices: jax.Array | None = None,
    ) -> GenerateOutput:
        """Generate text autoregressively with KV caching.

        Args:
            max_length: Maximum sequence length for fixed-size buffers (default: 512).

        Returns:
            GenerateOutput containing generated_ids, stop_reasons, and optionally logprobs.
        """
        batch_size, prompt_length = input_ids.shape
        assert len(sampling_params) == batch_size
        max_new_tokens = max(sampling_param.max_tokens for sampling_param in sampling_params)

        # Round up prompt_length for prefill JIT efficiency
        rounded_prompt_length = tx.utils.models.round_up_seq_len(prompt_length)
        # Round up total length for decode JIT efficiency
        max_length = tx.utils.models.round_up_seq_len(prompt_length + max_new_tokens)

        temperatures = jnp.array([sampling_param.temperature for sampling_param in sampling_params])

        # One PRNGKey per provided seed. If the caller supplies identical seeds, the corresponding
        # per-request streams will be identical.
        seeds = [sampling_param.seed for sampling_param in sampling_params]
        rngs = jax.vmap(jax.random.PRNGKey)(jnp.array(seeds))

        # Extract stop tokens and pad to same length
        max_stop_tokens = max(len(sp.stop) if sp.stop else 0 for sp in sampling_params)
        stop_tokens = jnp.full((batch_size, max(1, max_stop_tokens)), -1, dtype=jnp.int32)
        for i, sp in enumerate(sampling_params):
            if sp.stop:
                stop_tokens = stop_tokens.at[i, : len(sp.stop)].set(jnp.array(sp.stop))

        # Pad inputs to rounded_prompt_length for prefill
        prefill_pad_length = rounded_prompt_length - prompt_length
        attention_mask_prefill = jnp.pad(attention_mask, ((0, 0), (0, prefill_pad_length)))
        input_ids_prefill = jnp.pad(input_ids, ((0, 0), (0, prefill_pad_length)))

        # Prefill: process full prompt
        positions = compute_positions(attention_mask_prefill)
        outputs = self._prefill_fn(self, input_ids_prefill, attention_mask_prefill, positions, adapter_indices)
        kv_cache = outputs.kv_cache.pad_to_length(max_length)

        # Pad inputs to max_length for decode
        decode_pad_length = max_length - prompt_length
        attention_mask = jnp.pad(attention_mask, ((0, 0), (0, decode_pad_length)))
        generated_ids = jnp.pad(input_ids, ((0, 0), (0, decode_pad_length)))

        all_logprobs = jnp.zeros((batch_size, max_length), dtype=outputs.logits.dtype)
        stop_pos = jnp.full((batch_size, 1), -1, dtype=jnp.int32)

        initial_carry = (
            kv_cache,
            rngs,
            generated_ids,
            attention_mask,
            positions[:, prompt_length - 1 : prompt_length],
            outputs.logits[:, prompt_length - 1, :],
            all_logprobs,
            stop_pos,
            self,
            temperatures,
            stop_tokens,
            adapter_indices,
        )
        (kv_cache, rngs, generated_ids, attention_mask, last_positions, logits, all_logprobs, stop_pos,
         _, _, _, _), _ = (
            jax.lax.scan(decode_fn, initial_carry, xs=None, length=max_new_tokens - 1)
        )

        # Sample final token
        rngs, next_token, all_logprobs, stop_pos = next_token_and_logprobs(
            logits, temperatures, rngs, all_logprobs, kv_cache.cache_position, stop_tokens, stop_pos
        )
        generated_ids = lax.dynamic_update_slice(generated_ids, next_token, (0, kv_cache.cache_position))

        # Compute end position for each sequence: stop_pos + 1 if stopped, else prompt_length + max_tokens
        end_positions = jnp.where(
            stop_pos[:, 0] >= 0,
            stop_pos[:, 0] + 1,
            prompt_length + jnp.array([sp.max_tokens for sp in sampling_params]),
        )

        return GenerateOutput(
            generated_ids=[generated_ids[i, prompt_length : end_positions[i]].tolist() for i in range(batch_size)],
            stop_reasons=["stop" if stop_pos[i, 0] >= 0 else "length" for i in range(batch_size)],
            logprobs=[all_logprobs[i, prompt_length : end_positions[i]].tolist() for i in range(batch_size)],
        )
