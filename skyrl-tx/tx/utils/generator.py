"""Generator mixin for autoregressive text generation with KV caching."""

from __future__ import annotations
from dataclasses import dataclass

import jax
from jax import lax
import jax.numpy as jnp
from flax import nnx

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


@jax.tree_util.register_dataclass
@dataclass
class DecodeState:
    """State of the decode loop."""

    # Constant throughout decode loop:
    model: nnx.Module
    temperatures: jax.Array
    stop_tokens: jax.Array
    adapter_indices: jax.Array

    # Updated each iteration:
    kv_cache: KVCache
    rngs: jax.Array  # of shape [B, key_dim]
    generated_ids: jax.Array
    attention_mask: jax.Array
    last_positions: jax.Array
    logits: jax.Array
    all_logprobs: jax.Array
    stop_pos: jax.Array


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


def next_token_and_logprobs(s: DecodeState) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sample next token and compute logprobs, updating the logprobs array."""
    split_keys = jax.vmap(jax.random.split)(s.rngs)
    next_rngs, sample_keys = split_keys[:, 0], split_keys[:, 1]
    next_token = batched_sample_token(s.logits, temperatures=s.temperatures, sample_keys=sample_keys)

    logprobs = jax.nn.log_softmax(s.logits, axis=-1)
    sampled_logprobs = jnp.take_along_axis(logprobs, next_token, axis=-1)  # [batch_size, 1]
    all_logprobs = lax.dynamic_update_slice(s.all_logprobs, sampled_logprobs, (0, s.kv_cache.cache_position))

    # Check if sampled token is in stop tokens and update stop position
    is_stop = jnp.any(next_token == s.stop_tokens, axis=1, keepdims=True)
    # Only update stop_pos if not already stopped (stop_pos == -1)
    stop_pos = jnp.where((s.stop_pos == -1) & is_stop, s.kv_cache.cache_position, s.stop_pos)

    return next_rngs, next_token, all_logprobs, stop_pos


def decode_fn(s: DecodeState, _) -> tuple[DecodeState, None]:
    """Decode one token step for use with jax.lax.scan."""
    rngs, next_token, all_logprobs, stop_pos = next_token_and_logprobs(s)

    generated_ids = lax.dynamic_update_slice(s.generated_ids, next_token, (0, s.kv_cache.cache_position))
    attention_mask = lax.dynamic_update_slice(
        s.attention_mask,
        jnp.ones((s.generated_ids.shape[0], 1), dtype=s.attention_mask.dtype),
        (0, s.kv_cache.cache_position),
    )

    outputs = s.model(
        next_token,
        attention_mask=attention_mask,
        positions=s.last_positions + 1,
        kv_cache=s.kv_cache,
        adapter_indices=s.adapter_indices,
    )
    next_state = DecodeState(
        model=s.model,
        temperatures=s.temperatures,
        stop_tokens=s.stop_tokens,
        adapter_indices=s.adapter_indices,
        kv_cache=outputs.kv_cache,
        rngs=rngs,
        generated_ids=generated_ids,
        attention_mask=attention_mask,
        last_positions=s.last_positions + 1,
        logits=outputs.logits[:, -1, :],
        all_logprobs=all_logprobs,
        stop_pos=stop_pos,
    )
    return next_state, None


class GeneratorMixin:
    """Adds autoregressive generation with KV caching to causal language models."""

    @staticmethod
    @jax.jit
    def _prefill_fn(
        model, input_ids: jax.Array, attention_mask: jax.Array, positions: jax.Array, adapter_indices: jax.Array | None
    ):
        return model(input_ids, attention_mask=attention_mask, positions=positions, adapter_indices=adapter_indices)

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
        max_length = tx.utils.models.round_up_seq_len(prompt_length + max_new_tokens)
        temperatures = jnp.array([sampling_param.temperature for sampling_param in sampling_params])

        # One PRNGKey per provided seed. If the caller supplies identical seeds, the corresponding
        # per-request streams will be identical.
        seeds = [sampling_param.seed for sampling_param in sampling_params]
        rngs = jax.vmap(jax.random.PRNGKey)(jnp.array(seeds))

        # Extract stop tokens and pad to same length
        max_stop_tokens = max(len(sp.stop) if sp.stop else 0 for sp in sampling_params)
        stop_tokens = []
        for sp in sampling_params:
            stop = sp.stop or []
            stop_tokens.append(stop + [-1] * (max_stop_tokens - len(stop)))
        stop_tokens = jnp.array(stop_tokens, dtype=jnp.int32)

        # Prefill: process full prompt
        positions = compute_positions(attention_mask)
        outputs = self._prefill_fn(self, input_ids, attention_mask, positions, adapter_indices)
        kv_cache = outputs.kv_cache.pad_to_length(max_length)

        # Pad inputs to max_length
        pad_length = max_length - prompt_length
        attention_mask = jnp.pad(attention_mask, ((0, 0), (0, pad_length)))
        generated_ids = jnp.pad(input_ids, ((0, 0), (0, pad_length)))
        all_logprobs = jnp.zeros((batch_size, max_length), dtype=outputs.logits.dtype)
        stop_pos = jnp.full((batch_size, 1), -1, dtype=jnp.int32)

        initial_state = DecodeState(
            model=self,
            temperatures=temperatures,
            stop_tokens=stop_tokens,
            adapter_indices=adapter_indices,
            kv_cache=kv_cache,
            rngs=rngs,
            generated_ids=generated_ids,
            attention_mask=attention_mask,
            last_positions=positions[:, -1:],
            logits=outputs.logits[:, -1, :],
            all_logprobs=all_logprobs,
            stop_pos=stop_pos,
        )
        final_state, _ = jax.lax.scan(decode_fn, initial_state, xs=None, length=max_new_tokens - 1)

        # Sample final token
        rngs, next_token, all_logprobs, stop_pos = next_token_and_logprobs(final_state)
        generated_ids = lax.dynamic_update_slice(
            final_state.generated_ids, next_token, (0, final_state.kv_cache.cache_position)
        )

        # Compute end position for each sequence: stop_pos + 1 if stopped, else prompt_length + max_tokens
        end_positions = jnp.where(
            stop_pos[:, 0] >= 0,
            stop_pos[:, 0] + 1,
            prompt_length + jnp.array([sp.max_tokens for sp in sampling_params]),
        )

        # Single device-to-host transfer for all data
        generated_ids_host, stop_pos_host, all_logprobs_host, end_positions_host = jax.device_get(
            (generated_ids[:, prompt_length:], stop_pos, all_logprobs[:, prompt_length:], end_positions - prompt_length)
        )

        return GenerateOutput(
            generated_ids=[generated_ids_host[i][: end_positions_host[i]].tolist() for i in range(batch_size)],
            stop_reasons=["stop" if stop_pos_host[i, 0] >= 0 else "length" for i in range(batch_size)],
            logprobs=[all_logprobs_host[i][: end_positions_host[i]].tolist() for i in range(batch_size)],
        )
