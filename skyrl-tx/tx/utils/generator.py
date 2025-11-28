"""Generator mixin for autoregressive text generation with KV caching."""

from __future__ import annotations
from dataclasses import dataclass
import functools

import jax
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


@jax.tree_util.register_dataclass
@dataclass
class DecodeState:
    """State of the decode loop. Lightweight - no large buffers carried."""

    # Constant throughout decode loop:
    stop_tokens: jax.Array
    adapter_indices: jax.Array

    # Updated each iteration:
    kv_cache: KVCache
    rngs: jax.Array  # of shape [B, key_dim]
    last_positions: jax.Array
    logits: jax.Array
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


class GeneratorMixin:
    """Adds autoregressive generation with KV caching to causal language models."""

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("max_length", "max_new_tokens"))
    def _prefill_and_decode(
        model,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        max_length: int,
        max_new_tokens: int,
        adapter_indices: jax.Array | None,
        inv_temperatures: jax.Array,
        zero_temp_mask: jax.Array,
        rngs: jax.Array,
        stop_tokens: jax.Array,
    ):
        """JIT-compiled prefill + decode loop. Fuses everything for maximum efficiency."""
        batch_size = input_ids.shape[0]
        prompt_length = input_ids.shape[1]

        # Compute positions from attention mask
        first_token_idx = jnp.argmax(attention_mask, axis=1, keepdims=True)
        positions = jnp.arange(prompt_length)[None, :] - first_token_idx

        # Prefill: process full prompt
        outputs = model(input_ids, attention_mask=attention_mask, positions=positions, adapter_indices=adapter_indices)

        # Pad KV cache inside JIT (fast)
        kv_cache = outputs.kv_cache.pad_to_length(max_length)

        # Pre-compute index array for mask generation
        cache_index_array = jnp.arange(max_length)

        def decode_fn(s: DecodeState, _) -> tuple[DecodeState, tuple[jax.Array, jax.Array]]:
            """Decode one token step. Returns (state, (token, logprob)) for scan accumulation."""
            # Sample next token
            split_keys = jax.vmap(jax.random.split)(s.rngs)
            rngs, sample_keys = split_keys[:, 0], split_keys[:, 1]

            log_probs = jax.nn.log_softmax(s.logits, axis=-1)
            sampled = jax.vmap(lambda key, logit: jax.random.categorical(key, logit, axis=-1))(
                sample_keys, log_probs * inv_temperatures
            )
            greedy = jnp.argmax(s.logits, axis=-1)
            next_token = jnp.where(zero_temp_mask, greedy[:, None], sampled[:, None])
            sampled_logprob = jnp.take_along_axis(log_probs, next_token, axis=-1)

            # Update stop position if we hit a stop token
            is_stop = jnp.any(next_token == s.stop_tokens, axis=1, keepdims=True)
            stop_pos = jnp.where((s.stop_pos == -1) & is_stop, s.kv_cache.cache_position, s.stop_pos)

            # Generate attention mask on-the-fly from cache_position
            attention_mask = (cache_index_array >= first_token_idx) & (cache_index_array <= s.kv_cache.cache_position)

            outputs = model(
                next_token,
                attention_mask=attention_mask,
                positions=s.last_positions + 1,
                kv_cache=s.kv_cache,
                adapter_indices=s.adapter_indices,
            )
            next_state = DecodeState(
                stop_tokens=s.stop_tokens,
                adapter_indices=s.adapter_indices,
                kv_cache=outputs.kv_cache,
                rngs=rngs,
                last_positions=s.last_positions + 1,
                logits=outputs.logits[:, -1, :],
                stop_pos=stop_pos,
            )
            return next_state, (next_token, sampled_logprob)

        stop_pos = jnp.full((batch_size, 1), -1, dtype=jnp.int32)

        # Build initial state for decode loop
        initial_state = DecodeState(
            stop_tokens=stop_tokens,
            adapter_indices=adapter_indices,
            kv_cache=kv_cache,
            rngs=rngs,
            last_positions=positions[:, -1:],
            logits=outputs.logits[:, -1, :],
            stop_pos=stop_pos,
        )

        # Decode loop - scan accumulates outputs automatically
        final_state, (tokens_stacked, logprobs_stacked) = jax.lax.scan(
            decode_fn, initial_state, xs=None, length=max_new_tokens
        )

        # Post-process: transpose scan outputs from [Steps, Batch, 1] to [Batch, Steps]
        new_tokens = jnp.swapaxes(tokens_stacked, 0, 1).squeeze(-1)
        new_logprobs = jnp.swapaxes(logprobs_stacked, 0, 1).squeeze(-1)

        return new_tokens, new_logprobs, final_state.stop_pos

    def generate(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        *,
        sampling_params: list[types.SamplingParams],
        adapter_indices: jax.Array | None = None,
    ) -> GenerateOutput:
        """Generate text autoregressively with KV caching.

        Returns:
            GenerateOutput containing generated_ids, stop_reasons, and optionally logprobs.
        """
        batch_size, prompt_length = input_ids.shape
        assert len(sampling_params) == batch_size
        max_new_tokens = max(sampling_param.max_tokens for sampling_param in sampling_params)
        max_length = tx.utils.models.round_up_seq_len(prompt_length + max_new_tokens)
        temperatures = jnp.array([sampling_param.temperature for sampling_param in sampling_params])

        # Pre-compute inverse temperatures to avoid division in the decode loop
        zero_temp_mask = (temperatures == 0.0)[:, None]
        inv_temperatures = (1.0 / jnp.maximum(temperatures, 1e-10))[:, None]

        # One PRNGKey per provided seed
        seeds = [sampling_param.seed for sampling_param in sampling_params]
        rngs = jax.vmap(jax.random.PRNGKey)(jnp.array(seeds))

        # Extract stop tokens and pad to same length
        max_stop_tokens = max(len(sp.stop) if sp.stop else 0 for sp in sampling_params)
        stop_tokens = []
        for sp in sampling_params:
            stop = sp.stop or []
            stop_tokens.append(stop + [-1] * (max_stop_tokens - len(stop)))
        stop_tokens = jnp.array(stop_tokens, dtype=jnp.int32)

        # FAST: Single fused JIT call for prefill + decode
        new_tokens, new_logprobs, stop_pos = self._prefill_and_decode(
            self,
            input_ids,
            attention_mask,
            max_length,
            max_new_tokens,
            adapter_indices,
            inv_temperatures,
            zero_temp_mask,
            rngs,
            stop_tokens,
        )

        # Compute end position for each sequence
        end_positions = jnp.where(
            stop_pos[:, 0] >= 0,
            stop_pos[:, 0] + 1 - prompt_length,  # Convert to offset from prompt
            jnp.array([sp.max_tokens for sp in sampling_params]),
        )

        # Single device-to-host transfer
        new_tokens_host, stop_pos_host, new_logprobs_host, end_positions_host = jax.device_get(
            (new_tokens, stop_pos, new_logprobs, end_positions)
        )

        return GenerateOutput(
            generated_ids=[new_tokens_host[i][: end_positions_host[i]].tolist() for i in range(batch_size)],
            stop_reasons=["stop" if stop_pos_host[i, 0] >= 0 else "length" for i in range(batch_size)],
            logprobs=[new_logprobs_host[i][: end_positions_host[i]].tolist() for i in range(batch_size)],
        )
