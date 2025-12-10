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
    """State of the decode loop."""

    kv_cache: KVCache
    rngs: jax.Array  # of shape [B, key_dim]
    attention_mask: jax.Array
    last_positions: jax.Array
    logits: jax.Array
    stop_pos: jax.Array  # Position where stop token was found


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
    prompt_logprobs: list[list[float]] | None = None


def compute_positions(attention_mask: jax.Array) -> jax.Array:
    """Compute positions from attention mask.

    Positions start at 0 from the first non-zero value in the attention mask
    and increment sequentially.
    """
    first_token_idx = jnp.argmax(attention_mask, axis=1, keepdims=True)
    return jnp.arange(attention_mask.shape[1])[None, :] - first_token_idx


def compute_prompt_logprobs(prefill_logits: jax.Array, input_ids: jax.Array) -> jax.Array:
    """Compute log probabilities of prompt tokens from prefill logits"""
    # TODO: Optimize memory usage by avoiding allocation of full vocab dimension.
    logits_for_prompt = prefill_logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits_for_prompt, axis=-1)
    prompt_tokens = input_ids[:, 1:]
    prompt_logprobs = jnp.take_along_axis(log_probs, prompt_tokens[..., None], axis=-1).squeeze(-1)
    return prompt_logprobs


class GeneratorMixin:
    """Adds autoregressive generation with KV caching to causal language models."""

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("max_length", "max_new_tokens", "prompt_logprobs"))
    def _prefill_and_decode(
        model,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        max_length: int,
        max_new_tokens: int,
        adapter_indices: jax.Array | None,
        temperatures: jax.Array,
        rngs: jax.Array,
        stop_tokens: jax.Array,
        prompt_logprobs: bool = False,
    ):
        """JIT-compiled prefill + decode loop. Fuses everything for maximum efficiency."""
        # Compute positions from attention mask
        positions = compute_positions(attention_mask)

        # Prefill: process full prompt
        outputs = model(input_ids, attention_mask=attention_mask, positions=positions, adapter_indices=adapter_indices)

        # Compute prompt logprobs if requested
        prompt_logprobs_array = compute_prompt_logprobs(outputs.logits, input_ids) if prompt_logprobs else None

        # Pad KV cache and attention mask
        kv_cache = outputs.kv_cache.pad_to_length(max_length)
        decode_attention_mask = jnp.pad(attention_mask, ((0, 0), (0, max_length - attention_mask.shape[1])))

        def decode_fn(s: DecodeState, step: jax.Array) -> tuple[DecodeState, tuple[jax.Array, jax.Array]]:
            """Decode one token step. Returns (state, (token, logprob)) for scan accumulation."""
            # Sample next token
            split_keys = jax.vmap(jax.random.split)(s.rngs)
            rngs, sample_keys = split_keys[:, 0], split_keys[:, 1]

            zero_temp_mask = temperatures == 0.0
            scaled_logits = s.logits / jnp.where(zero_temp_mask, 1.0, temperatures)[:, None]
            sampled = jax.vmap(lambda key, logit: jax.random.categorical(key, logit, axis=-1))(
                sample_keys, scaled_logits
            )
            greedy = jnp.argmax(s.logits, axis=-1)
            next_token = jnp.where(zero_temp_mask[:, None], greedy[:, None], sampled[:, None])
            log_probs = jax.nn.log_softmax(s.logits, axis=-1)
            sampled_logprob = jnp.take_along_axis(log_probs, next_token, axis=-1)

            # Track first stop token position (-1 means not stopped yet)
            is_stop = jnp.any(next_token == stop_tokens, axis=1)
            stop_pos = jnp.where((s.stop_pos == -1) & is_stop, step + 1, s.stop_pos)

            # Update attention mask: set next position to 1
            next_attention_mask = s.attention_mask.at[:, s.kv_cache.cache_position].set(1)

            outputs = model(
                next_token,
                attention_mask=next_attention_mask,
                positions=s.last_positions + 1,
                kv_cache=s.kv_cache,
                adapter_indices=adapter_indices,
            )
            next_state = DecodeState(
                kv_cache=outputs.kv_cache,
                rngs=rngs,
                attention_mask=next_attention_mask,
                last_positions=s.last_positions + 1,
                logits=outputs.logits[:, -1, :],
                stop_pos=stop_pos,
            )
            return next_state, (next_token, sampled_logprob)

        initial_state = DecodeState(
            kv_cache=kv_cache,
            rngs=rngs,
            attention_mask=decode_attention_mask,
            last_positions=positions[:, -1:],
            logits=outputs.logits[:, -1, :],
            stop_pos=jnp.full((input_ids.shape[0],), -1),
        )

        final_state, (tokens_stacked, logprobs_stacked) = jax.lax.scan(
            decode_fn, initial_state, xs=jnp.arange(max_new_tokens)
        )

        # Post-process: transpose scan outputs from [Steps, Batch, 1] to [Batch, Steps]
        new_tokens = jnp.swapaxes(tokens_stacked, 0, 1).squeeze(-1)
        new_logprobs = jnp.swapaxes(logprobs_stacked, 0, 1).squeeze(-1)

        return new_tokens, new_logprobs, final_state.stop_pos, prompt_logprobs_array

    def generate(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        *,
        sampling_params: list[types.SamplingParams],
        adapter_indices: jax.Array | None = None,
        prompt_logprobs: bool = False,
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

        # Capture prompt lengths for prompt_logprobs if requested
        prompt_lengths = attention_mask.sum(axis=1) if prompt_logprobs else None

        new_tokens, new_logprobs, stop_pos, prompt_logprobs_array = self._prefill_and_decode(
            self,
            input_ids,
            attention_mask,
            max_length,
            max_new_tokens,
            adapter_indices,
            temperatures,
            rngs,
            stop_tokens,
            prompt_logprobs=prompt_logprobs,
        )

        max_tokens = jnp.array([sp.max_tokens for sp in sampling_params])
        # stop_pos is -1 if no stop token found; has_stop is true only if found within limit
        has_stop = (stop_pos != -1) & (stop_pos <= max_tokens)
        end_positions = jnp.where(has_stop, stop_pos, max_tokens)

        # Single device-to-host transfer
        (
            new_tokens_host,
            has_stop_host,
            new_logprobs_host,
            end_positions_host,
            prompt_logprobs_host,
            prompt_lengths_host,
        ) = jax.device_get((new_tokens, has_stop, new_logprobs, end_positions, prompt_logprobs_array, prompt_lengths))

        return GenerateOutput(
            generated_ids=[new_tokens_host[i][: end_positions_host[i]].tolist() for i in range(batch_size)],
            stop_reasons=["stop" if has_stop_host[i] else "length" for i in range(batch_size)],
            logprobs=[new_logprobs_host[i][: end_positions_host[i]].tolist() for i in range(batch_size)],
            prompt_logprobs=(
                [prompt_logprobs_host[i, : prompt_lengths_host[i] - 1].tolist() for i in range(batch_size)]
                if prompt_logprobs
                else None
            ),
        )
