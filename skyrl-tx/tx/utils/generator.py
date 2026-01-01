"""Generator mixin for autoregressive text generation with KV caching."""

from __future__ import annotations
from dataclasses import dataclass
import functools

import jax
import jax.numpy as jnp
from tokenizers.decoders import DecodeStream
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


def find_string_stop_position(
    tokens: list[int],
    tokenizer,
    stop_strings: list[str],
) -> int | None:
    """Find the token position where a stop string first appears.

    Incrementally decodes tokens and checks for stop string matches.
    Uses the tokenizers DecodeStream for efficient incremental decoding.

    Args:
        tokens: List of generated token IDs
        tokenizer: HuggingFace tokenizer instance
        stop_strings: List of stop strings to search for

    Returns:
        Token index to truncate to (exclusive), or None if no stop found.
    """
    if not stop_strings or not tokens:
        return None

    # Incremental decode using DecodeStream
    stream = DecodeStream(skip_special_tokens=False)
    text = ""
    for i, token in enumerate(tokens):
        chunk = stream.step(tokenizer._tokenizer, token)
        if chunk is not None:
            text += chunk
        for stop_string in stop_strings:
            if stop_string in text:
                return i + 1

    return None


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
    @functools.partial(jax.jit, static_argnames=("max_length", "max_new_tokens", "max_top_k", "prompt_logprobs"))
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
        top_k_values: jax.Array,
        max_top_k: int,
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

            # Apply top_k filtering using static max_top_k
            filtered_logits = apply_top_k_batch(scaled_logits, top_k_values, max_top_k)

            sampled = jax.vmap(lambda key, logit: jax.random.categorical(key, logit, axis=-1))(
                sample_keys, filtered_logits
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
        tokenizer=None,
    ) -> GenerateOutput:
        """Generate text autoregressively with KV caching.

        Args:
            tokenizer: Optional tokenizer for string stop sequence detection.
                Required if any sampling_params has stop_strings set.

        Returns:
            GenerateOutput containing generated_ids, stop_reasons, and optionally logprobs.
        """
        batch_size, prompt_length = input_ids.shape
        assert len(sampling_params) == batch_size
        max_new_tokens = max(sampling_param.max_tokens for sampling_param in sampling_params)
        max_length = tx.utils.models.round_up_seq_len(prompt_length + max_new_tokens)
        temperatures = jnp.array([sampling_param.temperature for sampling_param in sampling_params])
        top_k_values = jnp.array([sampling_param.top_k for sampling_param in sampling_params], dtype=jnp.int32)

        # One PRNGKey per provided seed
        seeds = [sampling_param.seed for sampling_param in sampling_params]
        rngs = jax.vmap(jax.random.PRNGKey)(jnp.array(seeds))

        # Extract stop tokens and pad to same length
        max_stop_tokens = max(len(sp.stop_tokens) if sp.stop_tokens else 0 for sp in sampling_params)
        stop_tokens = []
        for sp in sampling_params:
            stop = sp.stop_tokens or []
            stop_tokens.append(stop + [-1] * (max_stop_tokens - len(stop)))
        stop_tokens = jnp.array(stop_tokens, dtype=jnp.int32)

        # Capture prompt lengths for prompt_logprobs if requested
        prompt_lengths = attention_mask.sum(axis=1) if prompt_logprobs else None

        # Compute max_top_k as static value (0 means no filtering)
        max_top_k = max((sp.top_k for sp in sampling_params if sp.top_k > 0), default=0)

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
            top_k_values,
            max_top_k,
            prompt_logprobs=prompt_logprobs,
        )

        max_tokens = jnp.array([sp.max_tokens for sp in sampling_params])
        # stop_pos is -1 if no stop token found; has_stop is true only if found within limit
        has_stop = (stop_pos != -1) & (stop_pos <= max_tokens)
        end_positions = jnp.where(has_stop, stop_pos, max_tokens)

        # In multi-host mode, gather all shards before device_get
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils

            (new_tokens, has_stop, new_logprobs, end_positions, prompt_logprobs_array, prompt_lengths) = jax.tree.map(
                lambda x: multihost_utils.process_allgather(x, tiled=True),
                (new_tokens, has_stop, new_logprobs, end_positions, prompt_logprobs_array, prompt_lengths),
            )

        # Single device-to-host transfer
        (
            new_tokens_host,
            has_stop_host,
            new_logprobs_host,
            end_positions_host,
            prompt_logprobs_host,
            prompt_lengths_host,
        ) = jax.device_get((new_tokens, has_stop, new_logprobs, end_positions, prompt_logprobs_array, prompt_lengths))

        # Build output lists, applying string stop detection where needed
        generated_ids = []
        stop_reasons = []
        logprobs_out = []

        for i in range(batch_size):
            tokens = new_tokens_host[i][: end_positions_host[i]].tolist()
            token_logprobs = new_logprobs_host[i][: end_positions_host[i]].tolist()
            stop_reason = "stop" if has_stop_host[i] else "length"

            # Apply string stop detection if stop_strings specified
            if sampling_params[i].stop_strings:
                assert tokenizer is not None, "tokenizer is required when stop_strings is specified"
                assert stop_reason == "length", "stop_tokens cannot be specified when stop_strings is specified"
                string_stop_pos = find_string_stop_position(tokens, tokenizer, sampling_params[i].stop_strings)
                if string_stop_pos is not None:
                    tokens = tokens[:string_stop_pos]
                    token_logprobs = token_logprobs[:string_stop_pos]
                    stop_reason = "stop"

            generated_ids.append(tokens)
            stop_reasons.append(stop_reason)
            logprobs_out.append(token_logprobs)

        return GenerateOutput(
            generated_ids=generated_ids,
            stop_reasons=stop_reasons,
            logprobs=logprobs_out,
            prompt_logprobs=(
                [prompt_logprobs_host[i, : prompt_lengths_host[i] - 1].tolist() for i in range(batch_size)]
                if prompt_logprobs
                else None
            ),
        )


def apply_top_k_batch(logits: jax.Array, k_values: jax.Array, max_k: int) -> jax.Array:
    """Keep only top-k logits per example, set rest to -inf.

    Args:
        logits: Logits tensor of shape [batch_size, vocab_size]
        k_values: Per-example k values of shape [batch_size]. If k <= 0, no filtering.
        max_k: Static maximum k value (must be > 0 if any filtering is applied).

    Returns:
        Filtered logits with the same shape.
    """
    if max_k <= 0:
        return logits

    top_values, top_indices = jax.lax.top_k(logits, max_k)

    # Keep only first k values per example
    keep = jnp.arange(max_k) < k_values[:, None]
    top_values = jnp.where(keep, top_values, -jnp.inf)

    # Scatter back to original positions
    batch_idx = jnp.arange(logits.shape[0])[:, None]
    result = jnp.full_like(logits, -jnp.inf).at[batch_idx, top_indices].set(top_values)

    return jnp.where(k_values[:, None] <= 0, logits, result)
