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
    cache_position: jax.Array  # Per-sequence positions of shape [B] for left-aligned decoding

    @staticmethod
    def update(
        kv_cache: KVCache | None,
        keys: list[jax.Array],
        values: list[jax.Array],
        positions: jax.Array,
        attention_mask: jax.Array,
    ) -> KVCache:
        """Create an updated KVCache with computed cache positions for left-aligned decoding.

        Args:
            kv_cache: Existing KVCache (None during prefill).
            keys: List of key arrays per layer.
            values: List of value arrays per layer.
            positions: Position indices with shape [B, seq_len].
            attention_mask: Attention mask with shape [B, seq_len].

        Returns:
            New KVCache with computed cache_position.
        """
        if kv_cache is not None:
            # Decode: next position is current position + 1
            cache_position = positions[:, 0] + 1
        else:
            # Prefill: next position is the sequence length (number of real tokens)
            cache_position = attention_mask.sum(axis=1)
        return KVCache(keys=keys, values=values, cache_position=cache_position)

    @staticmethod
    def update_layer(kv_cache, k, v, positions):
        """Update a single layer's KV cache at the given positions (for left-aligned decoding).

        Args:
            kv_cache: Tuple of (k_cache, v_cache) arrays for this layer.
            k: New key values with shape [B, seq_len, num_heads, head_dim].
            v: New value values with shape [B, seq_len, num_heads, head_dim].
            positions: Position indices with shape [B, seq_len].
        """
        k_cache, v_cache = kv_cache

        def update_at_pos(cache_slice, new_val_slice, pos):
            return jax.lax.dynamic_update_slice(cache_slice, new_val_slice, (pos, 0, 0))

        k = jax.vmap(update_at_pos)(k_cache, k, positions[:, 0])
        v = jax.vmap(update_at_pos)(v_cache, v, positions[:, 0])
        return k, v

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


class GeneratorMixin:
    """Adds autoregressive generation with KV caching to causal language models."""

    @staticmethod
    @functools.partial(
        jax.jit, static_argnames=("max_length", "max_new_tokens", "max_top_k", "use_top_p", "prompt_logprobs")
    )
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
        top_p_values: jax.Array,
        max_top_k: int,
        use_top_p: bool,
        prompt_logprobs: bool = False,
    ):
        """JIT-compiled prefill + decode loop. Fuses everything for maximum efficiency."""
        # Prefill: process full prompt (left-aligned, so positions start at 0)
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            adapter_indices=adapter_indices,
        )

        # For left-aligned sequences, find the last real token position for each sequence
        last_token_idx = attention_mask.sum(axis=1) - 1  # Shape: [B]
        batch_idx = jnp.arange(input_ids.shape[0])

        # Compute logits for sampling and optionally for prompt logprobs
        if prompt_logprobs:
            # Compute all logits for prompt logprobs and sampling the first token
            all_logits = model.compute_logits(outputs.last_hidden_state, adapter_indices)
            last_logits = all_logits[batch_idx, last_token_idx, :]  # Shape: [B, vocab_size]
            prompt_logprobs_array = model.logits_to_logprobs(all_logits[:, :-1, :], input_ids[:, 1:])
        else:
            # Only compute logits for the last position for sampling
            last_hidden = outputs.last_hidden_state[batch_idx, last_token_idx][:, None, :]  # Shape: [B, 1, H]
            last_logits = model.compute_logits(last_hidden, adapter_indices)[:, 0, :]
            prompt_logprobs_array = None

        # Pad KV cache and attention mask
        kv_cache = outputs.kv_cache.pad_to_length(max_length)

        # Pad KV cache and attention mask to max_length
        kv_cache = kv_cache.pad_to_length(max_length)
        decode_attention_mask = jnp.pad(attention_mask, ((0, 0), (0, max_length - attention_mask.shape[1])))

        def decode_fn(s: DecodeState, step: jax.Array) -> tuple[DecodeState, tuple[jax.Array, jax.Array]]:
            """Decode one token step. Returns (state, (token, logprob)) for scan accumulation."""
            # Sample next token
            split_keys = jax.vmap(jax.random.split)(s.rngs)
            rngs, sample_keys = split_keys[:, 0], split_keys[:, 1]

            zero_temp_mask = temperatures == 0.0
            scaled_logits = s.logits / jnp.where(zero_temp_mask, 1.0, temperatures)[:, None]

            # Apply top_k and top_p filtering
            if max_top_k > 0:
                scaled_logits = apply_top_k_batch(scaled_logits, top_k_values, max_top_k)
            if use_top_p:
                scaled_logits = apply_top_p_batch(scaled_logits, top_p_values)

            sampled = jax.vmap(lambda key, logit: jax.random.categorical(key, logit, axis=-1))(
                sample_keys, scaled_logits
            )
            greedy = jnp.argmax(s.logits, axis=-1)
            next_token = jnp.where(zero_temp_mask[:, None], greedy[:, None], sampled[:, None])
            sampled_logprob = model.logits_to_logprobs(s.logits, next_token[:, 0])[:, None]

            # Track first stop token position (-1 means not stopped yet)
            is_stop = jnp.any(next_token == stop_tokens, axis=1)
            stop_pos = jnp.where((s.stop_pos == -1) & is_stop, step + 1, s.stop_pos)

            # Update attention mask at per-sequence positions (for left-aligned sequences)
            batch_idx = jnp.arange(s.attention_mask.shape[0])
            next_attention_mask = s.attention_mask.at[batch_idx, s.kv_cache.cache_position].set(1)

            outputs = model(
                next_token,
                attention_mask=next_attention_mask,
                positions=s.last_positions + 1,
                kv_cache=s.kv_cache,
                adapter_indices=adapter_indices,
            )
            # Compute logits for the next token
            next_logits = model.compute_logits(outputs.last_hidden_state, adapter_indices)[:, 0, :]
            next_state = DecodeState(
                kv_cache=outputs.kv_cache,
                rngs=rngs,
                attention_mask=next_attention_mask,
                last_positions=s.last_positions + 1,
                logits=next_logits,
                stop_pos=stop_pos,
            )
            return next_state, (next_token, sampled_logprob)

        initial_state = DecodeState(
            kv_cache=kv_cache,
            rngs=rngs,
            attention_mask=decode_attention_mask,
            last_positions=last_token_idx[:, None],
            logits=last_logits,
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
        top_p_values = jnp.array([sampling_param.top_p for sampling_param in sampling_params], dtype=jnp.float32)

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

        # Compute static flags for top_k and top_p filtering
        max_top_k = max((sp.top_k for sp in sampling_params if sp.top_k > 0), default=0)
        use_top_p = any(sp.top_p < 1.0 for sp in sampling_params)

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
            top_p_values,
            max_top_k,
            use_top_p,
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
        max_k: Static maximum k value, must be > 0.

    Returns:
        Filtered logits with the same shape.
    """
    assert max_k > 0
    top_values, top_indices = jax.lax.top_k(logits, max_k)

    # Keep only first k values per example
    keep = jnp.arange(max_k) < k_values[:, None]
    top_values = jnp.where(keep, top_values, -jnp.inf)

    # Scatter back to original positions
    batch_idx = jnp.arange(logits.shape[0])[:, None]
    result = jnp.full_like(logits, -jnp.inf).at[batch_idx, top_indices].set(top_values)

    return jnp.where(k_values[:, None] <= 0, logits, result)


def apply_top_p_batch(logits: jax.Array, p_values: jax.Array) -> jax.Array:
    """Keep only tokens with cumulative probability up to p, set rest to -inf.

    Args:
        logits: Logits tensor of shape [batch_size, vocab_size]
        p_values: Per-example p values of shape [batch_size]. If p >= 1.0, no filtering.

    Returns:
        Filtered logits with the same shape.
    """
    # Sort by logits (equivalent to sorting by probs since softmax is monotonic)
    sorted_indices = jnp.argsort(-logits, axis=-1)
    sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
    sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)

    # Exclusive cumsum: cumsum[i] - prob[i] gives sum of probs *before* position i
    cumsum_exclusive = jnp.cumsum(sorted_probs, axis=-1) - sorted_probs

    keep_mask = cumsum_exclusive < p_values[:, None]
    keep_mask = keep_mask.at[:, 0].set(True)  # Always keep top token
    filtered_sorted_logits = jnp.where(keep_mask, sorted_logits, -jnp.inf)

    # Scatter back to original positions
    batch_idx = jnp.arange(logits.shape[0])[:, None]
    result = jnp.empty_like(logits).at[batch_idx, sorted_indices].set(filtered_sorted_logits)

    return jnp.where(p_values[:, None] >= 1.0, logits, result)
