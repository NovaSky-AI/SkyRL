"""Generator mixin for autoregressive text generation with KV caching."""

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

    def pad_to_length(self, max_length: int) -> "KVCache":
        """Pad KV cache to a specified maximum length.

        Args:
            max_length: Target length to pad the cache to.

        Returns:
            New KVCache with padded keys and values.
        """
        padded_keys = []
        padded_values = []
        for k, v in zip(self.keys, self.values):
            # k and v have shape [B, T, num_heads, head_dim]
            cache_pad_length = max_length - k.shape[1]
            padded_k = jnp.pad(k, ((0, 0), (0, cache_pad_length), (0, 0), (0, 0)), constant_values=0)
            padded_v = jnp.pad(v, ((0, 0), (0, cache_pad_length), (0, 0), (0, 0)), constant_values=0)
            padded_keys.append(padded_k)
            padded_values.append(padded_v)
        return KVCache(keys=padded_keys, values=padded_values, cache_position=self.cache_position)


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

    def init_generator(self):
        """Initialize the generator by creating a jitted decoder step function."""

        @jax.jit
        def decoder_step(model_fn, next_token, attention_mask, last_positions, kv_cache, adapter_indices):
            """Single decoder step with KV caching."""
            outputs = model_fn(
                next_token,
                attention_mask=attention_mask,
                positions=last_positions,
                kv_cache=kv_cache,
                adapter_indices=adapter_indices,
            )
            return outputs

        self.decoder_step = decoder_step

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
        rng = jax.random.PRNGKey(seed)
        batch_size = input_ids.shape[0]
        prompt_length = input_ids.shape[1]

        # Create fixed-size buffers
        generated_ids = input_ids
        scores = [] if return_scores else None
        stop_reasons = ["length"] * batch_size

        # Pad attention_mask to max_length
        pad_length = max_length - attention_mask.shape[1]
        attention_mask_padded = jnp.pad(attention_mask, ((0, 0), (0, pad_length)), constant_values=0)

        # Prefill: process full prompt
        positions = compute_positions(attention_mask)
        outputs = self(input_ids, attention_mask=attention_mask, positions=positions, adapter_indices=adapter_indices)

        # Pad KV cache to max_length
        outputs["kv_cache"] = outputs["kv_cache"].pad_to_length(max_length)

        # Keep track of only the last position for decoding
        last_positions = positions[:, -1:]

        #  Pre allocate in advance generated_ids buffer with fixed size to avoid rechaching
        generated_ids_buf = jnp.zeros((batch_size, max_length), dtype=input_ids.dtype)
        generated_ids_buf = generated_ids_buf.at[:, :prompt_length].set(input_ids)

        def scan_fn(carry, _):
            """Autoregressively generate with jax.scan for efficiency"""
            kv_cache, rng, generated_ids, attention_mask_padded, last_positions, logits = carry

            rng, sample_key = jax.random.split(rng)

            next_token = sample_token(logits, temperature=temperature, key=sample_key)

            # Update generated_ids and attention mask
            generated_ids = lax.dynamic_update_slice(generated_ids, next_token, (0, kv_cache.cache_position))
            mask_update = jnp.ones((batch_size, 1), dtype=attention_mask_padded.dtype)
            attention_mask_padded = lax.dynamic_update_slice(
                attention_mask_padded, mask_update, (0, kv_cache.cache_position)
            )

            last_positions = last_positions + 1

            # Run decoder step (cache_position will be incremented inside)
            outputs = self.decoder_step(
                self, next_token, attention_mask_padded, last_positions, kv_cache, adapter_indices
            )

            new_logits = outputs["logits"][:, -1, :]
            new_carry = (outputs["kv_cache"], rng, generated_ids, attention_mask_padded, last_positions, new_logits)
            return new_carry, logits if return_scores else None

        # Initial carry state
        initial_carry = (
            outputs["kv_cache"],
            rng,
            generated_ids_buf,
            attention_mask_padded,
            last_positions,
            outputs["logits"][:, -1, :],
        )

        # Run scan loop (replaces the Python for loop)
        final_carry, logits_seq = jax.lax.scan(scan_fn, initial_carry, xs=None, length=max_new_tokens)

        # Unpack final results
        kv_cache_final, rng_final, generated_ids, attention_mask_final, last_pos_final, logits_final = final_carry

        # Use logits_seq for scores if requested
        if return_scores:
            scores = [logits_seq[i] for i in range(logits_seq.shape[0])]

        return GenerateResult(
            generated_ids=generated_ids[:, : prompt_length + max_new_tokens],
            stop_reasons=stop_reasons,
            scores=scores,
        )
