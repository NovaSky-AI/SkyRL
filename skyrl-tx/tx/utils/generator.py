"""Generator mixin for autoregressive text generation with KV caching."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class KVCache:
    """Key-value cache for all layers, each entry in the list corresponds to one layer."""

    keys: list[jax.Array]
    values: list[jax.Array]


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
        stop_tokens: set[int] | None = None,
    ) -> GenerateResult:
        """Generate text autoregressively with KV caching.

        Returns:
            GenerateResult containing generated_ids, stop_reasons, and optionally scores.
        """
        rng = jax.random.PRNGKey(seed)
        generated_ids = input_ids
        scores = [] if return_scores else None
        batch_size = input_ids.shape[0]
        stop_reasons = ["length"] * batch_size

        # Prefill: process full prompt
        positions = compute_positions(attention_mask)
        outputs = self(input_ids, attention_mask=attention_mask, positions=positions, adapter_indices=adapter_indices)

        # Keep track of only the last position for decoding
        last_positions = positions[:, -1:]

        # Decode: generate tokens one at a time
        for step in range(max_new_tokens):
            rng, sample_key = jax.random.split(rng)
            logits = outputs["logits"][:, -1, :]

            if return_scores:
                scores.append(logits)

            next_token = sample_token(logits, temperature=temperature, key=sample_key)
            generated_ids = jnp.concatenate([generated_ids, next_token], axis=1)

            # Check if any sequence hit a stop token
            if stop_tokens is not None:
                for i, reason in enumerate(stop_reasons):
                    if reason == "length" and int(next_token[i, 0]) in stop_tokens:
                        stop_reasons[i] = "stop"

            # Early exit if all sequences are finished
            if all(reason == "stop" for reason in stop_reasons):
                break

            if step < max_new_tokens - 1:
                attention_mask = jnp.concatenate([attention_mask, jnp.ones_like(next_token)], axis=1)
                # Increment position for the new token
                last_positions = last_positions + 1
                outputs = self(
                    next_token,
                    attention_mask=attention_mask,
                    positions=last_positions,
                    kv_cache=outputs["kv_cache"],
                    adapter_indices=adapter_indices,
                )

        return GenerateResult(
            generated_ids=generated_ids,
            stop_reasons=stop_reasons,
            scores=scores,
        )
