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


def apply_top_k(logits: jax.Array, top_k: int) -> jax.Array:
    """Apply top-k filtering to logits."""
    if top_k <= 0:
        if top_k == 0:
            # top_k=0 means no tokens are allowed, set all to -inf
            return jnp.full_like(logits, -jnp.inf)
        return logits
    
    # Get top-k values and their indices
    top_k_logits, _ = jax.lax.top_k(logits, top_k)
    # Set threshold to the k-th largest value
    threshold = top_k_logits[..., -1, None]
    # Set all values below threshold to -inf
    return jnp.where(logits < threshold, -jnp.inf, logits)


def apply_top_p(logits: jax.Array, top_p: float) -> jax.Array:
    """Apply top-p (nucleus) filtering to logits."""
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        # top_p=0 means no probability mass is allowed, set all to -inf
        return jnp.full_like(logits, -jnp.inf)
    
    # Sort logits in descending order
    sorted_logits = jnp.sort(logits, axis=-1)[..., ::-1]
    # Compute softmax probabilities
    sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
    # Compute cumulative probabilities
    cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)
    # Find cutoff point where cumulative probability exceeds top_p
    cutoff = jnp.sum(cumsum_probs <= top_p, axis=-1, keepdims=True)
    # Get threshold value
    threshold = jnp.take_along_axis(sorted_logits, cutoff, axis=-1)
    # Set all values below threshold to -inf
    return jnp.where(logits < threshold, -jnp.inf, logits)


def sample_token(
    logits: jax.Array, 
    *, 
    temperature: float, 
    key: jax.Array,
    top_k: int = -1,
    top_p: float = 1.0,
) -> jax.Array:
    """Sample next token from logits using temperature, top_k, and top_p."""
    # Apply top_k filtering if specified
    if top_k > 0:
        logits = apply_top_k(logits, top_k)
    
    # Apply top_p filtering if specified
    if top_p < 1.0:
        logits = apply_top_p(logits, top_p)
    
    # Sample with temperature
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
        stop_tokens: list[int] | None = None,
        top_k: int = -1,
        top_p: float = 1.0,
    ) -> GenerateResult:
        """Generate text autoregressively with KV caching.

        Returns:
            GenerateResult containing generated_ids, stop_reasons, and optionally scores.
        """
        rng = jax.random.PRNGKey(seed)
        generated_ids = input_ids
        scores = [] if return_scores else None
        stop_reasons = ["length"] * input_ids.shape[0]
        
        # Convert stop_tokens to a set for efficient lookup
        stop_token_set = set(stop_tokens) if stop_tokens else set()

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

            next_token = sample_token(
                logits, 
                temperature=temperature, 
                key=sample_key,
                top_k=top_k,
                top_p=top_p,
            )
            generated_ids = jnp.concatenate([generated_ids, next_token], axis=1)

            # Check for stop tokens
            if stop_token_set:
                for batch_idx in range(next_token.shape[0]):
                    if stop_reasons[batch_idx] == "length" and next_token[batch_idx, 0].item() in stop_token_set:
                        stop_reasons[batch_idx] = "stop"

            # Early termination if all sequences have stopped
            if all(reason != "length" for reason in stop_reasons):
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
