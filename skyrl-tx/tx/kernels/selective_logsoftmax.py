"""Selective log-softmax kernel for JAX."""

import jax
import jax.numpy as jnp
import jax.scipy.special


@jax.jit
def selective_log_softmax_jax(
    logits: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    """
    Compute log_softmax(logits)[target_ids] efficiently.

    Avoids materializing the full log_softmax array by only computing
    the log probabilities for the target indices.

    Args:
        logits: [B, T, V] logits tensor
        target_ids: [B, T] target token indices

    Returns:
        [B, T] log probabilities at target indices
    """
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    target_ids_flat = target_ids.reshape(-1)

    # Compute log_sum_exp for numerical stability
    log_sum_exp = jax.scipy.special.logsumexp(logits_flat, axis=-1, keepdims=True)

    # Extract target logits and compute log probabilities
    batch_indices = jnp.arange(B * T)
    target_logits = logits_flat[batch_indices, target_ids_flat]
    target_logprobs = target_logits - log_sum_exp.squeeze(-1)

    return target_logprobs.reshape(B, T)
