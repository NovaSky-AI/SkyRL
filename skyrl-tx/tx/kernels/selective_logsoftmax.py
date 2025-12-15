"""Selective log-softmax kernel for JAX."""

import jax
import jax.numpy as jnp


@jax.jit
def selective_log_softmax_jax(
    logits: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    """
    Compute log_softmax(logits)[target_ids] efficiently.

    Args:
        logits: [B, T, V] logits tensor
        target_ids: [B, T] target token indices

    Returns:
        [B, T] log probabilities at target indices
    """
    log_sum_exp = jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    target_logits = jnp.take_along_axis(logits, target_ids[..., None], axis=-1)
    return (target_logits - log_sum_exp).squeeze(-1)
