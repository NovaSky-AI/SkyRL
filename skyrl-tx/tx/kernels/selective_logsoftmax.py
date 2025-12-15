"""Selective log-softmax kernel for JAX."""

import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=['chunk_size'])
def selective_log_softmax_jax(
    logits: jax.Array,
    target_ids: jax.Array,
    chunk_size: int = 4096,
) -> jax.Array:
    """
    Compute log_softmax(logits)[target_ids] efficiently.
    
    Avoids materializing the full log_softmax array by only computing
    the log probabilities for the target indices.
    
    Args:
        logits: [B, T, V] logits tensor
        target_ids: [B, T] target token indices
        chunk_size: unused, kept for API compatibility
        
    Returns:
        [B, T] log probabilities at target indices
    """
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    target_ids_flat = target_ids.reshape(-1)
    
    # Compute log_sum_exp for numerical stability
    max_logits = jnp.max(logits_flat, axis=-1, keepdims=True)
    exp_sum = jnp.sum(jnp.exp(logits_flat - max_logits), axis=-1, keepdims=True)
    log_sum_exp = max_logits + jnp.log(exp_sum)
    
    # Extract target logits and compute log probabilities
    batch_indices = jnp.arange(B * T)
    target_logits = logits_flat[batch_indices, target_ids_flat]
    target_logprobs = target_logits - log_sum_exp.squeeze(-1)
    
    return target_logprobs.reshape(B, T)
