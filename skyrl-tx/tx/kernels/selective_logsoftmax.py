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
    """Compute log_softmax(logits)[target_ids] efficiently."""
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    target_ids_flat = target_ids.reshape(-1)
    
    max_logits = jnp.max(logits_flat, axis=-1, keepdims=True)
    exp_sum = jnp.sum(jnp.exp(logits_flat - max_logits), axis=-1, keepdims=True)
    log_sum_exp = max_logits + jnp.log(exp_sum)
    
    batch_indices = jnp.arange(B * T)
    target_logits = logits_flat[batch_indices, target_ids_flat]
    target_logprobs = target_logits - log_sum_exp.squeeze(-1)
    
    return target_logprobs.reshape(B, T)
