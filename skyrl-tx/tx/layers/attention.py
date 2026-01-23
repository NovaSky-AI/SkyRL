"""Shared attention utilities for transformer models."""

import jax
import jax.numpy as jnp


def dot_product_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_mask: jax.Array,
    is_causal: bool,
    head_dim: int,
) -> jax.Array:
    """Compute dot-product attention with automatic backend selection.

    Uses cuDNN on GPU for memory-efficient attention. Falls back to XLA for CPU/TPU.

    Args:
        q: Query tensor of shape [batch, q_len, num_heads, head_dim]
        k: Key tensor of shape [batch, kv_len, num_kv_heads, head_dim]
        v: Value tensor of shape [batch, kv_len, num_kv_heads, head_dim]
        attention_mask: Mask of shape [batch, kv_len] where 1 = valid, 0 = masked.
            Sequences must be right-padded (valid tokens first, then padding).
        is_causal: Whether to apply causal masking (for prefill/training)
        head_dim: Dimension of each attention head (for scaling)

    Returns:
        Attention output of shape [batch, q_len, num_heads, head_dim]
    """
    scale = 1.0 / head_dim**0.5

    if jax.default_backend() == "gpu":
        seq_lengths = attention_mask.sum(axis=1).astype(jnp.int32)
        return jax.nn.dot_product_attention(
            q,
            k,
            v,
            scale=scale,
            is_causal=is_causal,
            query_seq_lengths=seq_lengths,
            key_value_seq_lengths=seq_lengths,
            implementation="cudnn",
        )

    # CPU/TPU fallback
    return jax.nn.dot_product_attention(
        q, k, v, scale=scale, mask=attention_mask[:, None, None, :].astype(bool), is_causal=is_causal
    )
