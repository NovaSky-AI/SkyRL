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

    Uses cuDNN flash attention on GPU for right-padded sequences (O(seq) memory).
    Falls back to mask-based attention for left-padded sequences or CPU/TPU.

    In practice:
    - Training uses right-padding → cuDNN flash attention
    - Generation uses left-padding → mask-based fallback (both prefill and decode)

    TODO: Support left-padded sequences with cuDNN using BOTTOM_RIGHT diagonal alignment
    when JAX exposes this option. See:
    https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html

    Args:
        q: Query tensor of shape [B, T, num_heads, head_dim]
        k: Key tensor of shape [B, S, num_kv_heads, head_dim]
        v: Value tensor of shape [B, S, num_kv_heads, head_dim]
        attention_mask: Mask of shape [B, S] where 1 = valid, 0 = masked
        is_causal: Whether this is prefill (causal) or decode (non-causal)
        head_dim: Dimension of each attention head (for scaling)

    Returns:
        Attention output of shape [B, T, num_heads, head_dim]
    """
    scale = 1.0 / head_dim**0.5

    if jax.default_backend() != 'gpu':
        return jax.nn.dot_product_attention(
            q, k, v, scale=scale, mask=attention_mask[:, None, None, :].astype(bool), is_causal=is_causal
        )

    # Check if right-padded (all batches have 1 at position 0)
    # Right-padded: [1,1,1,0,0], Left-padded: [0,0,1,1,1]
    is_right_padded = attention_mask[:, 0].min() == 1

    def cudnn_path():
        seq_lengths = attention_mask.sum(axis=1).astype(jnp.int32)
        query_seq_lengths = seq_lengths if is_causal else jnp.ones_like(seq_lengths)
        return jax.nn.dot_product_attention(
            q, k, v, scale=scale, is_causal=is_causal,
            query_seq_lengths=query_seq_lengths,
            key_value_seq_lengths=seq_lengths,
            implementation='cudnn',
        )

    def mask_path():
        return jax.nn.dot_product_attention(
            q, k, v, scale=scale, mask=attention_mask[:, None, None, :].astype(bool), is_causal=is_causal
        )

    return jax.lax.cond(is_right_padded, cudnn_path, mask_path)
