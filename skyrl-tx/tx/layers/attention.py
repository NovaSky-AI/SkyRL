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

    Uses cuDNN flash attention on GPU for causal attention (training/prefill),
    reducing memory from O(seq_len^2) to O(seq_len).

    Falls back to mask-based attention for decode (is_causal=False) or CPU/TPU.
    Decode doesn't benefit from flash attention since attention is already O(seq_len).

    Args:
        q: Query tensor of shape [batch, q_len, num_heads, head_dim]
        k: Key tensor of shape [batch, kv_len, num_kv_heads, head_dim]
        v: Value tensor of shape [batch, kv_len, num_kv_heads, head_dim]
        attention_mask: Mask of shape [batch, kv_len] where 1 = valid, 0 = masked.
            Sequences must be right-padded (valid tokens first, then padding).
        is_causal: Whether this is causal (training/prefill) or non-causal (decode)
        head_dim: Dimension of each attention head (for scaling)

    Returns:
        Attention output of shape [batch, q_len, num_heads, head_dim]
    """
    scale = 1.0 / head_dim**0.5

    # Decode: use mask-based attention (flash attention provides minimal benefit
    # for single-token queries since attention is already O(seq_len) not O(seq_len^2))
    # TODO(haochen): enable flash attention for TPUs.
    if not is_causal or jax.default_backend() != "gpu":
        return jax.nn.dot_product_attention(
            q, k, v, scale=scale, mask=attention_mask[:, None, None, :].astype(bool), is_causal=is_causal
        )

    # Causal attention on GPU: use cuDNN flash attention
    seq_lengths = attention_mask.sum(axis=1).astype(jnp.int32)

    return jax.nn.dot_product_attention(
        q,
        k,
        v,
        scale=scale,
        is_causal=True,
        query_seq_lengths=seq_lengths,
        key_value_seq_lengths=seq_lengths,
        implementation="cudnn",
    )
