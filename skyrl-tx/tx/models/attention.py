"""Shared attention utilities for transformer models."""

import jax
import jax.numpy as jnp


def _shift_sequences(x: jax.Array, shift_amounts: jax.Array) -> jax.Array:
    """Shift sequences along axis 1 by per-batch amounts.

    Args:
        x: Tensor of shape [B, S, ...]
        shift_amounts: Per-batch shift amounts [B]. Positive shifts left (for left -> right pad conversion).

    Returns:
        Shifted tensor with same shape as x.
    """
    S = x.shape[1]
    indices = (jnp.arange(S)[None, :] + shift_amounts[:, None]) % S
    # Broadcast indices to match x's shape for take_along_axis
    indices = indices.reshape(indices.shape + (1,) * (x.ndim - 2))
    indices = jnp.broadcast_to(indices, x.shape)
    return jnp.take_along_axis(x, indices, axis=1)


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
    reducing memory from O(S^2) to O(S). For left-padded sequences, temporarily
    shifts to right-padding before applying flash attention.

    Falls back to mask-based attention for decode (is_causal=False) or CPU/TPU.
    Decode doesn't benefit from flash attention since attention is already O(S).

    Args:
        q: Query tensor of shape [B, T, num_heads, head_dim]
        k: Key tensor of shape [B, S, num_kv_heads, head_dim]
        v: Value tensor of shape [B, S, num_kv_heads, head_dim]
        attention_mask: Mask of shape [B, S] where 1 = valid, 0 = masked
        is_causal: Whether this is causal (training/prefill) or non-causal (decode)
        head_dim: Dimension of each attention head (for scaling)

    Returns:
        Attention output of shape [B, T, num_heads, head_dim]
    """
    scale = 1.0 / head_dim**0.5

    # Decode: use mask-based attention (flash attention provides minimal benefit
    # for single-token queries since attention is already O(S) not O(S^2))
    if not is_causal or jax.default_backend() != 'gpu':
        return jax.nn.dot_product_attention(
            q, k, v, scale=scale, mask=attention_mask[:, None, None, :].astype(bool), is_causal=is_causal
        )

    # Causal attention on GPU: use cuDNN flash attention
    seq_lengths = attention_mask.sum(axis=1).astype(jnp.int32)

    # Check if right-padded (all batches have 1 at position 0)
    # Right-padded: [1,1,1,0,0], Left-padded: [0,0,1,1,1]
    is_right_padded = attention_mask[:, 0].min() == 1

    def cudnn_right_padded():
        return jax.nn.dot_product_attention(
            q, k, v, scale=scale, is_causal=True,
            query_seq_lengths=seq_lengths,
            key_value_seq_lengths=seq_lengths,
            implementation='cudnn',
        )

    def cudnn_left_padded():
        # Shift left-padded to right-padded for cuDNN compatibility
        shift = attention_mask.shape[1] - seq_lengths
        q_shifted = _shift_sequences(q, shift)
        k_shifted = _shift_sequences(k, shift)
        v_shifted = _shift_sequences(v, shift)

        out = jax.nn.dot_product_attention(
            q_shifted, k_shifted, v_shifted, scale=scale, is_causal=True,
            query_seq_lengths=seq_lengths,
            key_value_seq_lengths=seq_lengths,
            implementation='cudnn',
        )

        # Shift output back to left-padded
        return _shift_sequences(out, -shift)

    return jax.lax.cond(is_right_padded, cudnn_right_padded, cudnn_left_padded)
