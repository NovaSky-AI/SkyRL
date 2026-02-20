"""Shared attention utilities for transformer models."""

import jax
import jax.numpy as jnp
from jax.sharding import get_abstract_mesh

# cuDNN flash attention supported dtypes
# https://github.com/jax-ml/jax/blob/8b1f782540f71fbe230a2dccd331975faafc6c83/jax/_src/cudnn/fused_attention_stablehlo.py#L290
_CUDNN_SUPPORTED_DTYPES = (jnp.float16, jnp.bfloat16, jnp.float8_e4m3fn, jnp.float8_e5m2)


def _ring_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_mask: jax.Array,
    positions: jax.Array,
    scale: float,
) -> jax.Array:
    """Streaming causal attention with ring KV exchange via ppermute."""
    cp = get_abstract_mesh().shape.get("cp", 1)
    axis_idx = jax.lax.axis_index("cp")
    local_len = k.shape[1]

    # qh: [B, H, Tq, D]
    qh = jnp.transpose(q, (0, 2, 1, 3))

    # GQA handling: expand KV heads to match query heads.
    # k/v: [B, Tk, H_kv, D] -> [B, Tk, H, D]
    kv_repeat = q.shape[2] // k.shape[2]
    k_block = jnp.repeat(k, kv_repeat, axis=2)
    v_block = jnp.repeat(v, kv_repeat, axis=2)
    mask_block = attention_mask

    # Online softmax state (kept per [B, H, Tq]):
    # carry_max = running max score
    # denom     = running denominator sum(exp(score - carry_max))
    # acc = running numerator sum(exp(score - m) * value), shape [B, H, Tq, D]
    B, H, Tq, D = qh.shape
    carry_max = jnp.full((B, H, Tq), -jnp.inf, dtype=q.dtype)
    denom = jnp.zeros((B, H, Tq), dtype=q.dtype)
    acc = jnp.zeros((B, H, Tq, D), dtype=q.dtype)
    neg_large = jnp.array(jnp.finfo(q.dtype).min, dtype=q.dtype)

    # Ring exchange: source i -> destination (i + 1) % cp.
    perm = [(i, (i + 1) % cp) for i in range(cp)]

    for step in range(cp):
        source_shard = (axis_idx - step) % cp
        # Absolute token positions for the current KV block, shape [Tk].
        key_positions = source_shard * local_len + jnp.arange(local_len, dtype=jnp.int32)

        # vh:  [B, H, Tk, D]
        # kht: [B, H, D, Tk] (K transposed for Q @ K^T)
        kht = jnp.transpose(k_block, (0, 2, 3, 1))
        vh = jnp.transpose(v_block, (0, 2, 1, 3))
        scores = jnp.matmul(qh, kht) * scale

        # Mask invalid keys (future tokens + padding) before softmax update.
        causal = key_positions[None, None, None, :] <= positions[:, None, :, None]
        padding = mask_block[:, None, None, :].astype(bool)
        valid = causal & padding
        scores = jnp.where(valid, scores, neg_large)

        # Numerically stable online softmax merge:
        # merge previous state (carry_max, denom, acc) with current block scores/values.
        m_block = jnp.max(scores, axis=-1)
        carry_max_new = jnp.maximum(carry_max, m_block)
        prev_scale = jnp.where(jnp.isfinite(carry_max), jnp.exp(carry_max - carry_max_new), 0.0)
        p = jnp.exp(scores - carry_max_new[..., None])
        p = jnp.where(valid, p, 0.0)
        denom_new = prev_scale * denom + jnp.sum(p, axis=-1)
        acc_new = prev_scale[..., None] * acc + jnp.matmul(p, vh)
        carry_max, denom, acc = carry_max_new, denom_new, acc_new

        # Rotate KV/mask so the next iteration sees the next shard's block.
        if step < cp - 1:
            k_block = jax.lax.ppermute(k_block, axis_name="cp", perm=perm)
            v_block = jax.lax.ppermute(v_block, axis_name="cp", perm=perm)
            mask_block = jax.lax.ppermute(mask_block, axis_name="cp", perm=perm)

    # Final normalize and restore [B, Tq, H, D]
    out = jnp.where(
        denom[..., None] > 0,
        acc / jnp.maximum(denom[..., None], jnp.asarray(1e-9, dtype=denom.dtype)),
        0.0,
    )
    return jnp.transpose(out, (0, 2, 1, 3))


def default_positions(input_ids: jax.Array) -> jax.Array:
    """Build token positions from input token shape, with CP shard offset."""
    start, local_len = 0, input_ids.shape[1]
    cp = get_abstract_mesh().shape.get("cp", 1)
    if cp > 1:
        axis_idx = jax.lax.axis_index("cp")
        start = axis_idx * local_len
    return (start + jnp.arange(local_len, dtype=jnp.int32))[None, :]


def dot_product_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_mask: jax.Array,
    is_causal: bool,
    head_dim: int,
    *,
    positions: jax.Array,
    scale: float | None = None,
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
        head_dim: Dimension of each attention head (for scaling when scale is not provided)
        positions: Query positions, shape [batch, q_len], used for causal masking
        scale: Optional explicit scale factor for attention logits

    Returns:
        Attention output of shape [batch, q_len, num_heads, head_dim]
    """
    scale = scale if scale is not None else 1.0 / head_dim**0.5
    cp = get_abstract_mesh().shape.get("cp", 1)

    # TODO: constraints for running ring attention
    if cp > 1 and (is_causal or q.shape[1] == 1):
        return _ring_attention(q, k, v, attention_mask, positions, scale)

    if jax.default_backend() == "gpu" and q.dtype in _CUDNN_SUPPORTED_DTYPES:
        kv_seq_lengths = attention_mask.sum(axis=1).astype(jnp.int32)
        q_seq_lengths = jnp.minimum(kv_seq_lengths, q.shape[1])
        return jax.nn.dot_product_attention(
            q,
            k,
            v,
            scale=scale,
            is_causal=is_causal,
            query_seq_lengths=q_seq_lengths,
            key_value_seq_lengths=kv_seq_lengths,
            implementation="cudnn",
        )

    # CPU/TPU fallback
    return jax.nn.dot_product_attention(
        q, k, v, scale=scale, mask=attention_mask[:, None, None, :].astype(bool), is_causal=is_causal
    )
