"""Shared attention utilities for transformer models."""

import jax
import jax.numpy as jnp
from jax.sharding import get_abstract_mesh

# cuDNN flash attention supported dtypes
# https://github.com/jax-ml/jax/blob/8b1f782540f71fbe230a2dccd331975faafc6c83/jax/_src/cudnn/fused_attention_stablehlo.py#L290
_CUDNN_SUPPORTED_DTYPES = (jnp.float16, jnp.bfloat16, jnp.float8_e4m3fn, jnp.float8_e5m2)


def _repeat_kv_for_gqa(x: jax.Array, num_heads: int) -> jax.Array:
    """Repeat KV heads to match query heads for manual attention math."""
    kv_heads = x.shape[2]
    if kv_heads == num_heads:
        return x
    if num_heads % kv_heads != 0:
        raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={kv_heads}")
    return jnp.repeat(x, num_heads // kv_heads, axis=2)


def _ring_attention_streaming(
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

    # [B, Tq, H, D] -> [B, H, Tq, D]
    qh = jnp.swapaxes(q, 1, 2).astype(jnp.float32)
    k_block = _repeat_kv_for_gqa(k, q.shape[2])
    v_block = _repeat_kv_for_gqa(v, q.shape[2])
    mask_block = attention_mask

    B, H, Tq, D = qh.shape
    m = jnp.full((B, H, Tq), -jnp.inf, dtype=jnp.float32)
    l = jnp.zeros((B, H, Tq), dtype=jnp.float32)
    acc = jnp.zeros((B, H, Tq, D), dtype=jnp.float32)
    neg_large = jnp.array(-1e30, dtype=jnp.float32)

    # source i -> dest (i + 1) % cp
    perm = [(i, (i + 1) % cp) for i in range(cp)]

    for step in range(cp):
        source_shard = (axis_idx - step) % cp
        key_positions = source_shard * local_len + jnp.arange(local_len, dtype=jnp.int32)

        kh = jnp.swapaxes(k_block, 1, 2).astype(jnp.float32)  # [B, H, Tk, D]
        vh = jnp.swapaxes(v_block, 1, 2).astype(jnp.float32)  # [B, H, Tk, D]
        scores = jnp.einsum("bhtd,bhsd->bhts", qh, kh) * scale

        causal = key_positions[None, None, None, :] <= positions[:, None, :, None]
        padding = mask_block[:, None, None, :].astype(bool)
        valid = causal & padding
        scores = jnp.where(valid, scores, neg_large)

        m_block = jnp.max(scores, axis=-1)
        m_new = jnp.maximum(m, m_block)
        prev_scale = jnp.where(jnp.isfinite(m), jnp.exp(m - m_new), 0.0)
        p = jnp.exp(scores - m_new[..., None])
        p = jnp.where(valid, p, 0.0)
        l_new = prev_scale * l + jnp.sum(p, axis=-1)
        acc_new = prev_scale[..., None] * acc + jnp.einsum("bhts,bhsd->bhtd", p, vh)
        m, l, acc = m_new, l_new, acc_new

        if step < cp - 1:
            k_block = jax.lax.ppermute(k_block, axis_name="cp", perm=perm)
            v_block = jax.lax.ppermute(v_block, axis_name="cp", perm=perm)
            mask_block = jax.lax.ppermute(mask_block, axis_name="cp", perm=perm)

    out = jnp.where(l[..., None] > 0, acc / jnp.maximum(l[..., None], 1e-9), 0.0)
    return jnp.swapaxes(out.astype(q.dtype), 1, 2)  # [B, Tq, H, D]


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

    # CP path: stream KV blocks around the ring and accumulate online softmax.
    # For decode (q_len == 1), the causal check against positions is equivalent to
    # attending all valid cached keys.
    if cp > 1 and (is_causal or q.shape[1] == 1):
        return _ring_attention_streaming(q, k, v, attention_mask, positions, scale)

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
