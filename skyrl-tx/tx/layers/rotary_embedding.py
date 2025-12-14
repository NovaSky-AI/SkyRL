"""Rotary Position Embeddings (RoPE) implementation."""

import jax
from jax import numpy as jnp


def apply_rope(inputs: jax.Array, position_ids: jax.Array, head_dim: int, theta: float) -> jax.Array:
    """Apply Rotary Position Embeddings (RoPE).

    Args:
        inputs: Input tensor of shape [B, T, num_heads, head_dim]
        position_ids: Position indices of shape [B, T]
        head_dim: Dimension of each attention head
        theta: Base for the geometric progression (rope_theta)

    Returns:
        Tensor with RoPE applied, same shape as inputs
    """
    fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
    timescale = jnp.pow(theta, fraction)
    x = (position_ids[..., None] / timescale[None, None, :])[..., None, :]
    sin, cos = jnp.sin(x), jnp.cos(x)
    a, b = jnp.split(inputs, 2, axis=-1)
    return jnp.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1).astype(inputs.dtype)
