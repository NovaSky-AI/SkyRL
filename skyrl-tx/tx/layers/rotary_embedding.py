"""Rotary Position Embeddings (RoPE) implementation."""

import math
from typing import Any, Callable

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


def apply_rope_interleave(inputs: jax.Array, position_ids: jax.Array, head_dim: int, theta: float) -> jax.Array:
    """Apply interleaved Rotary Position Embeddings (RoPE).

    Args:
        inputs: Input tensor of shape [B, T, num_heads, head_dim]
        position_ids: Position indices of shape [B, T]
        head_dim: Dimension of each attention head
        theta: Base for the geometric progression (rope_theta)

    Returns:
        Tensor with RoPE applied, same shape as inputs, in grouped order
    """
    fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
    timescale = jnp.pow(theta, fraction)

    x = (position_ids[..., None] / timescale[None, None, :])[..., None, :]
    sin, cos = jnp.sin(x), jnp.cos(x)

    x1 = inputs[..., ::2]
    x2 = inputs[..., 1::2]

    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1).astype(inputs.dtype)


def get_rope(
    head_dim: int,
    rope_theta: float,
    rope_scaling: dict[str, Any] | None = None,
) -> tuple[Callable[[jax.Array, jax.Array], jax.Array], float]:
    """Factory function to create a rotary embedding function.

    Args:
        head_dim: Dimension of each attention head.
        rope_theta: Base for the geometric progression.
        rope_scaling: Optional dict with scaling configuration. The "type" or
            "rope_type" field determines the RoPE variant to use.

    Returns:
        A tuple of (rope_fn, mscale) where rope_fn takes (inputs, positions)
        and returns RoPE-applied outputs, and mscale is the attention magnitude
        scale factor for YaRN-style scaling.
    """
    rope_type = "default"
    mscale = 1.0

    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type", "default")

        if rope_type != "default":
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = rope_scaling["factor"]
            if mscale_all_dim and scaling_factor > 1.0:
                mscale = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0

    if rope_type in ("deepseek_yarn", "yarn"):

        def rope_fn(inputs: jax.Array, positions: jax.Array) -> jax.Array:
            return apply_rope_interleave(inputs, positions, head_dim, rope_theta)

    elif rope_type == "default":

        def rope_fn(inputs: jax.Array, positions: jax.Array) -> jax.Array:
            return apply_rope(inputs, positions, head_dim, rope_theta)

    else:
        raise ValueError(f"Unsupported rope_type: {rope_type}")

    return rope_fn, mscale
