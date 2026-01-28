"""Rotary Position Embeddings (RoPE) implementation."""

import math
from typing import Any, Callable

import jax
from jax import numpy as jnp


def apply_rope(
    inputs: jax.Array, position_ids: jax.Array, head_dim: int, theta: float, interleave: bool = False
) -> jax.Array:
    """Apply Rotary Position Embeddings (RoPE).

    Args:
        inputs: Input tensor of shape [B, T, num_heads, head_dim]
        position_ids: Position indices of shape [B, T]
        head_dim: Dimension of each attention head
        theta: Base for the geometric progression (rope_theta)
        interleave: If True, use interleaved slicing (x[..., ::2], x[..., 1::2])
            instead of splitting the last dimension in half.

    Returns:
        Tensor with RoPE applied, same shape as inputs
    """
    fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
    timescale = jnp.pow(theta, fraction)
    x = (position_ids[..., None] / timescale[None, None, :])[..., None, :]
    sin, cos = jnp.sin(x), jnp.cos(x)

    if interleave:
        a, b = inputs[..., ::2], inputs[..., 1::2]
    else:
        a, b = jnp.split(inputs, 2, axis=-1)

    return jnp.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1).astype(inputs.dtype)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


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
        A tuple of (rotary_emb, mscale) where rotary_emb takes (inputs, positions)
        and returns RoPE-applied outputs, and mscale is the attention magnitude
        scale factor for YaRN-style scaling.
    """
    rope_scaling = rope_scaling or {}
    rope_type = rope_scaling.get("rope_type", "default")

    match rope_type:
        case "deepseek_yarn":
            mscale = yarn_get_mscale(rope_scaling["factor"], rope_scaling["mscale_all_dim"])

            def rotary_emb(inputs: jax.Array, positions: jax.Array) -> jax.Array:
                return apply_rope(inputs, positions, head_dim, rope_theta, interleave=True)

        case "default":
            mscale = 1.0

            def rotary_emb(inputs: jax.Array, positions: jax.Array) -> jax.Array:
                return apply_rope(inputs, positions, head_dim, rope_theta)

        case _:
            raise ValueError(f"Unsupported rope_type: {rope_type}")

    return rotary_emb, mscale
