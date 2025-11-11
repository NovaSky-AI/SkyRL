"""Shared layers used across multiple model architectures."""

from flax import nnx
import jax
from jax import numpy as jnp

from tx.layers.lora import LoRALinear
from tx.layers.util import Param


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization.

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, size: int, *, eps: float = 1e-6, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.eps = eps
        self.weight = Param(
            size, dtype=dtype, kernel_init=nnx.with_partitioning(nnx.initializers.normal(), jax.P(None)), rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.weight * x / rms


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


class SwiGLUMLP(nnx.Module):
    """SwiGLU Feed-Forward Network used in LLaMA and Qwen models.

    This implements the gated feed-forward network:
    FFN_SwiGLU(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension of the FFN
        max_lora_adapters: Maximum number of LoRA adapters
        max_lora_rank: Maximum rank for LoRA adapters
        dtype: Data type for computations
        rngs: Random number generators
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        max_lora_adapters: int = 0,
        max_lora_rank: int = 8,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ) -> None:
        # Gate projection (for gating mechanism)
        self.gate_proj = LoRALinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, "tp")),
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            rngs=rngs,
        )

        # Up projection
        self.up_proj = LoRALinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, "tp")),
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            rngs=rngs,
        )

        # Down projection
        self.down_proj = LoRALinear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P("tp", None)),
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, adapter_indices: jax.Array | None = None) -> jax.Array:
        gate_out = self.gate_proj(x, adapter_indices)
        up_out = self.up_proj(x, adapter_indices)
        return self.down_proj(nnx.silu(gate_out) * up_out, adapter_indices)
