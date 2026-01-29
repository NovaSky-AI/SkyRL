"""Utility functions for model forward passes with stacked decoder layers.

This module provides a unified forward_layers function that works for both training
(with gradient checkpointing) and inference. The key insight is that jax.checkpoint
is a no-op when not computing gradients, so we can use the same scan-based code path.

Prerequisites:
- Layers must be created with nnx.vmap (stacked weights)
- KVCache must use stacked format: (num_layers, batch, seq, heads, dim)
"""

from typing import TypeVar

from flax import nnx
import jax
from jax import numpy as jnp

from tx.utils.generator import KVCache

T = TypeVar("T", bound=nnx.Module)


def create_stacked_layers(
    create_layer_fn: callable,
    num_layers: int,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Create stacked decoder layers using nnx.vmap.

    This creates a single module object where all parameters have shape (num_layers, ...).
    This enables efficient scanning over layers without runtime stacking.

    Args:
        create_layer_fn: Function that takes rngs and returns a single layer module.
        num_layers: Number of layers to create.
        rngs: Random number generators for initialization.

    Returns:
        A single module with stacked parameters.

    Example:
        >>> def create_layer(rngs):
        ...     return Llama3DecoderLayer(config, dtype=dtype, rngs=rngs)
        >>> layers = create_stacked_layers(create_layer, config.num_hidden_layers, rngs)
        >>> # layers.self_attn.q_proj.kernel.shape == (num_layers, hidden, head_dim*num_heads)
    """

    @nnx.split_rngs(splits=num_layers)
    @nnx.vmap(in_axes=(0,), out_axes=0)
    def vmapped_create(rngs: nnx.Rngs):
        return create_layer_fn(rngs)

    return vmapped_create(rngs)


def forward_layers(
    layers: nnx.Module,
    hidden_states: jax.Array,
    num_layers: int,
    *,
    attention_mask: jax.Array,
    positions: jax.Array,
    adapter_indices: jax.Array | None,
    kv_cache: KVCache | None,
    output_hidden_states: bool,
    gradient_checkpointing: bool,
) -> tuple[jax.Array, list[jax.Array], KVCache | None]:
    """Unified forward pass through stacked decoder layers.

    Uses jax.lax.scan for both training and inference. When gradient_checkpointing=True,
    wraps the body function with jax.checkpoint. This is a no-op during inference
    (when not computing gradients), so we can use a single code path.

    Args:
        layers: Stacked decoder layers (created with create_stacked_layers/nnx.vmap).
        hidden_states: Input hidden states of shape (batch, seq, hidden).
        num_layers: Number of decoder layers.
        attention_mask: Attention mask of shape (batch, seq).
        positions: Position indices of shape (batch, seq).
        adapter_indices: Optional LoRA adapter indices of shape (batch,).
        kv_cache: Optional KV cache with stacked keys/values.
        output_hidden_states: Whether to return intermediate hidden states.
        gradient_checkpointing: Whether to use gradient checkpointing.

    Returns:
        Tuple of:
        - Final hidden states of shape (batch, seq, hidden)
        - List of intermediate hidden states (if output_hidden_states=True)
        - Updated KV cache (if kv_cache was provided)
    """
    if num_layers == 0:
        return hidden_states, [], kv_cache

    # Split layers into graph definition and stacked state
    layer_graphdef, layer_state = nnx.split(layers)

    # Prepare stacked KV cache
    stacked_kv: tuple[jax.Array, jax.Array] | None = None
    if kv_cache is not None:
        stacked_kv = (kv_cache.keys, kv_cache.values)

    def body_fn(carry, layer_idx):
        hs, kv = carry

        # Extract this layer's weights by indexing into stacked state
        layer_weights = jax.tree.map(lambda x: x[layer_idx], layer_state)
        layer = nnx.merge(layer_graphdef, layer_weights)

        # Get this layer's KV cache slice
        layer_kv = None
        if kv is not None:
            layer_kv = (kv[0][layer_idx], kv[1][layer_idx])

        # Forward through layer
        new_hs, (k, v) = layer(
            hs,
            attention_mask=attention_mask,
            positions=positions,
            adapter_indices=adapter_indices,
            kv_cache=layer_kv,
        )

        # Update stacked KV cache
        new_kv = kv
        if kv is not None:
            new_kv = (
                kv[0].at[layer_idx].set(k),
                kv[1].at[layer_idx].set(v),
            )

        # Return updated carry and output for this iteration
        output = hs if output_hidden_states else None
        return (new_hs, new_kv), output

    # Apply gradient checkpointing if requested
    if gradient_checkpointing:
        body_fn = jax.checkpoint(body_fn)

    # Scan over layer indices
    (final_hs, final_kv), all_hs = jax.lax.scan(
        body_fn,
        (hidden_states, stacked_kv),
        jnp.arange(num_layers),
    )

    # Collect hidden states if requested
    all_hidden_states: list[jax.Array] = []
    if output_hidden_states:
        # all_hs has shape (num_layers, batch, seq, hidden)
        # We want [input, layer0_out, layer1_out, ...] excluding final (it gets normed)
        all_hidden_states = [hidden_states] + [all_hs[i] for i in range(num_layers - 1)]

    # Reconstruct KVCache if it was provided
    new_kv_cache = None
    if kv_cache is not None and final_kv is not None:
        new_kv_cache = KVCache(
            keys=final_kv[0],
            values=final_kv[1],
            cache_position=kv_cache.cache_position,
        )

    return final_hs, all_hidden_states, new_kv_cache
