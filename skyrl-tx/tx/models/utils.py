"""Utility functions for model forward passes with stacked decoder layers.

This module provides a unified forward_layers function that works for both training
(with gradient checkpointing) and inference. The key insight is that jax.checkpoint
is a no-op when not computing gradients, so we can use the same scan-based code path.

Prerequisites:
- Layers must be created with nnx.vmap (stacked weights)
- KVCache must use stacked format: (num_layers, batch, seq, heads, dim)
"""

from typing import Callable

from flax import nnx
import jax
from jax import numpy as jnp

from tx.utils.generator import KVCache


def create_stacked_layers(
    create_layer_fn: Callable[[nnx.Rngs], nnx.Module],
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
) -> tuple[jax.Array, list[jax.Array], KVCache]:
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
        kv_cache: Optional KV cache for decode mode (None for prefill).
        output_hidden_states: Whether to return intermediate hidden states.
        gradient_checkpointing: Whether to use gradient checkpointing.

    Returns:
        Tuple of (final_hidden_states, all_hidden_states, kv_cache).
    """
    assert num_layers > 0, "num_layers must be positive"

    layer_graphdef, layer_state = nnx.split(layers)
    is_decode = kv_cache is not None

    def body_fn(hs, xs):
        # Unpack xs based on mode (structure differs between prefill and decode)
        if is_decode:
            layer_idx, layer_k, layer_v = xs
            layer_kv = (layer_k, layer_v)
        else:
            layer_idx = xs
            layer_kv = None

        # Reconstruct layer module from stacked weights
        layer = nnx.merge(layer_graphdef, jax.tree.map(lambda x: x[layer_idx], layer_state))

        new_hs, (k, v) = layer(
            hs,
            attention_mask=attention_mask,
            positions=positions,
            adapter_indices=adapter_indices,
            kv_cache=layer_kv,
        )

        hs_output = new_hs if output_hidden_states else None
        return new_hs, (hs_output, k, v)

    if gradient_checkpointing:
        body_fn = jax.checkpoint(body_fn)

    # Prepare scan inputs: in decode mode, pass per-layer caches via xs
    # Scan automatically slices along axis 0, so each iteration gets one layer's cache
    layer_indices = jnp.arange(num_layers)
    xs = (layer_indices, kv_cache.keys, kv_cache.values) if is_decode else layer_indices

    final_hs, (all_hs, all_keys, all_values) = jax.lax.scan(body_fn, hidden_states, xs)

    # [embed, layer0_out, ..., layer(N-2)_out]; final layer output gets normed by caller
    all_hidden_states = [hidden_states] + list(all_hs[:-1]) if output_hidden_states else []

    if is_decode:
        # Decode mode: scan stacked the per-layer updated caches into (num_layers, ...)
        new_kv_cache = KVCache(
            keys=all_keys,
            values=all_values,
            cache_position=kv_cache.cache_position + positions.shape[1],
        )
    else:
        # Prefill mode: build cache from collected k,v outputs
        new_kv_cache = KVCache.from_layer_outputs(all_keys, all_values, attention_mask)

    return final_hs, all_hidden_states, new_kv_cache
