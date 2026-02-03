"""Utility functions for model forward passes with stacked decoder layers.

This module provides:
- create_stacked_layers: Create decoder layers with stacked weights
- forward_layers: Unified forward pass using scan (skips KV cache during training)

Prerequisites:
- Layers must be created with create_stacked_layers (stacked weights)
- KVCache must use stacked format: (num_layers, batch, seq, heads, dim)
"""

from typing import Callable

from flax import nnx
import jax

from tx.utils.generator import KVCache


def create_stacked_layers(
    create_layer_fn: Callable[[nnx.Rngs], nnx.Module],
    num_layers: int,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Create stacked decoder layers by creating individual layers and stacking their parameters.

    This creates a single module object where all parameters have shape (num_layers, ...).
    This enables efficient scanning over layers without runtime stacking.

    Note: We avoid using nnx.vmap for layer creation because vmap breaks eager sharding,
    causing ~4x memory overhead. Instead, we create layers individually (which respects
    eager sharding) and then stack their parameters with jnp.stack.

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
    import warnings
    from functools import partial

    import jax.numpy as jnp
    import jax.random
    from jax.sharding import NamedSharding, PartitionSpec

    # Split the RNG key to get unique keys for each layer
    base_key = rngs.params()
    layer_keys = jax.random.split(base_key, num_layers)

    # Get the current mesh for sharding
    mesh = jax.sharding.get_mesh()

    # Create all layers individually - this respects eager sharding
    layers = [create_layer_fn(nnx.Rngs(layer_keys[i])) for i in range(num_layers)]

    # Get graphdef from first layer (all layers have same structure)
    graphdef, first_state = nnx.split(layers[0])

    # Extract flattened states from all layers
    states = [nnx.split(layer)[1] for layer in layers]
    del layers

    flat_states = [jax.tree_util.tree_flatten(s)[0] for s in states]
    treedef = jax.tree_util.tree_flatten(states[0])[1]
    del states

    # Stack each parameter array using jit with donate_argnums for memory efficiency.
    # This tells XLA to try to reuse input buffers for the output, reducing peak memory.
    stacked_flat = []
    for i in range(len(flat_states[0])):
        # Get arrays for this parameter across all layers
        arrays = [flat_states[j][i] for j in range(num_layers)]

        # Get original sharding spec and extend it for the stacked dimension
        original_sharding = arrays[0].sharding
        if hasattr(original_sharding, "spec"):
            original_spec = original_sharding.spec
            # Prepend None for the new layer dimension
            new_spec = PartitionSpec(None, *original_spec)
            new_sharding = NamedSharding(mesh, new_spec)

            # Use jit with donate_argnums and out_shardings for memory-efficient stacking.
            # The donation hints help XLA manage memory better during the stacking operation.
            @partial(jax.jit, donate_argnums=tuple(range(num_layers)), out_shardings=new_sharding)
            def do_stack(*arrs):
                return jnp.stack(arrs, axis=0)

            # Suppress donation warnings since we expect some buffers can't be donated
            # (stacking changes array shapes so direct donation isn't always possible)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Some donated buffers were not usable")
                stacked = do_stack(*arrays)
        else:
            stacked = jnp.stack(arrays, axis=0)
        stacked_flat.append(stacked)
        del arrays

    del flat_states

    # Reconstruct the state tree with stacked arrays
    stacked_state = jax.tree_util.tree_unflatten(treedef, stacked_flat)
    del stacked_flat

    # Merge back into a module with stacked parameters
    return nnx.merge(graphdef, stacked_state)


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
    is_training: bool = False,
) -> tuple[jax.Array, list[jax.Array], KVCache | None]:
    """Unified forward pass through stacked decoder layers using scan.

    Args:
        layers: Stacked decoder layers (created with create_stacked_layers/nnx.vmap).
        hidden_states: Input hidden states of shape (batch, seq, hidden).
        num_layers: Number of decoder layers.
        attention_mask: Attention mask of shape (batch, seq).
        positions: Position indices of shape (batch, seq).
        adapter_indices: Optional LoRA adapter indices of shape (batch,).
        kv_cache: Optional KV cache for decode mode (None for prefill).
        output_hidden_states: Whether to return intermediate hidden states.
        gradient_checkpointing: Whether to use gradient checkpointing (training only).
        is_training: Whether in training mode. Skips KV cache to save memory.

    Returns:
        Tuple of (final_hidden_states, all_hidden_states, kv_cache).
        kv_cache is None when is_training=True.
    """
    assert num_layers > 0, "num_layers must be positive"

    layer_graphdef, layer_state = nnx.split(layers)
    is_decode = kv_cache is not None

    def body_fn(hs, xs):
        # Unpack xs: scan automatically slices the leading dimension of layer_state
        if is_decode:
            layer_params, layer_k, layer_v = xs
            layer_kv = (layer_k, layer_v)
        else:
            layer_params = xs
            layer_kv = None

        # Merge using the sliced params directly - no manual gather needed
        layer = nnx.merge(layer_graphdef, layer_params)
        new_hs, (k, v) = layer(
            hs,
            attention_mask=attention_mask,
            positions=positions,
            adapter_indices=adapter_indices,
            kv_cache=layer_kv,
        )
        hs_output = new_hs if output_hidden_states else None

        if is_training:
            # Avoid accumulating large KV tensors for training.
            k = v = None
        return new_hs, (hs_output, k, v)

    if gradient_checkpointing:
        body_fn = jax.checkpoint(body_fn)

    # Pass layer_state as xs so scan handles the slicing automatically.
    # This avoids capturing layer_state as a closure and manually gathering,
    # which causes slow XLA compilation with jax.checkpoint.
    xs = (layer_state, kv_cache.keys, kv_cache.values) if is_decode else layer_state

    final_hs, (all_hs, all_keys, all_values) = jax.lax.scan(body_fn, hidden_states, xs)

    # [embed, layer0_out, ..., layer(N-2)_out]; final layer output gets normed by caller
    all_hidden_states = [hidden_states] + list(all_hs[:-1]) if output_hidden_states else []

    if is_training:
        new_kv_cache = None
    elif is_decode:
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
