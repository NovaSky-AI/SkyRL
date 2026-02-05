"""Utility functions for model forward passes with stacked decoder layers.

This module provides:
- create_stacked_layers: Create decoder layers with stacked weights
- forward_layers: Unified forward pass using scan (skips KV cache during training)

Prerequisites:
- Layers must be created with create_stacked_layers (stacked weights)
- KVCache must use stacked format: (num_layers, batch, seq, heads, dim)
"""

import logging
import subprocess
from typing import Callable

from flax import nnx
import jax

from tx.utils.generator import KVCache

logger = logging.getLogger(__name__)


def _log_mem(label: str):
    """Log GPU memory usage via nvidia-smi and JAX memory stats."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        nvidia_mem = max(int(x) for x in result.stdout.strip().split("\n"))
    except Exception:
        nvidia_mem = -1

    try:
        # Get JAX's view of memory usage
        devices = jax.devices()
        jax_mems = []
        for d in devices:
            stats = d.memory_stats()
            if stats:
                # bytes_in_use is the actual memory used by JAX arrays
                jax_mems.append(stats.get("bytes_in_use", 0) / 1024 / 1024)
        jax_mem = max(jax_mems) if jax_mems else -1
    except Exception:
        jax_mem = -1

    logger.info(f"[MEM] {label}: nvidia={nvidia_mem} MiB, jax={jax_mem:.1f} MiB")


def create_stacked_layers(
    create_layer_fn: Callable[[nnx.Rngs], nnx.Module],
    num_layers: int,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Create stacked decoder layers by creating one layer at a time and copying to pre-allocated arrays.

    This creates a single module object where all parameters have shape (num_layers, ...).
    This enables efficient scanning over layers without runtime stacking.

    Memory optimization: Instead of creating all layers then stacking (which requires 2x memory),
    we pre-allocate the stacked arrays and copy each layer's params directly, keeping only
    one layer in memory at a time.

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
    from functools import partial

    import jax.numpy as jnp
    import jax.random
    from jax.sharding import NamedSharding, PartitionSpec

    _log_mem("create_stacked_layers:start")

    # Split the RNG key to get unique keys for each layer
    base_key = rngs.params()
    layer_keys = jax.random.split(base_key, num_layers)

    # Get the current mesh for sharding
    mesh = jax.sharding.get_mesh()

    # Step 1: Create first layer to get structure and shapes
    first_layer = create_layer_fn(nnx.Rngs(layer_keys[0]))
    graphdef, first_state = nnx.split(first_layer)
    flat_first, treedef = jax.tree_util.tree_flatten(first_state)

    num_params = len(flat_first)
    logger.info(f"[MEM] Creating {num_layers} layers with {num_params} params each")
    _log_mem("create_stacked_layers:after_first_layer")

    # Step 2: Pre-allocate stacked arrays with proper sharding
    stacked_flat = []
    for arr in flat_first:
        # Determine sharding for stacked array
        original_sharding = arr.sharding
        if hasattr(original_sharding, "spec"):
            original_spec = original_sharding.spec
            new_spec = PartitionSpec(None, *original_spec)
            new_sharding = NamedSharding(mesh, new_spec)
        else:
            new_sharding = None

        # Pre-allocate with zeros
        stacked_shape = (num_layers,) + arr.shape
        if new_sharding is not None:
            stacked = jax.device_put(jnp.zeros(stacked_shape, dtype=arr.dtype), new_sharding)
        else:
            stacked = jnp.zeros(stacked_shape, dtype=arr.dtype)
        stacked_flat.append(stacked)

    _log_mem("create_stacked_layers:after_preallocate")

    # Step 3: Copy first layer's params to slice 0
    @jax.jit
    def copy_to_slice(stacked, arr, idx):
        return jax.lax.dynamic_update_slice(stacked, arr[None], (idx,) + (0,) * arr.ndim)

    for param_idx in range(num_params):
        stacked_flat[param_idx] = copy_to_slice(stacked_flat[param_idx], flat_first[param_idx], 0)

    # Free first layer
    del first_layer, first_state, flat_first
    _log_mem("create_stacked_layers:after_layer_0")

    # Step 4: Create remaining layers one at a time, copy params, then free
    for layer_idx in range(1, num_layers):
        layer = create_layer_fn(nnx.Rngs(layer_keys[layer_idx]))
        _, state = nnx.split(layer)
        flat_state, _ = jax.tree_util.tree_flatten(state)

        # Copy each param to the appropriate slice
        for param_idx in range(num_params):
            stacked_flat[param_idx] = copy_to_slice(
                stacked_flat[param_idx], flat_state[param_idx], layer_idx
            )

        # Free this layer immediately
        del layer, state, flat_state

        if layer_idx == num_layers - 1 or (layer_idx + 1) % 6 == 0:
            _log_mem(f"create_stacked_layers:after_layer_{layer_idx}")

    _log_mem("create_stacked_layers:after_all_layers")

    # Step 5: Reconstruct the state tree with stacked arrays
    stacked_state = jax.tree_util.tree_unflatten(treedef, stacked_flat)
    del stacked_flat

    # Merge back into a module with stacked parameters
    result = nnx.merge(graphdef, stacked_state)
    _log_mem("create_stacked_layers:end")
    return result


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

    def body_fn(carry, layer_params):
        hs, cache_keys, cache_values, layer_idx = carry

        # Extract layer's cache slice if available
        if cache_keys is not None:
            layer_kv = (cache_keys[layer_idx], cache_values[layer_idx])
        else:
            layer_kv = None

        # Forward through layer
        layer = nnx.merge(layer_graphdef, layer_params)
        new_hs, (k, v) = layer(
            hs,
            attention_mask=attention_mask,
            positions=positions,
            adapter_indices=adapter_indices,
            kv_cache=layer_kv,
        )

        hs_output = new_hs if output_hidden_states else None

        # Update cache in carry if present (decode), otherwise accumulate outputs (prefill)
        if cache_keys is not None:
            cache_keys = cache_keys.at[layer_idx].set(k)
            cache_values = cache_values.at[layer_idx].set(v)
            k = v = None  # Don't accumulate in output - cache is in carry
        elif is_training:
            k = v = None

        return (new_hs, cache_keys, cache_values, layer_idx + 1), (hs_output, k, v)

    if gradient_checkpointing:
        body_fn = jax.checkpoint(body_fn)

    cache_keys = kv_cache.keys if kv_cache else None
    cache_values = kv_cache.values if kv_cache else None
    init_carry = (hidden_states, cache_keys, cache_values, 0)

    (final_hs, final_keys, final_values, _), (all_hs, all_keys, all_values) = jax.lax.scan(
        body_fn, init_carry, layer_state
    )

    if is_decode:
        new_kv_cache = KVCache(
            keys=final_keys,
            values=final_values,
            cache_position=kv_cache.cache_position + positions.shape[1],
        )
    else:
        new_kv_cache = None if is_training else KVCache.from_layer_outputs(all_keys, all_values, attention_mask)

    all_hidden_states = [hidden_states] + list(all_hs[:-1]) if output_hidden_states else []
    return final_hs, all_hidden_states, new_kv_cache
