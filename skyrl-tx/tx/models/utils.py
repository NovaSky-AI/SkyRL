"""Utility functions for model forward passes."""

from flax import nnx
import jax
from jax import numpy as jnp

from tx.utils.generator import KVCache


def forward_layers_checkpointed(
    layers: nnx.List,
    hidden_states: jax.Array,
    *,
    attention_mask: jax.Array,
    positions: jax.Array,
    adapter_indices: jax.Array | None,
    output_hidden_states: bool,
) -> tuple[jax.Array, list[jax.Array]]:
    """Forward pass with gradient checkpointing using scan.

    Uses scan so XLA compiles ONE loop body and reuses buffers during
    backward recomputation. With a Python loop, XLA unrolls N separate
    checkpoint regions and can't optimize buffer reuse across them.

    Tradeoff: requires stacking all layer weights once per forward pass.
    This is acceptable because checkpointing already trades compute for memory.

    TODO(haochen): Load weights directly into stacked format to avoid 2x memory.
    Currently we have both self.layers (original) and stacked copy during forward.
    """
    num_layers = len(layers)
    if num_layers == 0:
        return hidden_states, []

    # Stack layer weights for dynamic indexing in scan
    layer_graphdef, _ = nnx.split(layers[0])
    stacked_weights = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *[nnx.state(layer) for layer in layers])

    def body_fn(hs, i):
        layer_weights = jax.tree.map(lambda x: x[i], stacked_weights)
        layer = nnx.merge(layer_graphdef, layer_weights)
        hs, _ = layer(
            hs, attention_mask=attention_mask, positions=positions, adapter_indices=adapter_indices, kv_cache=None
        )
        return hs, hs if output_hidden_states else None

    body_fn = jax.checkpoint(body_fn)
    final_hs, all_hs = jax.lax.scan(body_fn, hidden_states, jnp.arange(num_layers))

    if output_hidden_states:
        # all_hs is [num_layers, batch, seq, hidden]. Exclude last layer output since
        # it gets normed and appended in __call__ (matching non-checkpointed path).
        all_hidden_states = [hidden_states] + [all_hs[i] for i in range(num_layers - 1)]
    else:
        all_hidden_states = []

    return final_hs, all_hidden_states


def forward_layers(
    layers: nnx.List,
    hidden_states: jax.Array,
    *,
    attention_mask: jax.Array,
    positions: jax.Array,
    adapter_indices: jax.Array | None,
    kv_cache: KVCache | None,
    output_hidden_states: bool,
) -> tuple[jax.Array, list[jax.Array], list[jax.Array], list[jax.Array]]:
    """Standard forward pass through decoder layers.

    Used for inference (with KV cache) and training without checkpointing.

    Returns:
        hidden_states: Final hidden states after all layers
        all_hidden_states: List of hidden states from each layer (if output_hidden_states)
        updated_keys: List of updated key caches
        updated_values: List of updated value caches
    """
    all_hidden_states: list[jax.Array] = []
    updated_keys, updated_values = [], []

    for layer_idx, layer in enumerate(layers):
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        layer_kv = kv_cache and (kv_cache.keys[layer_idx], kv_cache.values[layer_idx], kv_cache.cache_position)
        hidden_states, (k, v) = layer(
            hidden_states,
            attention_mask=attention_mask,
            positions=positions,
            adapter_indices=adapter_indices,
            kv_cache=layer_kv,
        )
        updated_keys.append(k)
        updated_values.append(v)

    return hidden_states, all_hidden_states, updated_keys, updated_values
