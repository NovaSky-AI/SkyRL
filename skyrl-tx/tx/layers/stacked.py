"""StackedDecoderLayers module for efficient transformer layer stacking."""

import functools
from typing import Callable

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from tx.utils.generator import KVCache


class ArrayRef(nnx.Variable):
    """A Variable providing a view into an indexed slice of a parent Variable."""

    def __init__(self, parent: nnx.Variable, idx: int):
        super().__init__(parent[idx])
        self.set_metadata("_parent", parent)
        self.set_metadata("_idx", idx)

    def __getitem__(self, key):
        parent, idx = self.get_metadata("_parent"), self.get_metadata("_idx")
        return parent[idx] if key is Ellipsis else parent[idx][key]

    def __setitem__(self, key, value):
        """Write through to parent when value is set via indexing."""
        parent, idx = self.get_metadata("_parent"), self.get_metadata("_idx")
        if key is Ellipsis:
            # param[...] = value -> update entire slice
            parent[...] = parent[...].at[idx].set(value)
        else:
            # param[key] = value -> update sub-slice
            parent[...] = parent[...].at[idx][key].set(value)
        # Also update our local value
        super().__setitem__(key, value)

    def set_raw_value(self, value, **kwargs):
        """Write through to parent when value is set."""
        parent, idx = self.get_metadata("_parent"), self.get_metadata("_idx")
        parent[...] = parent[...].at[idx].set(value)
        super().set_raw_value(value, **kwargs)

    @property
    def shape(self):
        return self.get_metadata("_parent")[self.get_metadata("_idx")].shape


class StackedDecoderLayers(nnx.Module):
    """Decoder layers with stacked weights for efficient scan-based forward pass.

    Parameters are stored in stacked format (num_layers, ...). The forward pass
    uses jax.lax.scan for all modes (training/prefill/decode) with KV cache as
    scan carry for efficient buffer donation.

    This class encapsulates both layer creation and forward pass logic.
    """

    def __init__(
        self,
        create_layer_fn: Callable[[nnx.Rngs], nnx.Module],
        num_layers: int,
        rngs: nnx.Rngs,
    ):
        """Create stacked decoder layers.

        This creates a single _stacked module where all parameters have shape (num_layers, ...).
        Layers are created individually and stacked to avoid nnx.vmap memory overhead.

        Args:
            create_layer_fn: Function that takes rngs and returns a single layer module.
            num_layers: Number of layers to create. Can be 0 for empty layer stack.
            rngs: Random number generators for initialization.
        """
        self.num_layers = num_layers

        # Handle empty layer case
        if num_layers == 0:
            self._stacked = None
            return

        layer_keys = jax.random.split(rngs.params(), num_layers)
        mesh = jax.sharding.get_mesh()

        # Create first layer to get structure and shapes
        first_layer = create_layer_fn(nnx.Rngs(layer_keys[0]))
        graphdef, first_state = nnx.split(first_layer)
        flat_first, treedef = jax.tree_util.tree_flatten(first_state)

        # Pre-allocate stacked arrays with correct sharding
        stacked_flat = []
        for arr in flat_first:
            stacked_shape = (num_layers,) + arr.shape
            original_sharding = arr.sharding
            if hasattr(original_sharding, "spec"):
                new_spec = PartitionSpec(None, *original_sharding.spec)
                stacked = jax.device_put(jnp.zeros(stacked_shape, arr.dtype), NamedSharding(mesh, new_spec))
            else:
                stacked = jnp.zeros(stacked_shape, arr.dtype)
            stacked_flat.append(stacked)

        # JIT with donate_argnums enables buffer reuse
        @functools.partial(jax.jit, donate_argnums=(0,))
        def copy_to_slice(stacked, arr, idx):
            return stacked.at[idx].set(arr)

        # Copy first layer's params to slot 0
        for i, arr in enumerate(flat_first):
            stacked_flat[i] = copy_to_slice(stacked_flat[i], flat_first[i], 0)

        # Create remaining layers one at a time and copy params
        for layer_idx in range(1, num_layers):
            layer = create_layer_fn(nnx.Rngs(layer_keys[layer_idx]))
            _, state = nnx.split(layer)
            flat, _ = jax.tree_util.tree_flatten(state)
            for i, arr in enumerate(flat):
                stacked_flat[i] = copy_to_slice(stacked_flat[i], flat[i], layer_idx)

        # Reconstruct and merge
        stacked_state = jax.tree_util.tree_unflatten(treedef, stacked_flat)
        self._stacked = nnx.merge(graphdef, stacked_state)

    def __len__(self) -> int:
        """Return the number of layers."""
        return self.num_layers

    def __getitem__(self, index: int) -> nnx.Module:
        """Get view into layer at index (stays synced with stacked state)."""
        if index < 0 or index >= self.num_layers:
            raise IndexError(f"Layer index {index} out of range [0, {self.num_layers})")
        graphdef, state = nnx.split(self._stacked)
        layer_state = jax.tree.map(
            lambda x: ArrayRef(x, index),
            state,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        )
        return nnx.merge(graphdef, layer_state)

    def __iter__(self):
        """Iterate over individual layers (for testing/weight loading)."""
        for i in range(self.num_layers):
            yield self[i]

    def __call__(
        self,
        hidden_states: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array,
        adapter_indices: jax.Array | None,
        kv_cache: KVCache | None,
        output_hidden_states: bool,
        gradient_checkpointing: bool,
        is_training: bool = False,
    ) -> tuple[jax.Array, list[jax.Array], KVCache | None]:
        """Forward pass through all layers using scan.

        Uses jax.lax.scan for all modes (training/prefill/decode). For decode mode,
        the KV cache is passed as scan carry for efficient buffer donation.

        Args:
            hidden_states: Input hidden states of shape (batch, seq, hidden).
            attention_mask: Attention mask of shape (batch, seq).
            positions: Position indices of shape (batch, seq).
            adapter_indices: Optional LoRA adapter indices of shape (batch,).
            kv_cache: Optional KV cache for decode mode (None for prefill).
            output_hidden_states: Whether to return intermediate hidden states.
            gradient_checkpointing: Whether to use gradient checkpointing.
            is_training: Whether in training mode. Skips KV cache to save memory.

        Returns:
            Tuple of (final_hidden_states, all_hidden_states, kv_cache).
            kv_cache is None when is_training=True.
        """
        # Handle empty layer case - pass through inputs unchanged
        if self.num_layers == 0:
            return hidden_states, [], kv_cache

        graphdef, state = nnx.split(self._stacked)
        is_decode = kv_cache is not None

        def body_fn(carry, layer_params):
            hs, cache_keys, cache_values, layer_idx = carry

            # Extract layer's cache slice if available
            if cache_keys is not None:
                layer_kv = (cache_keys[layer_idx], cache_values[layer_idx])
            else:
                layer_kv = None

            # Forward through layer
            layer = nnx.merge(graphdef, layer_params)
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
            body_fn, init_carry, state
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


def unstack_state(module: nnx.Module) -> nnx.GraphState:
    """Transform stacked layer state to unstacked ArrayRef views.

    Converts paths like `layers._stacked.xxx` to `layers.0.xxx`, `layers.1.xxx`, etc.
    Each entry is an ArrayRef that writes through to the original stacked variable.

    This is useful for checkpoint loading where weights are stored per-layer.


    For models with multiple StackedDecoderLayers (e.g., DeepSeek with dense + MoE),
    the model can provide get_stacked_layers_list() to specify ordering. Otherwise,
    falls back to simple per-stack numbering.

    Args:
        module: Module containing StackedDecoderLayers.

    Returns:
        GraphState with unstacked paths and ArrayRef views.
    """
    # Build mapping: StackedDecoderLayers object id → starting checkpoint index
    checkpoint_mapping = {}

    if hasattr(module, "model") and hasattr(module.model, "get_stacked_layers_list"):
        # Model provides explicit ordering - use sequential checkpoint indices
        counter = 0
        for stacked_layers in module.model.get_stacked_layers_list():
            checkpoint_mapping[id(stacked_layers)] = counter
            counter += stacked_layers.num_layers

    expanded = []
    for path, param in nnx.to_flat_state(nnx.state(module)):
        if "_stacked" not in path:
            expanded.append((path, param))
            continue

        stacked_idx = path.index("_stacked")

        # Find the StackedDecoderLayers object this parameter belongs to
        stacked_layers = module
        for key in path[:stacked_idx]:
            stacked_layers = getattr(stacked_layers, key)
        assert isinstance(stacked_layers, StackedDecoderLayers)

        if id(stacked_layers) in checkpoint_mapping:
            # Use checkpoint mapping - replace attribute name with "layers"
            start_idx = checkpoint_mapping[id(stacked_layers)]
            # Path: ("model", "dense_layers", "_stacked", ...) → ("model", "layers", "0", ...)
            base_path = path[:stacked_idx-1] + ("layers",)
            for layer_idx in range(stacked_layers.num_layers):
                checkpoint_idx = start_idx + layer_idx
                new_path = base_path + (str(checkpoint_idx),) + path[stacked_idx+1:]
                expanded.append((new_path, ArrayRef(param, layer_idx)))
        else:
            # Fallback: simple numbering within the same attribute
            for layer_idx in range(param[...].shape[0]):
                new_path = path[:stacked_idx] + (str(layer_idx),) + path[stacked_idx+1:]
                expanded.append((new_path, ArrayRef(param, layer_idx)))

    return nnx.from_flat_state(expanded)
