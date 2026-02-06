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

        # Reconstruct state from stacked arrays
        stacked_state = jax.tree_util.tree_unflatten(treedef, stacked_flat)

        # Sync NNX sharding metadata with actual array sharding.
        # The arrays have correct stacked sharding from device_put, but NNX APIs
        # (nnx.get_partition_spec, nnx.Optimizer) read from 'sharding_names' metadata.
        def update_sharding_metadata(var):
            if isinstance(var, nnx.Variable) and hasattr(var.value, "sharding"):
                array_sharding = var.value.sharding
                if hasattr(array_sharding, "spec"):
                    var.set_metadata("sharding_names", tuple(array_sharding.spec))
            return var

        jax.tree.map(update_sharding_metadata, stacked_state, is_leaf=lambda x: isinstance(x, nnx.Variable))

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

    def unstack_paths(self, state: nnx.GraphState, base_path: tuple = ()) -> list[tuple[tuple, ArrayRef]]:
        """Transform _stacked paths to per-layer paths with ArrayRef.

        Args:
            state: GraphState containing this module's state.
            base_path: Path prefix to this module (e.g., ("model", "layers")).

        Returns:
            List of (path, ArrayRef) tuples for unstacked parameters.
        """
        result = []
        for path, param in nnx.to_flat_state(state):
            # Only process paths belonging to this module
            if not path[: len(base_path)] == base_path:
                continue
            # Only process _stacked paths
            if "_stacked" not in path[len(base_path) :]:
                continue

            # Find _stacked in the relative path
            rel_path = path[len(base_path) :]
            stacked_idx = rel_path.index("_stacked")

            # Create per-layer paths: base_path + (layer_idx,) + rest
            for layer_idx in range(self.num_layers):
                new_path = base_path + (str(layer_idx),) + rel_path[stacked_idx + 1 :]
                result.append((new_path, ArrayRef(param, layer_idx)))

        return result

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
        """Forward pass through all layers.

        Uses scan for prefill/training (efficient, no KV cache needed).
        Uses Python loop for decode (with list-format KV cache) to enable buffer donation.

        Args:
            hidden_states: Input hidden states of shape (batch, seq, hidden).
            attention_mask: Attention mask of shape (batch, seq).
            positions: Position indices of shape (batch, seq).
            adapter_indices: Optional LoRA adapter indices of shape (batch,).
            kv_cache: Optional KV cache for decode mode (None for prefill). Uses list format.
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

        if is_decode:
            # Decode mode: Use Python loop with list KV cache for buffer donation.
            # We avoid jax.lax.scan here because carrying a stacked KV cache through scan
            # and updating it with cache.at[layer_idx].set() causes XLA to copy the entire
            # cache array on each layer (16MB per layer). XLA can't prove the buffer can be
            # donated since it doesn't know the slices are non-overlapping. With a Python
            # loop and list format, each layer's KV array is independent and can be donated.
            flat_state, treedef = jax.tree_util.tree_flatten(state)
            all_hidden_states: list[jax.Array] = []
            updated_keys: list[jax.Array] = []
            updated_values: list[jax.Array] = []

            for layer_idx in range(self.num_layers):
                if output_hidden_states:
                    all_hidden_states.append(hidden_states)

                # Extract this layer's parameters
                layer_params_flat = [p[layer_idx] for p in flat_state]
                layer_params = jax.tree_util.tree_unflatten(treedef, layer_params_flat)
                layer = nnx.merge(graphdef, layer_params)

                # Get this layer's KV cache
                layer_kv = (kv_cache.keys[layer_idx], kv_cache.values[layer_idx])

                hidden_states, (k, v) = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    positions=positions,
                    adapter_indices=adapter_indices,
                    kv_cache=layer_kv,
                )
                updated_keys.append(k)
                updated_values.append(v)

            new_kv_cache = KVCache.update(kv_cache, updated_keys, updated_values, positions, attention_mask)
            return hidden_states, all_hidden_states, new_kv_cache

        # Prefill/training mode: use scan for efficiency
        def body_fn(carry, layer_params):
            hs = carry

            # Forward through layer (no KV cache input for prefill)
            layer = nnx.merge(graphdef, layer_params)
            new_hs, (k, v) = layer(
                hs,
                attention_mask=attention_mask,
                positions=positions,
                adapter_indices=adapter_indices,
                kv_cache=None,
            )

            hs_output = new_hs if output_hidden_states else None

            # Skip KV accumulation in training mode to save memory
            if is_training:
                k = v = None

            return new_hs, (hs_output, k, v)

        if gradient_checkpointing:
            body_fn = jax.checkpoint(body_fn)

        final_hs, (all_hs, all_keys, all_values) = jax.lax.scan(body_fn, hidden_states, state)

        if is_training:
            new_kv_cache = None
        else:
            # Convert stacked scan outputs to list format
            keys_list = [all_keys[i] for i in range(self.num_layers)]
            values_list = [all_values[i] for i in range(self.num_layers)]
            new_kv_cache = KVCache.update(None, keys_list, values_list, positions, attention_mask)

        all_hidden_states = [hidden_states] + list(all_hs[:-1]) if output_hidden_states else []
        return final_hs, all_hidden_states, new_kv_cache


class MultiStackedDecoderLayers(nnx.Module):
    """Multiple StackedDecoderLayers groups with unified interface.

    This allows models like DeepSeek to have different layer types (dense/MoE)
    while presenting a unified interface for forward passes and checkpointing.
    """

    def __init__(self, *layer_groups: StackedDecoderLayers):
        """Create multi-stacked decoder layers.

        Args:
            *layer_groups: One or more StackedDecoderLayers to combine.
        """
        self.layer_groups = nnx.List(layer_groups)
        self.num_layers = sum(group.num_layers for group in self.layer_groups)

    def __len__(self) -> int:
        """Return the total number of layers across all groups."""
        return self.num_layers

    def __getitem__(self, index: int) -> nnx.Module:
        """Get view into layer at global index (across all groups)."""
        if index < 0 or index >= self.num_layers:
            raise IndexError(f"Layer index {index} out of range [0, {self.num_layers})")

        # Find which group contains this index
        offset = 0
        for group in self.layer_groups:
            if index < offset + group.num_layers:
                return group[index - offset]
            offset += group.num_layers

        raise IndexError(f"Layer index {index} not found")

    def __iter__(self):
        """Iterate over all layers across all groups."""
        for group in self.layer_groups:
            yield from group

    def get_stacked_layers_list(self) -> list[StackedDecoderLayers]:
        """Return list of StackedDecoderLayers for checkpoint loading."""
        return list(self.layer_groups)

    def unstack_paths(self, state: nnx.GraphState, base_path: tuple = ()) -> list[tuple[tuple, ArrayRef]]:
        """Transform _stacked paths from all groups to unified per-layer paths.

        Args:
            state: GraphState containing this module's state.
            base_path: Path prefix to this module (e.g., ("model", "layers")).

        Returns:
            List of (path, ArrayRef) tuples for unstacked parameters.
        """
        result = []
        checkpoint_idx = 0

        for i, group in enumerate(self.layer_groups):
            # Path to this group: base_path + ("layer_groups", i)
            group_path = base_path + ("layer_groups", i)

            # Get unstacked paths from the group
            for path, array_ref in group.unstack_paths(state, group_path):
                # Extract layer index from path: group_path + (layer_idx,) + rest
                layer_idx = int(path[len(group_path)])
                # New path: base_path + (checkpoint_idx + layer_idx,) + rest
                new_path = base_path + (str(checkpoint_idx + layer_idx),) + path[len(group_path) + 1 :]
                result.append((new_path, array_ref))

            checkpoint_idx += group.num_layers

        return result

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
        """Forward pass through all layer groups.

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
        """
        all_hidden_states: list[jax.Array] = []

        # Split KV cache for each group
        if kv_cache is not None:
            split_points = []
            cumsum = 0
            for group in self.layer_groups[:-1]:
                cumsum += group.num_layers
                split_points.append(cumsum)
            kv_caches = kv_cache.split(*split_points)
        else:
            kv_caches = (None,) * len(self.layer_groups)

        # Forward through each group
        kv_results = []
        for group, group_kv_cache in zip(self.layer_groups, kv_caches):
            hidden_states, layer_hidden_states, layer_kv_cache = group(
                hidden_states,
                attention_mask=attention_mask,
                positions=positions,
                adapter_indices=adapter_indices,
                kv_cache=group_kv_cache,
                output_hidden_states=output_hidden_states,
                gradient_checkpointing=gradient_checkpointing,
                is_training=is_training,
            )
            all_hidden_states.extend(layer_hidden_states)
            kv_results.append(layer_kv_cache)

        # Concatenate KV caches
        new_kv_cache = KVCache.concatenate(*kv_results) if kv_results else None

        return hidden_states, all_hidden_states, new_kv_cache


def unstack_state(module: nnx.Module) -> nnx.GraphState:
    """Transform stacked layer state to unstacked ArrayRef views.

    Converts paths like `layers._stacked.xxx` to `layers.0.xxx`, `layers.1.xxx`, etc.
    Each entry is an ArrayRef that writes through to the original stacked variable.

    This is useful for checkpoint loading where weights are stored per-layer.

    Args:
        module: Module containing StackedDecoderLayers.

    Returns:
        GraphState with unstacked paths and ArrayRef views.
    """
    state = nnx.state(module)
    expanded = []

    # Delegate to layers if they support unstacking
    if hasattr(module, "model") and hasattr(module.model, "layers"):
        layers = module.model.layers
        if isinstance(layers, (StackedDecoderLayers, MultiStackedDecoderLayers)):
            expanded.extend(layers.unstack_paths(state, base_path=("model", "layers")))

    # Keep all non-stacked paths as-is
    for path, param in nnx.to_flat_state(state):
        if "_stacked" not in path:
            expanded.append((path, param))

    return nnx.from_flat_state(expanded)
