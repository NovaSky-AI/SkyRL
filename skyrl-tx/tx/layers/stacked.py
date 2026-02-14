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
        return parent[idx][key]

    def __setitem__(self, key, value):
        """Write through to parent when value is set via indexing.

        Only supports Ellipsis key (param[...] = value) because JAX's .at[idx]
        returns _IndexUpdateRef which doesn't support further subscripting.
        """
        if key is not Ellipsis:
            raise NotImplementedError("ArrayRef only supports `ref[...] = value`")
        self.set_raw_value(value)

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
    uses jax.lax.scan for training/prefill, and uses a Python loop for decode
    to update per-layer KV cache entries without stacked-cache copy overhead.

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
        flat_first, state_treedef = jax.tree_util.tree_flatten(first_state)

        # Build a treedef with stacked partition metadata so tree_unflatten
        # reconstructs Variables with the correct leading-layer sharding axis.
        stacked_first_state = nnx.spmd.add_axis(first_state, 0, {nnx.PARTITION_NAME: None})
        _, stacked_treedef = jax.tree_util.tree_flatten(stacked_first_state)

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

        # Create layers one at a time and copy params into stacked slots
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                flat = flat_first
            else:
                layer = create_layer_fn(nnx.Rngs(layer_keys[layer_idx]))
                _, state = nnx.split(layer)
                flat, current_treedef = jax.tree_util.tree_flatten(state)
                assert current_treedef == state_treedef, "Layer state structure mismatch while stacking decoder layers."
            for i, arr in enumerate(flat):
                stacked_flat[i] = copy_to_slice(stacked_flat[i], arr, layer_idx)

        # Reconstruct state from stacked arrays
        stacked_state = jax.tree_util.tree_unflatten(stacked_treedef, stacked_flat)

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
        prefix_len = len(base_path)
        for path, param in nnx.to_flat_state(state):
            # Only process paths belonging to this module
            if path[:prefix_len] != base_path:
                continue

            rel_path = path[prefix_len:]
            # Only process _stacked paths
            if "_stacked" not in rel_path:
                continue

            suffix = rel_path[rel_path.index("_stacked") + 1 :]
            for layer_idx in range(self.num_layers):
                new_path = base_path + (str(layer_idx),) + suffix
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
    """Multiple StackedDecoderLayers groups with a unified forward/unstack interface."""

    def __init__(self, *layer_groups: StackedDecoderLayers):
        self.layer_groups = nnx.List([group for group in layer_groups if group.num_layers > 0])
        self.num_layers = sum(group.num_layers for group in self.layer_groups)
        assert self.num_layers > 0, "MultiStackedDecoderLayers requires at least one non-empty layer group."

    def __len__(self) -> int:
        return self.num_layers

    def __getitem__(self, index: int) -> nnx.Module:
        if index < 0 or index >= self.num_layers:
            raise IndexError(f"Layer index {index} out of range [0, {self.num_layers})")

        offset = 0
        for group in self.layer_groups:
            if index < offset + group.num_layers:
                return group[index - offset]
            offset += group.num_layers
        raise IndexError(f"Layer index {index} out of range [0, {self.num_layers})")

    def __iter__(self):
        for group in self.layer_groups:
            yield from group

    def unstack_paths(self, state: nnx.GraphState, base_path: tuple = ()) -> list[tuple[tuple, ArrayRef]]:
        result = []
        checkpoint_idx = 0

        for i, group in enumerate(self.layer_groups):
            group_path = base_path + ("layer_groups", i)
            for path, array_ref in group.unstack_paths(state, group_path):
                layer_idx = int(path[len(group_path)])
                new_path = base_path + (str(checkpoint_idx + layer_idx),) + path[len(group_path) + 1 :]
                result.append((new_path, array_ref))
            checkpoint_idx += group.num_layers

        return result

    @staticmethod
    def _split_kv_cache(kv_cache: KVCache, split_points: list[int]) -> tuple[KVCache, ...]:
        boundaries = [0, *split_points, len(kv_cache.keys)]
        caches: list[KVCache] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            caches.append(
                KVCache(
                    keys=kv_cache.keys[start:end],
                    values=kv_cache.values[start:end],
                    cache_position=kv_cache.cache_position,
                )
            )
        return tuple(caches)

    @staticmethod
    def _concat_kv_caches(caches: list[KVCache]) -> KVCache:
        assert caches, "Expected at least one KV cache."
        keys: list[jax.Array] = []
        values: list[jax.Array] = []
        for cache in caches:
            keys.extend(cache.keys)
            values.extend(cache.values)
        return KVCache(keys=keys, values=values, cache_position=caches[-1].cache_position)

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
        all_hidden_states: list[jax.Array] = []

        if kv_cache is not None:
            split_points = []
            cumsum = 0
            for group in self.layer_groups[:-1]:
                cumsum += group.num_layers
                split_points.append(cumsum)
            kv_caches = self._split_kv_cache(kv_cache, split_points)
        else:
            kv_caches = (None,) * len(self.layer_groups)

        kv_results: list[KVCache] = []
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
            if not is_training:
                assert layer_kv_cache is not None, "Expected KV cache in non-training mode."
                kv_results.append(layer_kv_cache)

        return hidden_states, all_hidden_states, None if is_training else self._concat_kv_caches(kv_results)


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
