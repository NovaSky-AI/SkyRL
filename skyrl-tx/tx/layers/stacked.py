"""StackedDecoderLayers module for efficient transformer layer stacking."""

from typing import Callable

from flax import nnx
import jax
import jax.numpy as jnp

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

    def set_raw_value(self, value, **kwargs):
        """Write through to parent when value is set."""
        parent, idx = self.get_metadata("_parent"), self.get_metadata("_idx")
        parent[...] = parent[...].at[idx].set(value)
        super().set_raw_value(value, **kwargs)

    @property
    def shape(self):
        return self.get_metadata("_parent")[self.get_metadata("_idx")].shape


class StackedDecoderLayers(nnx.Module):
    """Decoder layers with stacked weights created via nnx.vmap.

    Parameters are stored in stacked format (num_layers, ...). The forward pass
    uses jax.lax.scan for training/prefill and Python loops for decode.
    """

    def __init__(
        self,
        create_layer_fn: Callable[[nnx.Rngs], nnx.Module],
        num_layers: int,
        rngs: nnx.Rngs,
    ):
        self.num_layers = num_layers

        @nnx.split_rngs(splits=num_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0, transform_metadata={nnx.PARTITION_NAME: None})
        def vmapped_create(rngs: nnx.Rngs) -> nnx.Module:
            return create_layer_fn(rngs)

        self._stacked = vmapped_create(rngs)

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
        """Forward pass through all layers.

        Uses scan for training/prefill, Python loop for decode.

        Returns:
            (final_hidden_states, all_hidden_states, kv_cache)
        """
        graphdef, state = nnx.split(self._stacked)

        # Decode mode: use Python loop
        if kv_cache is not None:
            all_hidden_states = []
            new_keys, new_values = [], []

            for i in range(self.num_layers):
                if output_hidden_states:
                    all_hidden_states.append(hidden_states)

                layer = nnx.merge(graphdef, jax.tree.map(lambda x, i=i: x[i], state))
                hidden_states, (k, v) = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    positions=positions,
                    adapter_indices=adapter_indices,
                    kv_cache=(kv_cache.keys[i], kv_cache.values[i]),
                )
                new_keys.append(k)
                new_values.append(v)

            return hidden_states, all_hidden_states, KVCache(
                keys=new_keys,
                values=new_values,
                cache_position=kv_cache.cache_position + positions.shape[1],
            )

        # Training/prefill mode: use scan
        def body_fn(hs, layer_params):
            layer = nnx.merge(graphdef, layer_params)
            new_hs, (k, v) = layer(
                hs,
                attention_mask=attention_mask,
                positions=positions,
                adapter_indices=adapter_indices,
                kv_cache=None,
            )
            if is_training:
                k = v = None
            return new_hs, (new_hs if output_hidden_states else None, k, v)

        if gradient_checkpointing:
            body_fn = jax.checkpoint(body_fn)

        final_hs, (all_hs, all_keys, all_values) = jax.lax.scan(body_fn, hidden_states, state)

        all_hidden_states = [hidden_states] + list(all_hs[:-1]) if output_hidden_states else []

        if is_training:
            return final_hs, all_hidden_states, None

        return final_hs, all_hidden_states, KVCache(
            keys=[all_keys[i] for i in range(self.num_layers)],
            values=[all_values[i] for i in range(self.num_layers)],
            cache_position=attention_mask.sum(axis=1).astype(jnp.int32),
        )


def unstack_state(module: nnx.Module) -> nnx.GraphState:
    """Transform stacked layer state to unstacked ArrayRef views.

    Converts paths like `layers._stacked.xxx` to `layers.0.xxx`, `layers.1.xxx`, etc.
    Each entry is an ArrayRef that writes through to the original stacked variable.
    """
    expanded = []
    for path, var in nnx.to_flat_state(nnx.state(module)):
        if "_stacked" not in path:
            expanded.append((path, var))
            continue

        idx = path.index("_stacked")
        for i in range(var[...].shape[0]):
            new_path = path[:idx] + (str(i),) + path[idx + 1 :]
            expanded.append((new_path, ArrayRef(var, i)))

    return nnx.from_flat_state(expanded)
