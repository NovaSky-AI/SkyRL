from flax import nnx
import jax
from jax import numpy as jnp

from tx.utils.models import filter_lora
from tx.layers.util import Param, prepare_routing, ragged_dot
from tx.models.types import ModelForCausalLM
from tx.tinker.types import LoraConfig


class LoRAMixin:
    """A mixin for flax NNX modules to add multi-adapter LoRA support.
    This mixin adds LoRA parameters (lora_A, lora_B) and methods to apply
    the low-rank adaptation to a base module's output. It is designed to
    be used with layers like nnx.Linear.
    """

    lora_scaling: nnx.Variable | None
    lora_ranks: nnx.Variable | None
    lora_A: nnx.Param | None
    lora_B: nnx.Param | None

    def init_lora(
        self,
        *,
        max_lora_adapters: int,
        max_lora_rank: int,
        shape_A: tuple[int, ...],
        shape_B: tuple[int, ...],
        sharding_A: tuple[str | None, ...],
        sharding_B: tuple[str | None, ...],
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize LoRA parameter tensors."""
        self.max_lora_adapters = max_lora_adapters
        self.max_lora_rank = max_lora_rank

        if max_lora_adapters == 0:
            self.lora_scaling = None
            self.lora_ranks = None
            self.lora_A = None
            self.lora_B = None
        else:
            self.lora_scaling = nnx.Variable(jnp.full((max_lora_adapters,), 1.0, dtype=dtype))
            self.lora_ranks = nnx.Variable(jnp.full((max_lora_adapters,), max_lora_rank, dtype=jnp.int32))
            # lora_A and lora_B are initialized to zeros here. The actual weight
            # initialization (he_uniform for lora_A, zeros for lora_B) happens in
            # init_lora_adapter(), which must be called before training.
            self.lora_A = Param(
                *shape_A,
                dtype=dtype,
                kernel_init=nnx.with_partitioning(nnx.initializers.zeros_init(), sharding_A),
                rngs=rngs,
            )
            self.lora_B = Param(
                *shape_B,
                dtype=dtype,
                kernel_init=nnx.with_partitioning(nnx.initializers.zeros_init(), sharding_B),
                rngs=rngs,
            )

    def _apply_lora_weight(
        self,
        lora_weight: jax.Array,
        x_sorted: jax.Array,
        adapter_indices_sorted: jax.Array,
        group_sizes: jax.Array,
    ) -> jax.Array:
        """Apply a LoRA weight matrix to input. Default is linear case: x @ weight.

        Subclasses (e.g., LoRAEmbed) override this for different computation patterns.
        """
        assert lora_weight.ndim == 3
        assert x_sorted.ndim == 2  # (tokens, in_features)
        assert x_sorted.shape[1] == lora_weight.shape[1]
        return jax.lax.ragged_dot(x_sorted, lora_weight, group_sizes)

    def apply_lora(
        self,
        x: jax.Array,
        base_output: jax.Array,
        adapter_indices: jax.Array | None,
    ) -> jax.Array:
        if self.max_lora_adapters == 0 or adapter_indices is None:
            return base_output

        if self.lora_A is None or self.lora_B is None or self.lora_scaling is None:
            raise RuntimeError("LoRA parameters are not initialized. `init_lora` must be called.")

        (batch_size, seq_len, *dims) = x.shape
        assert adapter_indices.shape[0] == batch_size

        x_flat = x.reshape(-1, *dims)
        adapter_indices_expanded = jnp.repeat(adapter_indices, seq_len)

        # Sort tokens to prepare for ragged_dot
        x_sorted, group_sizes, unsort_indices, adapter_indices_sorted = prepare_routing(
            x_flat, adapter_indices_expanded, self.max_lora_adapters, adapter_indices=adapter_indices_expanded
        )

        # Apply LoRA: x @ A @ B (or A[x] @ B for embeddings)
        intermediate = self._apply_lora_weight(self.lora_A[...], x_sorted, adapter_indices_sorted, group_sizes)
        lora_output_sorted = jax.lax.ragged_dot(intermediate, self.lora_B[...], group_sizes)

        # Unsort, reshape, scale
        lora_output = lora_output_sorted[unsort_indices].reshape(batch_size, seq_len, -1)
        lora_output = lora_output * self.lora_scaling[...][adapter_indices, None, None]
        return base_output + lora_output.reshape(base_output.shape)


class LoRAEmbed(LoRAMixin, nnx.Embed):
    """An nnx.Embed layer with multi-adapter LoRA support"""

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        *,
        sharding: tuple[str | None, ...],
        max_lora_adapters: int = 0,
        max_lora_rank: int = 8,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype | None = None,
        embedding_init: nnx.Initializer,
        rngs: nnx.Rngs,
    ) -> None:
        param_dtype = param_dtype or dtype

        super().__init__(
            num_embeddings=num_embeddings,
            features=features,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=nnx.with_partitioning(embedding_init, sharding),
            rngs=rngs,
        )

        self.init_lora(
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            shape_A=(max_lora_adapters, num_embeddings, max_lora_rank),
            shape_B=(max_lora_adapters, max_lora_rank, features),
            sharding_A=(None, sharding[0], None),
            sharding_B=(None, None, sharding[1]),
            dtype=param_dtype,
            rngs=rngs,
        )

    def _apply_lora_weight(
        self,
        lora_weight: jax.Array,
        x_sorted: jax.Array,
        adapter_indices_sorted: jax.Array,
        group_sizes: jax.Array,
    ) -> jax.Array:
        """For embeddings, lookup in weight instead of matmul: weight[adapter, token_id, :]."""
        assert lora_weight.ndim == 3
        assert x_sorted.ndim == 1  # (tokens,) integer indices
        return lora_weight[adapter_indices_sorted, x_sorted, :]

    def __call__(self, x: jax.Array, adapter_indices: jax.Array | None = None) -> jax.Array:
        base_out = super().__call__(x)
        return self.apply_lora(x, base_out, adapter_indices)

    @property
    def T(self):
        """Return a callable that projects hidden states back to vocabulary space."""
        # TODO: Apply lora adapters here as well
        return lambda hidden_states, adapter_indices=None: hidden_states @ self.embedding[...].T


class LoRALinear(LoRAMixin, nnx.Linear):
    """An nnx.Linear layer with multi-adapter LoRA support."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        sharding: tuple[str | None, ...],
        max_lora_adapters: int = 0,
        max_lora_rank: int = 8,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype | None = None,
        use_bias: bool,
        kernel_init: nnx.Initializer,
        bias_init: nnx.Initializer | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        param_dtype = param_dtype or dtype
        if bias_init is None:
            bias_init = nnx.initializers.zeros_init()

        super().__init__(
            in_features,
            out_features,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(kernel_init, sharding),
            bias_init=nnx.with_partitioning(bias_init, (sharding[-1],)),
            rngs=rngs,
        )

        self.init_lora(
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            shape_A=(max_lora_adapters, in_features, max_lora_rank),
            shape_B=(max_lora_adapters, max_lora_rank, out_features),
            sharding_A=(None, sharding[0], None),
            sharding_B=(None, None, sharding[1]),
            dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, adapter_indices: jax.Array | None = None) -> jax.Array:
        base_out = super().__call__(x)
        return self.apply_lora(x, base_out, adapter_indices)


class LoRAExpert(LoRAMixin, nnx.Module):
    """Expert layer with multi-adapter LoRA support."""

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        *,
        sharding: tuple[str | None, ...],
        max_lora_adapters: int = 0,
        max_lora_rank: int = 8,
        dtype: jnp.dtype = jnp.float32,
        kernel_init: nnx.Initializer,
        rngs: nnx.Rngs,
    ) -> None:
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Param(
            num_experts,
            in_features,
            out_features,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(kernel_init, sharding),
            rngs=rngs,
        )

        self.init_lora(
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            shape_A=(max_lora_adapters, num_experts, in_features, max_lora_rank),
            shape_B=(max_lora_adapters, num_experts, max_lora_rank, out_features),
            sharding_A=(None, sharding[0], sharding[1], None),
            sharding_B=(None, sharding[0], None, sharding[2]),
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        group_sizes: jax.Array,
        adapter_indices_sorted: jax.Array | None = None,
        *,
        group_offset: jax.Array | None = None,
    ) -> jax.Array:
        base_out = ragged_dot(x, self.weight[...], group_sizes, group_offset=group_offset)

        if self.max_lora_adapters == 0 or adapter_indices_sorted is None:
            return base_out

        if self.lora_A is None or self.lora_B is None or self.lora_scaling is None:
            raise RuntimeError("LoRA parameters are not initialized. `init_lora` must be called.")

        # Reconstruct expert indices from group_sizes (global indices)
        expert_indices = jnp.repeat(jnp.arange(self.num_experts), group_sizes, total_repeat_length=x.shape[0])

        # Expert-first flattening so local expert groups are contiguous
        flattened_indices = expert_indices * self.max_lora_adapters + adapter_indices_sorted
        num_flattened_groups = self.num_experts * self.max_lora_adapters
        num_local_experts = self.lora_A[...].shape[1]

        # Reshape LoRA weights in expert-first order
        lora_A = (
            self.lora_A[...]
            .transpose((1, 0, 2, 3))
            .reshape(self.max_lora_adapters * num_local_experts, self.in_features, self.max_lora_rank)
        )
        lora_B = (
            self.lora_B[...]
            .transpose((1, 0, 2, 3))
            .reshape(self.max_lora_adapters * num_local_experts, self.max_lora_rank, self.out_features)
        )

        # Sort tokens by combined index
        x_sorted, combined_group_sizes, unsort_indices, _ = prepare_routing(x, flattened_indices, num_flattened_groups)

        # Compute group_offset for LoRA (scaled by max_lora_adapters)
        lora_group_offset = group_offset * self.max_lora_adapters if group_offset is not None else None

        # Apply LoRA using ragged_dot: x @ A @ B
        intermediate = ragged_dot(x_sorted, lora_A, combined_group_sizes, group_offset=lora_group_offset)
        lora_output_sorted = ragged_dot(intermediate, lora_B, combined_group_sizes, group_offset=lora_group_offset)

        # Unsort and apply scaling
        lora_output = lora_output_sorted[unsort_indices]
        lora_output = lora_output * self.lora_scaling[...][adapter_indices_sorted, None]

        return base_out + lora_output


def init_lora_adapter(model: ModelForCausalLM, adapter_index: int, lora_config: LoraConfig):
    """Initialize a LoRA adapter for training.

    Initializes the adapter: lora_A with he_uniform, lora_B with zeros,
    and sets the appropriate rank and scaling. This must be called BEFORE
    training begins on an adapter slot.

    Args:
        model: The model containing LoRA layers
        adapter_index: Index of the adapter to initialize
        lora_config: LoraConfig object containing rank, alpha, seed, and training flags
    """
    if lora_config.train_unembed and getattr(model.config, "tie_word_embeddings", False):
        raise ValueError(
            "train_unembed=True is incompatible with tie_word_embeddings=True. "
            "Tied embeddings use embed_tokens.T which does not support LoRA."
        )
    rngs = nnx.Rngs(lora_config.seed)
    state = nnx.state(model)

    def init_adapter(path, value):
        effective_rank = lora_config.rank
        normalized_path = tuple(p.key if hasattr(p, "key") else p.name for p in path)

        # Apply rank normalization for MoE expert layers
        # Following Thinking Machines' approach: divide rank by num_experts
        # to keep total LoRA parameters similar to non-MoE models
        if "experts" in normalized_path:
            effective_rank = max(1, lora_config.rank // model.config.get_num_experts())

        if not filter_lora(lora_config, normalized_path):
            effective_rank = 0

        key_name = path[-2].key
        if key_name == "lora_ranks":
            return value.at[adapter_index].set(effective_rank)
        if key_name == "lora_scaling":
            # Set scaling to 0.0 if rank is 0
            return value.at[adapter_index].set(lora_config.alpha / effective_rank if effective_rank > 0 else 0.0)
        if key_name == "lora_A":
            # Reinitialize with he_uniform, then zero columns beyond rank
            shape = value[adapter_index].shape
            new_A = nnx.initializers.he_uniform()(rngs.params(), shape, value.dtype)
            new_A = new_A.at[..., effective_rank:].set(0.0)
            return value.at[adapter_index].set(new_A)
        if key_name == "lora_B":
            # Explicitly zero lora_B
            return value.at[adapter_index].set(0.0)
        return value

    updated_state = jax.tree.map_with_path(init_adapter, state)
    nnx.update(model, updated_state)


def clear_lora_adapter(model: ModelForCausalLM, adapter_index: int):
    """Clear/reset a LoRA adapter, freeing it for reuse.

    Sets rank=0, scaling=0, and zeros out lora_A and lora_B for the adapter.
    Before reusing this adapter for training again, `init_lora_adapter` must be called.
    """
    state = nnx.state(model)

    def clear_adapter(path, value):
        key = path[-2].key
        if key == "lora_ranks":
            return value.at[adapter_index].set(0)
        if key in ("lora_scaling", "lora_A", "lora_B"):
            return value.at[adapter_index].set(0.0)
        return value

    updated_state = jax.tree.map_with_path(clear_adapter, state)
    nnx.update(model, updated_state)
