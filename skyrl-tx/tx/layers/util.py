from flax import nnx
import jax
from jax import lax
from jax import numpy as jnp
from jax.sharding import get_abstract_mesh, PartitionSpec


def ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    precision=None,
    preferred_element_type=None,
    group_offset: jax.Array | None = None,
) -> jax.Array:
    """Ragged dot product with group_offset support.

    When group_offset is specified, rhs contains groups [offset, offset + g_local).
    Tokens outside this range are routed to boundary groups and masked to zero.
    """
    if group_offset is None:
        return lax.ragged_dot(
            lhs,
            rhs,
            group_sizes,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )

    assert group_offset.shape == (1,), "group_offset must have shape (1,)"
    offset = group_offset[0]
    m = lhs.shape[0]
    g_local = rhs.shape[0]

    assert g_local > 0, "rhs must have at least one group"

    # Compute token boundaries for local groups
    cumsum = jnp.cumulative_sum(group_sizes, include_initial=True)
    shard_start = cumsum[offset]
    shard_end = cumsum[offset + g_local]

    # Valid mask for tokens in local groups
    token_idx = jnp.arange(m)
    valid_mask = (token_idx >= shard_start) & (token_idx < shard_end)

    # Adjust group sizes: absorb extra tokens at boundaries
    local_group_sizes = lax.dynamic_slice_in_dim(group_sizes, offset, g_local, axis=0)
    adjusted_group_sizes = local_group_sizes.at[0].add(shard_start).at[-1].add(m - shard_end)

    # Call ragged_dot - extra tokens use boundary groups but get masked out
    result = lax.ragged_dot(
        lhs,
        rhs,
        adjusted_group_sizes,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

    return jnp.where(valid_mask[:, None], result, 0)


def Param(*shape: int, dtype: jnp.dtype, kernel_init: nnx.Initializer, rngs: nnx.Rngs):
    return nnx.Param(kernel_init(rngs.param(), shape, dtype))


def prepare_routing(
    tokens: jax.Array, indices: jax.Array, num_groups: int, adapter_indices: jax.Array | None = None
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """Prepare inputs for ragged_dot operations by sorting tokens by group.

    Args:
        tokens: Array of shape (num_tokens, ...) to be sorted by group
        indices: Array of shape (num_tokens,) indicating group assignment for each token
        num_groups: Total number of groups
        adapter_indices: Optional array of shape (num_tokens,) to be sorted together with tokens

    Returns:
        sorted_tokens: Tokens sorted by group index
        group_sizes: Number of tokens in each group
        unsort_indices: Indices to restore original order after ragged operations
    """
    sort_indices = jnp.argsort(indices)
    sorted_tokens = tokens[sort_indices]
    sorted_adapter_indices = None if adapter_indices is None else adapter_indices[sort_indices]
    group_sizes = jnp.bincount(indices, length=num_groups)
    unsort_indices = jnp.argsort(sort_indices)
    return sorted_tokens, group_sizes, unsort_indices, sorted_adapter_indices


def shard_experts(
    hidden_states: jax.Array,
    selected_experts: jax.Array,
    routing_weights: jax.Array,
    experts: nnx.Module,
    forward,
    adapter_indices: jax.Array | None = None,
) -> jax.Array:
    """Dispatch tokens to experts and combine outputs using expert parallelism.

    Args:
        hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
        selected_experts: Expert indices of shape (batch, seq_len, num_experts_per_tok)
        routing_weights: Routing weights of shape (batch, seq_len, num_experts_per_tok)
        experts: Module with config (num_experts, num_experts_per_tok, hidden_size)
        forward: Callable (experts, hidden_states, group_sizes, adapter_indices, group_offset) -> output
        adapter_indices: Optional adapter indices for LoRA
    """
    num_experts = experts.config.num_experts
    num_experts_per_tok = experts.config.num_experts_per_tok
    hidden_size = experts.config.hidden_size

    mesh = get_abstract_mesh()
    ep_size = mesh.shape.get("ep", 1) if mesh is not None else 1

    assert num_experts % ep_size == 0
    experts_per_rank = num_experts // ep_size

    # Split module into graphdef and state
    graphdef, state = nnx.split(experts)

    # Get partition specs from state, keeping only 'ep' for shard_map (tp/fsdp handled by JAX)
    def ep_only(spec):
        return PartitionSpec(*(p if p == 'ep' else None for p in spec)) if isinstance(spec, PartitionSpec) else spec

    state_specs = jax.tree.map(ep_only, nnx.get_partition_spec(state), is_leaf=lambda x: isinstance(x, PartitionSpec))

    P = PartitionSpec
    in_specs = (
        P(), P(), P(), P(),  # hidden_states, selected_experts, routing_weights, adapter_indices
        state_specs,
    )

    def _shard_body(hidden_states, selected_experts, routing_weights, adapter_indices, experts_state):
        axis_idx = jax.lax.axis_index("ep")
        group_offset = jnp.array([axis_idx * experts_per_rank], dtype=jnp.int32)

        # Reconstruct module from state
        experts = nnx.merge(graphdef, experts_state)

        # Prepare routing
        hidden_expanded = jnp.repeat(hidden_states, num_experts_per_tok, axis=0)
        adapter_expanded = jnp.repeat(adapter_indices, num_experts_per_tok) if adapter_indices is not None else None
        hidden_sorted, group_sizes, unsort_indices, adapter_sorted = prepare_routing(
            hidden_expanded, selected_experts.reshape(-1), num_experts, adapter_indices=adapter_expanded
        )

        # Expert computation (model-specific)
        out = forward(experts, hidden_sorted, group_sizes, adapter_sorted, group_offset)

        # Unsort and combine
        out = out[unsort_indices].reshape(-1, num_experts_per_tok, hidden_size)
        local_out = jnp.sum(out * routing_weights[..., None], axis=1)
        return jax.lax.psum(local_out, axis_name="ep")

    sharded_fn = jax.shard_map(
        _shard_body, mesh=mesh, in_specs=in_specs, out_specs=P(),
        axis_names={'ep'},  # Only ep is manual; tp/fsdp handled automatically by JAX
    )
    return sharded_fn(
        hidden_states, selected_experts, routing_weights, adapter_indices,
        state,
    )
