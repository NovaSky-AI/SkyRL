from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import get_abstract_mesh, PartitionSpec


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


def _run_expert_logic(
    module: nnx.Module,
    expert_op,
    x: jax.Array,
    selected_experts: jax.Array,
    routing_weights: jax.Array,
    num_experts: int,
    num_experts_per_tok: int,
    hidden_size: int,
    adapter_indices: jax.Array | None,
) -> jax.Array:
    """Run the routing and expert computation on the current device."""
    # Expand inputs for routing
    x_expanded = jnp.repeat(x, num_experts_per_tok, axis=0)
    adapter_expanded = jnp.repeat(adapter_indices, num_experts_per_tok) if adapter_indices is not None else None

    # Prepare routing
    x_sorted, group_sizes, unsort_indices, adapter_sorted = prepare_routing(
        x_expanded, selected_experts.reshape(-1), num_experts, adapter_indices=adapter_expanded
    )

    # Call the expert operation
    expert_out = expert_op(module, x_sorted, group_sizes, adapter_sorted)

    # Unsort and combine
    reshaped_out = expert_out[unsort_indices].reshape(-1, num_experts_per_tok, hidden_size)
    return jnp.sum(reshaped_out * routing_weights[..., None], axis=1)


def expert_parallel_dispatch_combine(
    module: nnx.Module,
    expert_op,
    hidden_states: jax.Array,
    selected_experts: jax.Array,
    routing_weights: jax.Array,
    num_experts: int,
    num_experts_per_tok: int,
    hidden_size: int,
    adapter_indices: jax.Array | None = None,
) -> jax.Array:
    """Dispatch tokens to experts and combine outputs.

    Supports automatic EP sharding via nnx.split/shard_map.

    Args:
        module: The expert module (e.g., Qwen3Experts) to split and shard
        expert_op: A function (module, x, groups, adapter_idx) -> output
        hidden_states: Input hidden states
        selected_experts: Top-k expert indices per token
        routing_weights: Routing weights per token
        num_experts: Total number of experts (global)
        num_experts_per_tok: Number of experts selected per token
        hidden_size: Hidden dimension size
        adapter_indices: Optional adapter indices for LoRA
    """
    mesh = get_abstract_mesh()
    ep_size = mesh.shape.get("ep", 1) if mesh is not None else 1

    # Functionalize the module: separate static Graph from State (weights)
    graph, state = nnx.split(module)

    if ep_size == 1:
        # Local execution - merge back and run
        local_module = nnx.merge(graph, state)
        return _run_expert_logic(
            local_module, expert_op, hidden_states, selected_experts,
            routing_weights, num_experts, num_experts_per_tok, hidden_size, adapter_indices
        )

    # Calculate experts per rank
    assert num_experts % ep_size == 0, f"num_experts {num_experts} not divisible by ep {ep_size}"
    experts_per_rank = num_experts // ep_size

    def _shard_body(local_state, x, s_exp, r_weights, adapt_idx):
        # Merge the LOCAL state shard back into the graph
        # The module now thinks it only has `experts_per_rank` experts
        local_module = nnx.merge(graph, local_state)

        # Calculate local routing info
        axis_idx = jax.lax.axis_index("ep")
        shard_start = axis_idx * experts_per_rank
        shard_end = shard_start + experts_per_rank

        # Mask out experts not on this rank and map global expert IDs to local IDs [0, experts_per_rank)
        local_mask = (s_exp >= shard_start) & (s_exp < shard_end)
        local_selected = jnp.where(local_mask, s_exp - shard_start, 0)

        # Zero out weights for invalid experts so they don't affect the sum
        local_routing = r_weights * local_mask.astype(r_weights.dtype)

        # Run computation
        local_out = _run_expert_logic(
            local_module, expert_op, x, local_selected,
            local_routing, experts_per_rank, num_experts_per_tok, hidden_size, adapt_idx
        )

        # Sum results across EP dimension
        return jax.lax.psum(local_out, axis_name="ep")

    # JAX automatically shards the 'state' based on its PartitionSpecs
    # We replicate the inputs (hidden_states, etc) across the 'ep' axis
    return jax.shard_map(
        _shard_body,
        mesh=mesh,
        in_specs=(
            nnx.get_partition_spec(state),  # State follows its defined partitioning (EP)
            PartitionSpec(),                 # Replicate x
            PartitionSpec(),                 # Replicate selected_experts
            PartitionSpec(),                 # Replicate routing_weights
            PartitionSpec(),                 # Replicate adapter_indices
        ),
        out_specs=PartitionSpec(),  # Output is replicated
    )(state, hidden_states, selected_experts, routing_weights, adapter_indices)
