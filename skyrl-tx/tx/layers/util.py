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


def _replicated_spec_like(x: jax.Array) -> PartitionSpec:
    """Return a PartitionSpec that mirrors an array's sharding but omits 'ep'."""
    spec = getattr(getattr(x, "sharding", None), "spec", (None,) * x.ndim)
    return PartitionSpec(*[
        tuple(a for a in s if a != "ep") or None if isinstance(s, tuple) else (None if s == "ep" else s)
        for s in spec
    ])


def _local_expert_computation(
    hidden_states: jax.Array,
    selected_experts: jax.Array,
    routing_weights: jax.Array,
    expert_fn,
    num_experts: int,
    num_experts_per_tok: int,
    hidden_size: int,
    adapter_indices: jax.Array | None = None,
    expert_kwargs: dict | None = None,
) -> jax.Array:
    """Run expert computation locally without expert-parallel sharding."""
    hidden_states_expanded = jnp.repeat(hidden_states, num_experts_per_tok, axis=0)
    adapter_indices_expanded = (
        jnp.repeat(adapter_indices, num_experts_per_tok) if adapter_indices is not None else None
    )
    hidden_states_sorted, group_sizes, unsort_indices, adapter_indices_sorted = prepare_routing(
        hidden_states_expanded,
        selected_experts.reshape(-1),
        num_experts,
        adapter_indices=adapter_indices_expanded,
    )

    expert_out = expert_fn(hidden_states_sorted, group_sizes, adapter_indices_sorted, **(expert_kwargs or {}))
    reshaped_out = expert_out[unsort_indices].reshape(-1, num_experts_per_tok, hidden_size)
    return jnp.sum(reshaped_out * routing_weights[..., None], axis=1)


def expert_parallel_dispatch_combine(
    hidden_states: jax.Array,
    selected_experts: jax.Array,
    routing_weights: jax.Array,
    expert_fn,
    num_experts: int,
    num_experts_per_tok: int,
    hidden_size: int,
    adapter_indices: jax.Array | None = None,
) -> jax.Array:
    """Dispatch tokens to experts and combine outputs, optionally across EP shards."""
    mesh = get_abstract_mesh()
    ep_size = mesh.shape.get("ep", 1) if mesh is not None else 1

    if ep_size == 1:
        return _local_expert_computation(
            hidden_states, selected_experts, routing_weights, expert_fn,
            num_experts, num_experts_per_tok, hidden_size, adapter_indices
        )

    assert num_experts % ep_size == 0, f"num_experts {num_experts} not divisible by ep {ep_size}"
    experts_per_rank = num_experts // ep_size

    def _shard_body(payload):
        shard_h, shard_s, shard_r, *opt_adapter = payload
        shard_a = opt_adapter[0] if opt_adapter else None

        axis_idx = jax.lax.axis_index("ep")
        shard_start = axis_idx * experts_per_rank
        shard_end = shard_start + experts_per_rank

        # Mask out experts not on this rank
        local_mask = (shard_s >= shard_start) & (shard_s < shard_end)
        local_selected = jnp.where(local_mask, shard_s - shard_start, 0)
        local_routing = shard_r * local_mask.astype(shard_r.dtype)

        local_out = _local_expert_computation(
            shard_h, local_selected, local_routing, expert_fn,
            experts_per_rank, num_experts_per_tok, hidden_size, shard_a,
            expert_kwargs={"expert_start": jax.lax.stop_gradient(shard_start), "num_experts_chunk": experts_per_rank}
        )
        return jax.lax.psum(local_out, axis_name="ep")

    # Pack args (handle optional adapter_indices to keep specs clean)
    args = (hidden_states, selected_experts, routing_weights)
    if adapter_indices is not None:
        args += (adapter_indices,)

    sharded_fn = jax.shard_map(
        _shard_body,
        mesh=mesh,
        in_specs=(jax.tree.map(_replicated_spec_like, args),),
        out_specs=_replicated_spec_like(hidden_states),
        axis_names={"ep"},
    )
    return sharded_fn(args)
