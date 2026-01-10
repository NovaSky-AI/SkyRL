from flax import nnx
import jax
from jax import lax
from jax import numpy as jnp


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
