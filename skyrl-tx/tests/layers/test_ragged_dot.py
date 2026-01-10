import jax
import pytest
import jax.numpy as jnp
from tx.layers.util import ragged_dot


@pytest.mark.parametrize(
    "group_sizes,group_offset,g_local,expected_scale",
    [
        ([2, 2, 2], 1, 2, [0, 0, 1, 1, 2, 2]),  # middle shard
        ([2, 2, 2], 0, 2, [1, 1, 2, 2, 0, 0]),  # first shard
        ([2, 2, 2], 2, 1, [0, 0, 0, 0, 1, 1]),  # last shard
        ([6], 0, 1, [1, 1, 1, 1, 1, 1]),  # single group
        ([2, 0, 0, 4], 1, 2, [0, 0, 0, 0, 0, 0]),  # empty groups in shard
        ([1, 3, 2], 1, 2, [0, 1, 1, 1, 2, 2]),  # uneven sizes
    ],
)
def test_ragged_dot_with_group_offset(group_sizes, group_offset, g_local, expected_scale):
    """Test ragged_dot with group_offset for various edge cases."""
    group_sizes = jnp.array(group_sizes)
    m, d = 6, 2

    lhs = jnp.arange(m * d, dtype=jnp.float32).reshape(m, d)
    rhs = jnp.stack([(i + 1) * jnp.eye(d) for i in range(g_local)])  # 1*I, 2*I, ...

    result = jax.jit(ragged_dot)(lhs, rhs, group_sizes, group_offset=jnp.array([group_offset]))

    # expected_scale: 0 for masked tokens, else local_group_idx + 1
    scale = jnp.array(expected_scale, dtype=jnp.float32)[:, None]
    expected = lhs * scale

    assert jnp.allclose(result, expected), f"Got:\n{result}\nExpected:\n{expected}"
