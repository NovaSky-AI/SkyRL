import jax.numpy as jnp
from tx.layers.util import ragged_dot


def test_ragged_dot_with_group_offset():
    """Test ragged_dot with group_offset using a simple hand-verifiable example.

    Setup:
    - 6 tokens, 2 features each
    - 3 global groups with sizes [2, 2, 2]
    - Local rhs contains only groups 1 and 2 (g_local=2)
    - group_offset=1 means we start at global group 1

    Expected behavior:
    - Tokens 0-1 (global group 0): masked to 0 (not in our shard)
    - Tokens 2-3 (global group 1): use rhs[0] (local group 0)
    - Tokens 4-5 (global group 2): use rhs[1] (local group 1)
    """
    # lhs: 6 tokens, 2 features - use simple values
    # Token values: [[1,0], [0,1], [1,1], [2,0], [0,2], [1,2]]
    lhs = jnp.array([
        [1.0, 0.0],  # token 0 (group 0) - should be masked
        [0.0, 1.0],  # token 1 (group 0) - should be masked
        [1.0, 1.0],  # token 2 (group 1) - use rhs[0]
        [2.0, 0.0],  # token 3 (group 1) - use rhs[0]
        [0.0, 2.0],  # token 4 (group 2) - use rhs[1]
        [1.0, 2.0],  # token 5 (group 2) - use rhs[1]
    ])

    # rhs: 2 local groups, 2x2 weight matrices
    # rhs[0] (for global group 1): identity matrix
    # rhs[1] (for global group 2): [[1,0], [0,2]] (scales second output by 2)
    rhs = jnp.array([
        [[1.0, 0.0],   # rhs[0]: identity
         [0.0, 1.0]],
        [[1.0, 0.0],   # rhs[1]: scale y by 2
         [0.0, 2.0]],
    ])

    group_sizes = jnp.array([2, 2, 2])
    group_offset = jnp.array([1])

    result = ragged_dot(lhs, rhs, group_sizes, group_offset=group_offset)

    # Expected results:
    # Token 0: masked -> [0, 0]
    # Token 1: masked -> [0, 0]
    # Token 2: [1,1] @ identity = [1, 1]
    # Token 3: [2,0] @ identity = [2, 0]
    # Token 4: [0,2] @ [[1,0],[0,2]] = [0, 4]
    # Token 5: [1,2] @ [[1,0],[0,2]] = [1, 4]
    expected = jnp.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [0.0, 4.0],
        [1.0, 4.0],
    ])

    assert jnp.allclose(result, expected), f"Got:\n{result}\nExpected:\n{expected}"
    print("Test passed!")


def test_ragged_dot_no_offset():
    """Test that ragged_dot without offset works normally."""
    lhs = jnp.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    # 3 groups, 1 token each
    rhs = jnp.array([
        [[2.0, 0.0], [0.0, 2.0]],  # group 0: scale by 2
        [[1.0, 0.0], [0.0, 1.0]],  # group 1: identity
        [[0.0, 1.0], [1.0, 0.0]],  # group 2: swap
    ])

    group_sizes = jnp.array([1, 1, 1])

    result = ragged_dot(lhs, rhs, group_sizes)

    # Token 0: [1,0] @ scale2 = [2, 0]
    # Token 1: [0,1] @ identity = [0, 1]
    # Token 2: [1,1] @ swap = [1, 1]
    expected = jnp.array([
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    assert jnp.allclose(result, expected), f"Got:\n{result}\nExpected:\n{expected}"
    print("Test passed!")


if __name__ == "__main__":
    test_ragged_dot_no_offset()
    test_ragged_dot_with_group_offset()
    print("All tests passed!")
