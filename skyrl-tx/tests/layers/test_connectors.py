import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx


@pytest.fixture(scope="module")
def mesh():
    return jax.make_mesh((1, 1, 1), ("fsdp", "ep", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 3)


@pytest.mark.parametrize("expansion_rate", [1, 2, 4])
def test_connector_shapes(mesh, expansion_rate: int):
    """Test that Connector produces correct output shapes."""
    with jax.set_mesh(mesh):
        from tx.layers.connectors import Connector

        hidden_dim = 64
        batch, seq = 2, 8

        conn = Connector(hidden_dim, expansion_rate, dtype=jnp.float32, rngs=nnx.Rngs(0))

        x = jnp.ones((batch, seq, expansion_rate, hidden_dim))
        pre_out = conn.pre(x)
        post_out = conn.post(x, pre_out)

        assert pre_out.shape == (batch, seq, hidden_dim)
        assert post_out.shape == (batch, seq, expansion_rate, hidden_dim)


@pytest.mark.parametrize("expansion_rate", [1, 2, 4])
def test_connector_identity_initialization(mesh, expansion_rate: int):
    """Test that Connector with identity initialization behaves like residual connection."""
    with jax.set_mesh(mesh):
        from tx.layers.connectors import Connector
        from tx.layers.util import sinkhorn_knopp

        hidden_dim = 64
        n = expansion_rate

        conn = Connector(hidden_dim, n, dtype=jnp.float32, rngs=nnx.Rngs(0))

        # Verify H_pre = 1/n
        _, _, _, _, _, _, b_pre, _, _ = conn._get_params()
        h_pre = jax.nn.sigmoid(b_pre)
        assert np.allclose(h_pre, 1.0 / n, atol=1e-5)

        # Verify H_post = 1
        _, _, _, _, _, _, _, b_post, _ = conn._get_params()
        h_post = 2.0 * jax.nn.sigmoid(b_post)
        assert np.allclose(h_post, 1.0, atol=1e-6)

        # Verify M = I
        _, _, _, _, _, _, _, _, b_res = conn._get_params()
        M = sinkhorn_knopp(b_res)
        assert np.allclose(M, jnp.eye(n), atol=1e-3)

