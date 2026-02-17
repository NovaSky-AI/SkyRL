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
    """Test that LoRAConnector produces correct output shapes."""
    with jax.set_mesh(mesh):
        from tx.layers.connectors import LoRAConnector

        hidden_dim = 64
        batch, seq = 2, 8

        conn = LoRAConnector(
            hidden_dim, expansion_rate, max_lora_adapters=4, trainable=True, dtype=jnp.float32, rngs=nnx.Rngs(0)
        )
        adapter_indices = jnp.array([1, 2], dtype=jnp.int32)

        x = jnp.ones((batch, seq, expansion_rate, hidden_dim))
        pre_out, residual_norm = conn.pre(x, adapter_indices)
        post_out = conn.post(x, pre_out, residual_norm, adapter_indices)

        assert pre_out.shape == (batch, seq, hidden_dim)
        assert residual_norm.shape == (batch, seq, expansion_rate * hidden_dim)
        assert post_out.shape == (batch, seq, expansion_rate, hidden_dim)


@pytest.mark.parametrize("expansion_rate", [1, 2, 4])
def test_connector_identity_initialization(mesh, expansion_rate: int):
    """Test that LoRAConnector identity initialization behaves like residual connection per adapter slot."""
    with jax.set_mesh(mesh):
        from tx.layers.connectors import LoRAConnector
        from tx.layers.util import sinkhorn_knopp

        hidden_dim = 64
        n = expansion_rate

        conn = LoRAConnector(hidden_dim, n, max_lora_adapters=3, trainable=True, dtype=jnp.float32, rngs=nnx.Rngs(0))
        adapter_idx = 0
        adapter_indices = jnp.array([adapter_idx], dtype=jnp.int32)

        # Verify H_pre = 1/n
        _, _, _, _, _, _, _, b_pre, _, _ = conn._get_params(adapter_indices)
        h_pre = jax.nn.sigmoid(b_pre[0])
        assert np.allclose(h_pre, 1.0 / n, atol=1e-5)

        # Verify H_post = 1
        _, _, _, _, _, _, _, _, b_post, _ = conn._get_params(adapter_indices)
        h_post = 2.0 * jax.nn.sigmoid(b_post[0])
        assert np.allclose(h_post, 1.0, atol=1e-6)

        # Verify M = I
        _, _, _, _, _, _, _, _, _, b_res = conn._get_params(adapter_indices)
        M = sinkhorn_knopp(b_res[0])
        assert np.allclose(M, jnp.eye(n), atol=1e-3)
