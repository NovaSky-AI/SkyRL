import jax
import jax.numpy as jnp

from skyrl.tx.layers.attention import dot_product_attention


def test_causal_attention_bypasses_jax_mps_fused_path_on_cpu(monkeypatch):
    original_dot_product_attention = jax.nn.dot_product_attention

    def mps_patched_dot_product_attention(query, key, value, *args, implementation=None, **kwargs):
        if implementation is None and kwargs.get("is_causal") and kwargs.get("mask") is not None:
            raise ValueError("fused MPS attention cannot combine a causal and explicit mask")
        return original_dot_product_attention(query, key, value, *args, implementation=implementation, **kwargs)

    monkeypatch.setattr(jax.nn, "dot_product_attention", mps_patched_dot_product_attention)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")

    q = jnp.zeros((1, 3, 1, 1), dtype=jnp.float32)
    v = jnp.array([2.0, 4.0, 32.0]).reshape(q.shape)
    mask = jnp.array([[1, 1, 0]])

    result = dot_product_attention(q, q, v, mask, is_causal=True, head_dim=1)

    assert jnp.allclose(result.ravel(), jnp.array([2.0, 3.0, 3.0]))
