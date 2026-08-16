import jax
import jax.numpy as jnp

from skyrl.tx.layers.attention import dot_product_attention


def test_causal_attention_bypasses_jax_mps_fused_path_on_cpu(monkeypatch):
    original_dot_product_attention = jax.nn.dot_product_attention

    def mps_patched_dot_product_attention(query, key, value, *args, implementation=None, **kwargs):
        if implementation is None and kwargs.get("is_causal") and kwargs.get("mask") is not None:
            raise ValueError("fused MPS attention cannot combine a causal and explicit mask")
        return original_dot_product_attention(query, key, value, *args, implementation=implementation, **kwargs)

    setattr(mps_patched_dot_product_attention, "_mps_patched", True)
    monkeypatch.setattr(jax.nn, "dot_product_attention", mps_patched_dot_product_attention)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")

    q = jnp.ones((1, 2, 1, 4), dtype=jnp.float32)
    mask = jnp.ones((1, 2), dtype=jnp.int32)

    result = dot_product_attention(q, q, q, mask, is_causal=True, head_dim=4)
    expected = original_dot_product_attention(
        q,
        q,
        q,
        mask=mask[:, None, None, :].astype(bool),
        is_causal=True,
        scale=0.5,
        implementation="xla",
    )

    assert jnp.allclose(result, expected)
