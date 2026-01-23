"""GPU tests for flash attention.

These tests require a GPU and verify that cuDNN flash attention produces
numerically equivalent results to the mask-based implementation.
"""

import jax
import jax.numpy as jnp
import pytest

from tx.layers.attention import dot_product_attention

# Skip all tests if not on GPU
pytestmark = pytest.mark.skipif(jax.default_backend() != "gpu", reason="GPU tests require CUDA")


def make_qkv(batch, seq_len, num_heads, head_dim, num_kv_heads=None, dtype=jnp.bfloat16):
    """Create random Q, K, V tensors."""
    if num_kv_heads is None:
        num_kv_heads = num_heads
    q = jax.random.normal(jax.random.key(0), (batch, seq_len, num_heads, head_dim), dtype=dtype)
    k = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim), dtype=dtype)
    v = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim), dtype=dtype)
    return q, k, v


def make_right_padded_mask(batch, seq_len, seq_lengths):
    """Create right-padded mask: [1,1,1,...,0,0]."""
    seq_lengths = jnp.array(seq_lengths)
    return (jnp.arange(seq_len)[None, :] < seq_lengths[:, None]).astype(jnp.float32)


def mask_based_attention(q, k, v, mask, is_causal, head_dim):
    """Reference implementation using mask-based attention."""
    scale = 1.0 / head_dim**0.5
    return jax.nn.dot_product_attention(
        q, k, v, scale=scale, mask=mask[:, None, None, :].astype(bool), is_causal=is_causal
    )


class TestFlashAttention:
    """Verify cuDNN flash attention matches mask-based attention."""

    @pytest.mark.parametrize("seq_len", [32, 128, 512])
    def test_padded_equivalence(self, seq_len):
        """cuDNN matches mask-based for right-padded sequences."""
        batch, num_heads, head_dim = 2, 4, 64
        q, k, v = make_qkv(batch, seq_len, num_heads, head_dim)

        seq_lengths = [seq_len - 4, seq_len - 8]
        mask = make_right_padded_mask(batch, seq_len, seq_lengths)

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)
        expected = mask_based_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)

        # Check valid positions match
        for b, length in enumerate(seq_lengths):
            assert jnp.allclose(result[b, :length], expected[b, :length], atol=1e-5), f"Mismatch at batch {b}"

    def test_no_padding(self):
        """Full sequences (no padding) work correctly."""
        batch, seq_len, num_heads, head_dim = 2, 64, 4, 64
        q, k, v = make_qkv(batch, seq_len, num_heads, head_dim)
        mask = jnp.ones((batch, seq_len))

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)
        expected = mask_based_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)
        assert jnp.allclose(result, expected, atol=1e-5)

    @pytest.mark.parametrize(
        "seq_lengths",
        [
            [128, 96, 64, 32],  # decreasing lengths
            [32, 64, 96, 128],  # increasing lengths
            [128, 128, 128, 128],  # all full (no padding)
            [1, 1, 1, 1],  # minimal valid sequences
        ],
    )
    def test_mixed_seq_lengths(self, seq_lengths):
        """Batch with varying sequence lengths."""
        batch, seq_len, num_heads, head_dim = 4, 128, 4, 64
        q, k, v = make_qkv(batch, seq_len, num_heads, head_dim)
        mask = make_right_padded_mask(batch, seq_len, seq_lengths)

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)
        expected = mask_based_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)

        # Check valid positions match
        for b, length in enumerate(seq_lengths):
            assert jnp.allclose(result[b, :length], expected[b, :length], atol=1e-5), f"Mismatch at batch {b}"

    def test_gqa(self):
        """Grouped query attention (8 Q heads, 2 KV heads)."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 64, 8, 2, 64
        q, k, v = make_qkv(batch, seq_len, num_heads, head_dim, num_kv_heads)
        mask = jnp.ones((batch, seq_len))

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)
        expected = mask_based_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_decode(self):
        """Decode mode (is_causal=False, single query token)."""
        batch, kv_len, num_heads, head_dim = 2, 128, 4, 64
        q = jax.random.normal(jax.random.key(0), (batch, 1, num_heads, head_dim), dtype=jnp.bfloat16)
        k = jax.random.normal(jax.random.key(1), (batch, kv_len, num_heads, head_dim), dtype=jnp.bfloat16)
        v = jax.random.normal(jax.random.key(2), (batch, kv_len, num_heads, head_dim), dtype=jnp.bfloat16)
        mask = make_right_padded_mask(batch, kv_len, [100, 80])

        result = dot_product_attention(q, k, v, mask, is_causal=False, head_dim=head_dim)
        expected = mask_based_attention(q, k, v, mask, is_causal=False, head_dim=head_dim)
        assert jnp.allclose(result, expected, atol=1e-5)
