"""GPU tests for flash attention.

These tests require a GPU and verify that cuDNN flash attention produces
numerically equivalent results to the mask-based implementation.
"""

import jax
import jax.numpy as jnp
import pytest

from tx.models.attention import dot_product_attention

# Skip all tests if not on GPU
pytestmark = pytest.mark.skipif(jax.default_backend() != "gpu", reason="GPU tests require CUDA")


def mask_based_attention(q, k, v, attention_mask, is_causal, head_dim):
    """Reference implementation using mask-based attention."""
    scale = 1.0 / head_dim**0.5
    return jax.nn.dot_product_attention(
        q, k, v, scale=scale, mask=attention_mask[:, None, None, :].astype(bool), is_causal=is_causal
    )


class TestFlashAttentionNumericalEquivalence:
    """Verify cuDNN flash attention matches mask-based attention."""

    @pytest.mark.parametrize("seq_len", [32, 128, 512])
    def test_right_padded_equivalence(self, seq_len):
        """cuDNN matches mask-based for right-padded sequences."""
        batch, num_heads, head_dim = 2, 4, 64
        q = jax.random.normal(jax.random.key(0), (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(jax.random.key(1), (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(jax.random.key(2), (batch, seq_len, num_heads, head_dim))

        # Right-padded: [1,1,1,...,0,0]
        seq_lengths = jnp.array([seq_len - 4, seq_len - 8])
        mask = (jnp.arange(seq_len)[None, :] < seq_lengths[:, None]).astype(jnp.float32)

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)
        expected = mask_based_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)

        # Check only valid positions (masked positions may differ)
        for b in range(batch):
            valid_len = int(seq_lengths[b])
            assert jnp.allclose(result[b, :valid_len], expected[b, :valid_len], atol=1e-5), f"Mismatch at batch {b}"

    @pytest.mark.parametrize("seq_len", [32, 128, 512])
    def test_left_padded_equivalence(self, seq_len):
        """cuDNN matches mask-based for left-padded sequences (prefill)."""
        batch, num_heads, head_dim = 2, 4, 64
        q = jax.random.normal(jax.random.key(0), (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(jax.random.key(1), (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(jax.random.key(2), (batch, seq_len, num_heads, head_dim))

        # Left-padded: [0,0,...,1,1,1]
        seq_lengths = jnp.array([seq_len - 4, seq_len - 8])
        padding = seq_len - seq_lengths
        mask = (jnp.arange(seq_len)[None, :] >= padding[:, None]).astype(jnp.float32)

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)
        expected = mask_based_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)

        # Check only valid positions
        for b in range(batch):
            pad_len = int(padding[b])
            assert jnp.allclose(result[b, pad_len:], expected[b, pad_len:], atol=1e-5), f"Mismatch at batch {b}"

    def test_full_sequence_no_padding(self):
        """All-ones mask (no padding) works correctly."""
        batch, seq_len, num_heads, head_dim = 2, 64, 4, 64
        q = jax.random.normal(jax.random.key(0), (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(jax.random.key(1), (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(jax.random.key(2), (batch, seq_len, num_heads, head_dim))
        mask = jnp.ones((batch, seq_len))

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)
        expected = mask_based_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)

        assert jnp.allclose(result, expected, atol=1e-5)

    def test_mixed_seq_lengths_batch(self):
        """Batch with varying sequence lengths."""
        batch, seq_len, num_heads, head_dim = 4, 128, 4, 64
        q = jax.random.normal(jax.random.key(0), (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(jax.random.key(1), (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(jax.random.key(2), (batch, seq_len, num_heads, head_dim))

        # Left-padded with different lengths
        seq_lengths = jnp.array([128, 96, 64, 32])
        padding = seq_len - seq_lengths
        mask = (jnp.arange(seq_len)[None, :] >= padding[:, None]).astype(jnp.float32)

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)
        expected = mask_based_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)

        for b in range(batch):
            pad_len = int(padding[b])
            assert jnp.allclose(result[b, pad_len:], expected[b, pad_len:], atol=1e-5), f"Mismatch at batch {b}"


class TestFlashAttentionGQA:
    """Test grouped query attention with cuDNN."""

    def test_gqa_flash_attention(self):
        """GQA with num_kv_heads < num_heads."""
        batch, seq_len = 2, 64
        num_heads, num_kv_heads, head_dim = 8, 2, 64

        q = jax.random.normal(jax.random.key(0), (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        v = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))
        mask = jnp.ones((batch, seq_len))

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)
        expected = mask_based_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)

        assert jnp.allclose(result, expected, atol=1e-5)


class TestDecodePathOnGPU:
    """Verify decode (is_causal=False) uses mask-based even on GPU."""

    def test_decode_uses_mask_path(self):
        """Decode mode should work correctly on GPU."""
        batch, kv_len, num_heads, head_dim = 2, 128, 4, 64
        q = jax.random.normal(jax.random.key(0), (batch, 1, num_heads, head_dim))
        k = jax.random.normal(jax.random.key(1), (batch, kv_len, num_heads, head_dim))
        v = jax.random.normal(jax.random.key(2), (batch, kv_len, num_heads, head_dim))

        # Left-padded mask
        seq_lengths = jnp.array([100, 80])
        padding = kv_len - seq_lengths
        mask = (jnp.arange(kv_len)[None, :] >= padding[:, None]).astype(jnp.float32)

        result = dot_product_attention(q, k, v, mask, is_causal=False, head_dim=head_dim)
        expected = mask_based_attention(q, k, v, mask, is_causal=False, head_dim=head_dim)

        assert jnp.allclose(result, expected, atol=1e-5)
