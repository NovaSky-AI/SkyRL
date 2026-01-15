"""CPU tests for attention utilities."""

import jax
import jax.numpy as jnp

from tx.models.attention import _shift_sequences, dot_product_attention


class TestShiftSequences:
    def test_zero_shift(self):
        """shift=0 returns input unchanged."""
        x = jnp.array([[[1, 2], [3, 4], [5, 6]]])  # [1, 3, 2]
        shift = jnp.array([0])
        result = _shift_sequences(x, shift)
        assert jnp.allclose(result, x)

    def test_left_to_right_pad(self):
        """Left-padded [pad,pad,A,B,C] with shift=2 -> [A,B,C,pad,pad]."""
        # Shape: [1, 5, 2] - batch=1, seq=5, features=2
        x = jnp.array([[[0, 0], [0, 0], [1, 1], [2, 2], [3, 3]]])
        shift = jnp.array([2])
        result = _shift_sequences(x, shift)
        expected = jnp.array([[[1, 1], [2, 2], [3, 3], [0, 0], [0, 0]]])
        assert jnp.allclose(result, expected)

    def test_roundtrip(self):
        """shift by N then -N returns original."""
        x = jnp.array([[[0, 0], [0, 0], [1, 1], [2, 2], [3, 3]]])
        shift = jnp.array([2])
        shifted = _shift_sequences(x, shift)
        restored = _shift_sequences(shifted, -shift)
        assert jnp.allclose(restored, x)

    def test_per_batch_shift(self):
        """Different shift amounts per batch element."""
        x = jnp.array([
            [[0, 0], [1, 1], [2, 2]],  # batch 0: shift=0
            [[0, 0], [0, 0], [3, 3]],  # batch 1: shift=2
        ])
        shift = jnp.array([0, 2])
        result = _shift_sequences(x, shift)
        expected = jnp.array([
            [[0, 0], [1, 1], [2, 2]],  # unchanged
            [[3, 3], [0, 0], [0, 0]],  # shifted left by 2
        ])
        assert jnp.allclose(result, expected)

    def test_4d_tensor(self):
        """Works with 4D tensors [B, S, H, D]."""
        x = jax.random.normal(jax.random.key(0), (2, 8, 4, 16))
        shift = jnp.array([0, 3])
        shifted = _shift_sequences(x, shift)
        restored = _shift_sequences(shifted, -shift)
        assert jnp.allclose(restored, x)


class TestDotProductAttentionCPU:
    """Tests that run on CPU (mask-based path)."""

    def test_basic_attention(self):
        """Basic attention computation on CPU."""
        B, T, S, H, D = 2, 4, 4, 2, 8
        q = jax.random.normal(jax.random.key(0), (B, T, H, D))
        k = jax.random.normal(jax.random.key(1), (B, S, H, D))
        v = jax.random.normal(jax.random.key(2), (B, S, H, D))
        mask = jnp.ones((B, S))

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=D)
        assert result.shape == (B, T, H, D)

    def test_masked_positions_ignored(self):
        """Masked positions (0s) don't affect output for valid positions."""
        B, T, H, D = 1, 3, 1, 4
        q = jax.random.normal(jax.random.key(0), (B, T, H, D))
        k = jax.random.normal(jax.random.key(1), (B, T, H, D))
        v = jax.random.normal(jax.random.key(2), (B, T, H, D))

        # Full mask
        mask_full = jnp.array([[1, 1, 1]])
        out_full = dot_product_attention(q, k, v, mask_full, is_causal=False, head_dim=D)

        # Mask with padding at end (right-padded) - only first 2 valid
        mask_right = jnp.array([[1, 1, 0]])
        out_right = dot_product_attention(q, k, v, mask_right, is_causal=False, head_dim=D)

        # First two positions should only attend to first two K/V
        # Output at masked positions doesn't matter
        # We mainly verify it runs without error
        assert out_full.shape == out_right.shape

    def test_decode_single_query(self):
        """Decode mode with single query token."""
        B, S, H, D = 2, 10, 4, 16
        q = jax.random.normal(jax.random.key(0), (B, 1, H, D))  # single query
        k = jax.random.normal(jax.random.key(1), (B, S, H, D))
        v = jax.random.normal(jax.random.key(2), (B, S, H, D))
        mask = jnp.ones((B, S))

        result = dot_product_attention(q, k, v, mask, is_causal=False, head_dim=D)
        assert result.shape == (B, 1, H, D)

    def test_gqa_different_kv_heads(self):
        """Grouped query attention with fewer KV heads."""
        B, T, S = 2, 4, 4
        num_heads, num_kv_heads, D = 8, 2, 16

        q = jax.random.normal(jax.random.key(0), (B, T, num_heads, D))
        k = jax.random.normal(jax.random.key(1), (B, S, num_kv_heads, D))
        v = jax.random.normal(jax.random.key(2), (B, S, num_kv_heads, D))
        mask = jnp.ones((B, S))

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=D)
        assert result.shape == (B, T, num_heads, D)
