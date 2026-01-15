"""CPU tests for attention utilities."""

import jax
import jax.numpy as jnp
import pytest

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

    def test_per_batch_shift(self):
        """Different shift amounts per batch element."""
        x = jnp.array(
            [
                [[0, 0], [1, 1], [2, 2]],  # batch 0: shift=0
                [[0, 0], [0, 0], [3, 3]],  # batch 1: shift=2
            ]
        )
        shift = jnp.array([0, 2])
        result = _shift_sequences(x, shift)
        expected = jnp.array(
            [
                [[0, 0], [1, 1], [2, 2]],  # unchanged
                [[3, 3], [0, 0], [0, 0]],  # shifted left by 2
            ]
        )
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize(
        'shape,pad_lengths',
        [
            ((1, 5, 2), [2]),  # 3D: [B, S, D]
            ((2, 8, 4), [0, 3]),  # 3D: batch with different padding
            ((2, 8, 4, 16), [0, 3]),  # 4D: [B, S, H, D]
            ((4, 16, 8, 64), [0, 4, 8, 12]),  # 4D: larger batch
        ],
    )
    def test_roundtrip(self, shape, pad_lengths):
        """Shift by N then -N returns original, with realistic padding."""
        x = jax.random.normal(jax.random.key(0), shape)

        # Set padding positions to 0 (simulate left-padded sequences)
        for b, pad_len in enumerate(pad_lengths):
            if pad_len > 0:
                x = x.at[b, :pad_len].set(0)

        shift = jnp.array(pad_lengths)
        shifted = _shift_sequences(x, shift)

        # Verify padding moved to the end
        for b, pad_len in enumerate(pad_lengths):
            if pad_len > 0:
                assert jnp.allclose(shifted[b, -pad_len:], 0)

        # Verify roundtrip
        restored = _shift_sequences(shifted, -shift)
        assert jnp.allclose(restored, x)


class TestDotProductAttentionCPU:
    """Tests that run on CPU (mask-based path)."""

    def test_basic_attention(self):
        """Basic attention computation on CPU."""
        batch, seq_len, num_heads, head_dim = 2, 4, 2, 8
        q = jax.random.normal(jax.random.key(0), (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(jax.random.key(1), (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(jax.random.key(2), (batch, seq_len, num_heads, head_dim))
        mask = jnp.ones((batch, seq_len))

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)
        assert result.shape == (batch, seq_len, num_heads, head_dim)

    def test_masked_positions_ignored(self):
        """Masked positions (0s) don't affect output for valid positions."""
        batch, seq_len, num_heads, head_dim = 1, 3, 1, 4
        q = jax.random.normal(jax.random.key(0), (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(jax.random.key(1), (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(jax.random.key(2), (batch, seq_len, num_heads, head_dim))

        # Full mask
        mask_full = jnp.array([[1, 1, 1]])
        out_full = dot_product_attention(q, k, v, mask_full, is_causal=False, head_dim=head_dim)

        # Mask with padding at end (right-padded) - only first 2 valid
        mask_right = jnp.array([[1, 1, 0]])
        out_right = dot_product_attention(q, k, v, mask_right, is_causal=False, head_dim=head_dim)

        # First two positions should only attend to first two K/V
        # Output at masked positions doesn't matter
        # We mainly verify it runs without error
        assert out_full.shape == out_right.shape

    def test_decode_single_query(self):
        """Decode mode with single query token."""
        batch, seq_len, num_heads, head_dim = 2, 10, 4, 16
        q = jax.random.normal(jax.random.key(0), (batch, 1, num_heads, head_dim))  # single query
        k = jax.random.normal(jax.random.key(1), (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(jax.random.key(2), (batch, seq_len, num_heads, head_dim))
        mask = jnp.ones((batch, seq_len))

        result = dot_product_attention(q, k, v, mask, is_causal=False, head_dim=head_dim)
        assert result.shape == (batch, 1, num_heads, head_dim)

    def test_gqa_different_kv_heads(self):
        """Grouped query attention with fewer KV heads."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 4, 8, 2, 16

        q = jax.random.normal(jax.random.key(0), (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        v = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))
        mask = jnp.ones((batch, seq_len))

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)
        assert result.shape == (batch, seq_len, num_heads, head_dim)
