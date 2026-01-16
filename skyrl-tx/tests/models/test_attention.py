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

    def test_masking_excludes_padding(self):
        """Changing values at masked positions doesn't affect output at valid positions."""
        batch, seq_len, num_heads, head_dim = 1, 4, 1, 8
        q = jax.random.normal(jax.random.key(0), (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(jax.random.key(1), (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(jax.random.key(2), (batch, seq_len, num_heads, head_dim))

        # Right-padded: first 2 positions valid, last 2 masked
        mask = jnp.array([[1, 1, 0, 0]])

        out1 = dot_product_attention(q, k, v, mask, is_causal=False, head_dim=head_dim)

        # Modify K/V at masked positions - should not affect output at valid positions
        k_modified = k.at[:, 2:, :, :].set(999.0)
        v_modified = v.at[:, 2:, :, :].set(999.0)
        out2 = dot_product_attention(q, k_modified, v_modified, mask, is_causal=False, head_dim=head_dim)

        # Valid positions (0, 1) should be identical
        assert jnp.allclose(out1[:, :2], out2[:, :2])

    def test_causal_mask_blocks_future(self):
        """With causal masking, position i only attends to positions <= i."""
        batch, seq_len, num_heads, head_dim = 1, 4, 1, 8
        q = jax.random.normal(jax.random.key(0), (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(jax.random.key(1), (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(jax.random.key(2), (batch, seq_len, num_heads, head_dim))
        mask = jnp.ones((batch, seq_len))

        out_causal = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=head_dim)

        # Modify future K/V (positions 2, 3) - should not affect output at position 1
        k_modified = k.at[:, 2:, :, :].set(999.0)
        v_modified = v.at[:, 2:, :, :].set(999.0)
        out_modified = dot_product_attention(q, k_modified, v_modified, mask, is_causal=True, head_dim=head_dim)

        # Positions 0 and 1 should be unaffected by changes to positions 2, 3
        assert jnp.allclose(out_causal[:, :2], out_modified[:, :2])

        # But position 2 should be affected (it can see position 2's K/V)
        assert not jnp.allclose(out_causal[:, 2], out_modified[:, 2])

    def test_decode_attends_to_all_kv(self):
        """In decode mode (is_causal=False), query attends to all K/V positions."""
        batch, kv_len, num_heads, head_dim = 1, 8, 1, 16
        q = jax.random.normal(jax.random.key(0), (batch, 1, num_heads, head_dim))
        k = jax.random.normal(jax.random.key(1), (batch, kv_len, num_heads, head_dim))
        v = jax.random.normal(jax.random.key(2), (batch, kv_len, num_heads, head_dim))
        mask = jnp.ones((batch, kv_len))

        out1 = dot_product_attention(q, k, v, mask, is_causal=False, head_dim=head_dim)

        # Modify last K/V position - should affect output since we attend to all
        k_modified = k.at[:, -1, :, :].set(999.0)
        v_modified = v.at[:, -1, :, :].set(999.0)
        out2 = dot_product_attention(q, k_modified, v_modified, mask, is_causal=False, head_dim=head_dim)

        # Output should be different
        assert not jnp.allclose(out1, out2)

    def test_gqa_head_broadcasting(self):
        """GQA: multiple query heads share the same KV head."""
        batch, seq_len = 1, 4
        num_heads, num_kv_heads, head_dim = 4, 1, 8  # 4 query heads share 1 KV head

        q = jax.random.normal(jax.random.key(0), (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        v = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))
        mask = jnp.ones((batch, seq_len))

        result = dot_product_attention(q, k, v, mask, is_causal=False, head_dim=head_dim)

        # All query heads see the same K/V, so with identical Q they'd produce identical output
        # With different Q per head, outputs differ but shape is correct
        assert result.shape == (batch, seq_len, num_heads, head_dim)

        # Verify output varies across heads (since Q varies)
        assert not jnp.allclose(result[:, :, 0, :], result[:, :, 1, :])
