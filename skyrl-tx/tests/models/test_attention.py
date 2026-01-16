"""CPU tests for attention utilities."""

import jax
import jax.numpy as jnp
import pytest

from tx.models.attention import _shift_sequences


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
        "shape,pad_lengths",
        [
            ((1, 5, 2), [2]),  # 3D: [batch, seq, features]
            ((2, 8, 4), [0, 3]),  # 3D: batch with different padding
            ((2, 8, 4, 16), [0, 3]),  # 4D: [batch, seq, heads, head_dim]
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

    def test_shift_from_mask(self):
        """argmax on attention_mask gives correct shift amounts."""
        # Right-padded: [1,1,1,0,0] -> first valid at 0, shift=0
        mask_right = jnp.array([[1, 1, 1, 0, 0]])
        assert jnp.argmax(mask_right, axis=1)[0] == 0

        # Left-padded: [0,0,1,1,1] -> first valid at 2, shift=2
        mask_left = jnp.array([[0, 0, 1, 1, 1]])
        assert jnp.argmax(mask_left, axis=1)[0] == 2

        # Mixed batch
        mask_mixed = jnp.array(
            [
                [1, 1, 1, 0, 0],  # right-padded, shift=0
                [0, 0, 1, 1, 1],  # left-padded, shift=2
                [0, 0, 0, 1, 1],  # left-padded, shift=3
            ]
        )
        shifts = jnp.argmax(mask_mixed, axis=1)
        assert list(shifts) == [0, 2, 3]
