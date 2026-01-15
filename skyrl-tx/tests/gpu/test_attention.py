"""GPU tests for flash attention.

These tests require a GPU and verify that cuDNN flash attention produces
numerically equivalent results to the mask-based implementation.
"""

import jax
import jax.numpy as jnp
import pytest

from tx.models.attention import dot_product_attention

# Skip all tests if not on GPU
pytestmark = pytest.mark.skipif(
    jax.default_backend() != 'gpu',
    reason='GPU tests require CUDA'
)


def mask_based_attention(q, k, v, attention_mask, is_causal, head_dim):
    """Reference implementation using mask-based attention."""
    scale = 1.0 / head_dim**0.5
    return jax.nn.dot_product_attention(
        q, k, v, scale=scale,
        mask=attention_mask[:, None, None, :].astype(bool),
        is_causal=is_causal
    )


class TestFlashAttentionNumericalEquivalence:
    """Verify cuDNN flash attention matches mask-based attention."""

    @pytest.mark.parametrize('seq_len', [32, 128, 512])
    def test_right_padded_equivalence(self, seq_len):
        """cuDNN matches mask-based for right-padded sequences."""
        B, H, D = 2, 4, 64
        q = jax.random.normal(jax.random.key(0), (B, seq_len, H, D))
        k = jax.random.normal(jax.random.key(1), (B, seq_len, H, D))
        v = jax.random.normal(jax.random.key(2), (B, seq_len, H, D))

        # Right-padded: [1,1,1,...,0,0]
        seq_lengths = jnp.array([seq_len - 4, seq_len - 8])
        mask = (jnp.arange(seq_len)[None, :] < seq_lengths[:, None]).astype(jnp.float32)

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=D)
        expected = mask_based_attention(q, k, v, mask, is_causal=True, head_dim=D)

        # Check only valid positions (masked positions may differ)
        for b in range(B):
            valid_len = int(seq_lengths[b])
            assert jnp.allclose(
                result[b, :valid_len], expected[b, :valid_len], atol=1e-5
            ), f'Mismatch at batch {b}'

    @pytest.mark.parametrize('seq_len', [32, 128, 512])
    def test_left_padded_equivalence(self, seq_len):
        """cuDNN matches mask-based for left-padded sequences (prefill)."""
        B, H, D = 2, 4, 64
        q = jax.random.normal(jax.random.key(0), (B, seq_len, H, D))
        k = jax.random.normal(jax.random.key(1), (B, seq_len, H, D))
        v = jax.random.normal(jax.random.key(2), (B, seq_len, H, D))

        # Left-padded: [0,0,...,1,1,1]
        seq_lengths = jnp.array([seq_len - 4, seq_len - 8])
        padding = seq_len - seq_lengths
        mask = (jnp.arange(seq_len)[None, :] >= padding[:, None]).astype(jnp.float32)

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=D)
        expected = mask_based_attention(q, k, v, mask, is_causal=True, head_dim=D)

        # Check only valid positions
        for b in range(B):
            pad_len = int(padding[b])
            assert jnp.allclose(
                result[b, pad_len:], expected[b, pad_len:], atol=1e-5
            ), f'Mismatch at batch {b}'

    def test_full_sequence_no_padding(self):
        """All-ones mask (no padding) works correctly."""
        B, T, H, D = 2, 64, 4, 64
        q = jax.random.normal(jax.random.key(0), (B, T, H, D))
        k = jax.random.normal(jax.random.key(1), (B, T, H, D))
        v = jax.random.normal(jax.random.key(2), (B, T, H, D))
        mask = jnp.ones((B, T))

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=D)
        expected = mask_based_attention(q, k, v, mask, is_causal=True, head_dim=D)

        assert jnp.allclose(result, expected, atol=1e-5)

    def test_mixed_seq_lengths_batch(self):
        """Batch with varying sequence lengths."""
        B, T, H, D = 4, 128, 4, 64
        q = jax.random.normal(jax.random.key(0), (B, T, H, D))
        k = jax.random.normal(jax.random.key(1), (B, T, H, D))
        v = jax.random.normal(jax.random.key(2), (B, T, H, D))

        # Left-padded with different lengths
        seq_lengths = jnp.array([128, 96, 64, 32])
        padding = T - seq_lengths
        mask = (jnp.arange(T)[None, :] >= padding[:, None]).astype(jnp.float32)

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=D)
        expected = mask_based_attention(q, k, v, mask, is_causal=True, head_dim=D)

        for b in range(B):
            pad_len = int(padding[b])
            assert jnp.allclose(
                result[b, pad_len:], expected[b, pad_len:], atol=1e-5
            ), f'Mismatch at batch {b}'


class TestFlashAttentionGQA:
    """Test grouped query attention with cuDNN."""

    def test_gqa_flash_attention(self):
        """GQA with num_kv_heads < num_heads."""
        B, T = 2, 64
        num_heads, num_kv_heads, D = 8, 2, 64

        q = jax.random.normal(jax.random.key(0), (B, T, num_heads, D))
        k = jax.random.normal(jax.random.key(1), (B, T, num_kv_heads, D))
        v = jax.random.normal(jax.random.key(2), (B, T, num_kv_heads, D))
        mask = jnp.ones((B, T))

        result = dot_product_attention(q, k, v, mask, is_causal=True, head_dim=D)
        expected = mask_based_attention(q, k, v, mask, is_causal=True, head_dim=D)

        assert jnp.allclose(result, expected, atol=1e-5)


class TestDecodePathOnGPU:
    """Verify decode (is_causal=False) uses mask-based even on GPU."""

    def test_decode_uses_mask_path(self):
        """Decode mode should work correctly on GPU."""
        B, S, H, D = 2, 128, 4, 64
        q = jax.random.normal(jax.random.key(0), (B, 1, H, D))
        k = jax.random.normal(jax.random.key(1), (B, S, H, D))
        v = jax.random.normal(jax.random.key(2), (B, S, H, D))

        # Left-padded mask
        seq_lengths = jnp.array([100, 80])
        padding = S - seq_lengths
        mask = (jnp.arange(S)[None, :] >= padding[:, None]).astype(jnp.float32)

        result = dot_product_attention(q, k, v, mask, is_causal=False, head_dim=D)
        expected = mask_based_attention(q, k, v, mask, is_causal=False, head_dim=D)

        assert jnp.allclose(result, expected, atol=1e-5)
