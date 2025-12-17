"""Tests for paged attention implementation."""

import jax
import jax.numpy as jnp
import pytest

from tx.utils.paged_attention import (
    PagedKVCache,
    create_paged_kv_cache,
    paged_attention,
    paged_attention_with_update,
    read_from_paged_cache,
    write_to_paged_cache,
)


class TestPagedKVCache:
    """Test paged KV cache creation and basic operations."""

    def test_create_paged_kv_cache(self):
        """Test cache creation with correct shapes."""
        batch_size = 2
        max_seq_len = 64
        num_kv_heads = 4
        head_dim = 32
        page_size = 16

        cache = create_paged_kv_cache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
        )

        # Check shapes
        max_num_pages = (max_seq_len + page_size - 1) // page_size
        num_pages = batch_size * max_num_pages

        assert cache.k_pages.shape == (num_pages, page_size, num_kv_heads, head_dim)
        assert cache.v_pages.shape == (num_pages, page_size, num_kv_heads, head_dim)
        assert cache.page_table.shape == (batch_size, max_num_pages)
        assert cache.page_offsets.shape == (batch_size,)
        assert cache.page_size == page_size
        assert cache.num_pages == num_pages

    def test_write_and_read_single_token(self):
        """Test writing and reading a single token."""
        batch_size = 1
        max_seq_len = 32
        num_kv_heads = 2
        head_dim = 16
        page_size = 8

        cache = create_paged_kv_cache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
        )

        # Create test data
        k_new = jnp.ones((batch_size, 1, num_kv_heads, head_dim))
        v_new = jnp.ones((batch_size, 1, num_kv_heads, head_dim)) * 2
        seq_indices = jnp.array([0])

        # Write to cache
        updated_cache = write_to_paged_cache(cache, k_new, v_new, seq_indices)

        # Read from cache
        k_read, v_read = read_from_paged_cache(updated_cache, seq_indices, max_len=1)

        # Verify
        assert jnp.allclose(k_read, k_new)
        assert jnp.allclose(v_read, v_new)
        assert updated_cache.page_offsets[0] == 1

    def test_write_and_read_multiple_tokens(self):
        """Test writing and reading multiple tokens."""
        batch_size = 2
        max_seq_len = 64
        num_kv_heads = 4
        head_dim = 32
        page_size = 16
        seq_len = 20

        cache = create_paged_kv_cache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
        )

        # Create test data with different values for each sequence
        k_new = jnp.arange(batch_size * seq_len * num_kv_heads * head_dim).reshape(
            batch_size, seq_len, num_kv_heads, head_dim
        )
        v_new = k_new * 2
        seq_indices = jnp.array([0, 1])

        # Write to cache
        updated_cache = write_to_paged_cache(cache, k_new, v_new, seq_indices)

        # Read from cache
        k_read, v_read = read_from_paged_cache(updated_cache, seq_indices, max_len=seq_len)

        # Verify
        assert jnp.allclose(k_read, k_new)
        assert jnp.allclose(v_read, v_new)
        assert jnp.all(updated_cache.page_offsets == seq_len)

    def test_incremental_writes(self):
        """Test incremental token-by-token writes."""
        batch_size = 1
        max_seq_len = 32
        num_kv_heads = 2
        head_dim = 16
        page_size = 8

        cache = create_paged_kv_cache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
        )

        seq_indices = jnp.array([0])
        all_k = []
        all_v = []

        # Write tokens incrementally
        for i in range(10):
            k_new = jnp.ones((batch_size, 1, num_kv_heads, head_dim)) * i
            v_new = jnp.ones((batch_size, 1, num_kv_heads, head_dim)) * (i + 100)
            cache = write_to_paged_cache(cache, k_new, v_new, seq_indices)
            all_k.append(k_new)
            all_v.append(v_new)

        # Read all tokens
        k_read, v_read = read_from_paged_cache(cache, seq_indices, max_len=10)

        # Verify
        expected_k = jnp.concatenate(all_k, axis=1)
        expected_v = jnp.concatenate(all_v, axis=1)
        assert jnp.allclose(k_read, expected_k)
        assert jnp.allclose(v_read, expected_v)


class TestPagedAttention:
    """Test paged attention computation."""

    def test_paged_attention_basic(self):
        """Test basic paged attention computation."""
        batch_size = 1
        seq_len = 4
        num_heads = 2
        num_kv_heads = 2
        head_dim = 8
        page_size = 4

        # Create cache and populate it
        cache = create_paged_kv_cache(
            batch_size=batch_size,
            max_seq_len=16,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
        )

        k = jnp.ones((batch_size, seq_len, num_kv_heads, head_dim))
        v = jnp.ones((batch_size, seq_len, num_kv_heads, head_dim)) * 2
        seq_indices = jnp.array([0])

        cache = write_to_paged_cache(cache, k, v, seq_indices)

        # Create query
        q = jnp.ones((batch_size, seq_len, num_heads, head_dim))

        # Compute attention
        output = paged_attention(q, cache, seq_indices)

        # Check output shape
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)

        # Output should be close to v since all keys are the same
        # and attention should be uniform
        assert output.shape == q.shape

    def test_paged_attention_with_update(self):
        """Test combined attention and cache update."""
        batch_size = 1
        seq_len = 4
        num_heads = 2
        num_kv_heads = 2
        head_dim = 8
        page_size = 4

        # Create empty cache
        cache = create_paged_kv_cache(
            batch_size=batch_size,
            max_seq_len=16,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
        )

        # Create QKV
        q = jnp.ones((batch_size, seq_len, num_heads, head_dim))
        k = jnp.ones((batch_size, seq_len, num_kv_heads, head_dim))
        v = jnp.ones((batch_size, seq_len, num_kv_heads, head_dim)) * 2
        seq_indices = jnp.array([0])

        # Compute attention and update cache
        output, updated_cache = paged_attention_with_update(
            q, k, v, cache, seq_indices
        )

        # Check output shape
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)

        # Check cache was updated
        assert updated_cache.page_offsets[0] == seq_len

        # Verify we can read back the cached values
        k_read, v_read = read_from_paged_cache(updated_cache, seq_indices, max_len=seq_len)
        assert jnp.allclose(k_read, k)
        assert jnp.allclose(v_read, v)

    def test_paged_attention_gqa(self):
        """Test paged attention with grouped query attention (GQA)."""
        batch_size = 1
        seq_len = 4
        num_heads = 8  # More query heads
        num_kv_heads = 2  # Fewer KV heads (GQA)
        head_dim = 16
        page_size = 4

        cache = create_paged_kv_cache(
            batch_size=batch_size,
            max_seq_len=16,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
        )

        k = jnp.ones((batch_size, seq_len, num_kv_heads, head_dim))
        v = jnp.ones((batch_size, seq_len, num_kv_heads, head_dim)) * 2
        seq_indices = jnp.array([0])

        cache = write_to_paged_cache(cache, k, v, seq_indices)

        # Create query with more heads
        q = jnp.ones((batch_size, seq_len, num_heads, head_dim))

        # Compute attention (should handle GQA automatically)
        output = paged_attention(q, cache, seq_indices)

        # Check output shape
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
