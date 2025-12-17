"""Paged attention implementation for efficient KV cache management.

This module implements paged attention, which stores KV cache in non-contiguous
memory blocks (pages) to reduce memory fragmentation and enable efficient memory
sharing across sequences.

Key benefits:
- Reduced memory fragmentation
- Better memory utilization for variable-length sequences
- Efficient memory sharing for beam search and batched generation
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PagedKVCache:
    """Paged KV cache structure.
    
    Attributes:
        k_pages: Key cache pages [num_pages, page_size, num_kv_heads, head_dim]
        v_pages: Value cache pages [num_pages, page_size, num_kv_heads, head_dim]
        page_table: Maps logical positions to physical pages [batch_size, max_num_pages]
        page_offsets: Current offset within each page [batch_size]
        page_size: Number of tokens per page
        num_pages: Total number of allocated pages
    """
    k_pages: jax.Array
    v_pages: jax.Array
    page_table: jax.Array
    page_offsets: jax.Array
    page_size: int
    num_pages: int


def create_paged_kv_cache(
    batch_size: int,
    max_seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int = 16,
    dtype: jnp.dtype = jnp.float32,
) -> PagedKVCache:
    """Create a paged KV cache.
    
    Args:
        batch_size: Number of sequences in the batch
        max_seq_len: Maximum sequence length
        num_kv_heads: Number of key-value heads
        head_dim: Dimension of each head
        page_size: Number of tokens per page (default: 16)
        dtype: Data type for cache
        
    Returns:
        Initialized PagedKVCache
    """
    # Calculate number of pages needed
    max_num_pages = (max_seq_len + page_size - 1) // page_size
    num_pages = batch_size * max_num_pages
    
    # Initialize page storage
    k_pages = jnp.zeros((num_pages, page_size, num_kv_heads, head_dim), dtype=dtype)
    v_pages = jnp.zeros((num_pages, page_size, num_kv_heads, head_dim), dtype=dtype)
    
    # Initialize page table: each sequence gets consecutive pages
    page_table = jnp.arange(num_pages).reshape(batch_size, max_num_pages)
    page_offsets = jnp.zeros(batch_size, dtype=jnp.int32)
    
    return PagedKVCache(
        k_pages=k_pages,
        v_pages=v_pages,
        page_table=page_table,
        page_offsets=page_offsets,
        page_size=page_size,
        num_pages=num_pages,
    )


def write_to_paged_cache(
    cache: PagedKVCache,
    k_new: jax.Array,
    v_new: jax.Array,
    seq_indices: jax.Array,
) -> PagedKVCache:
    """Write new key-value pairs to paged cache.
    
    Args:
        cache: Current paged KV cache
        k_new: New keys [batch_size, seq_len, num_kv_heads, head_dim]
        v_new: New values [batch_size, seq_len, num_kv_heads, head_dim]
        seq_indices: Sequence indices to update [batch_size]
        
    Returns:
        Updated PagedKVCache
    """
    batch_size, seq_len, num_kv_heads, head_dim = k_new.shape
    
    def write_sequence(carry, inputs):
        k_pages, v_pages = carry
        seq_idx, k_seq, v_seq, page_offset = inputs
        
        # Get page table for this sequence
        seq_page_table = cache.page_table[seq_idx]
        
        def write_token(token_carry, token_inputs):
            k_pages_inner, v_pages_inner, offset = token_carry
            k_token, v_token = token_inputs
            
            # Calculate page and position within page
            page_idx_in_table = offset // cache.page_size
            pos_in_page = offset % cache.page_size
            physical_page = seq_page_table[page_idx_in_table]
            
            # Update pages
            k_pages_inner = k_pages_inner.at[physical_page, pos_in_page].set(k_token)
            v_pages_inner = v_pages_inner.at[physical_page, pos_in_page].set(v_token)
            
            return (k_pages_inner, v_pages_inner, offset + 1), None
        
        (k_pages, v_pages, _), _ = jax.lax.scan(
            write_token,
            (k_pages, v_pages, page_offset),
            (k_seq, v_seq)
        )
        
        return (k_pages, v_pages), None
    
    page_offsets_for_seqs = cache.page_offsets[seq_indices]
    (k_pages_updated, v_pages_updated), _ = jax.lax.scan(
        write_sequence,
        (cache.k_pages, cache.v_pages),
        (seq_indices, k_new, v_new, page_offsets_for_seqs)
    )
    
    # Update page offsets
    new_offsets = cache.page_offsets.at[seq_indices].add(seq_len)
    
    return PagedKVCache(
        k_pages=k_pages_updated,
        v_pages=v_pages_updated,
        page_table=cache.page_table,
        page_offsets=new_offsets,
        page_size=cache.page_size,
        num_pages=cache.num_pages,
    )


def read_from_paged_cache(
    cache: PagedKVCache,
    seq_indices: jax.Array,
    max_len: Optional[int] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Read key-value pairs from paged cache.
    
    Args:
        cache: Paged KV cache
        seq_indices: Sequence indices to read [batch_size]
        max_len: Maximum length to read (if None, read all cached tokens)
        
    Returns:
        Tuple of (keys, values) with shape [batch_size, seq_len, num_kv_heads, head_dim]
    """
    batch_size = seq_indices.shape[0]
    num_kv_heads = cache.k_pages.shape[2]
    head_dim = cache.k_pages.shape[3]
    
    # Determine sequence lengths
    seq_lens = cache.page_offsets[seq_indices]
    if max_len is not None:
        seq_lens = jnp.minimum(seq_lens, max_len)
    max_seq_len = jnp.max(seq_lens)
    
    def read_sequence(seq_idx, seq_len):
        seq_page_table = cache.page_table[seq_idx]
        
        def read_token(pos):
            page_idx_in_table = pos // cache.page_size
            pos_in_page = pos % cache.page_size
            physical_page = seq_page_table[page_idx_in_table]
            
            k_token = cache.k_pages[physical_page, pos_in_page]
            v_token = cache.v_pages[physical_page, pos_in_page]
            
            # Mask out positions beyond sequence length
            mask = pos < seq_len
            k_token = jnp.where(mask, k_token, 0.0)
            v_token = jnp.where(mask, v_token, 0.0)
            
            return k_token, v_token
        
        positions = jnp.arange(max_seq_len)
        k_seq, v_seq = jax.vmap(read_token)(positions)
        return k_seq, v_seq
    
    k_batch, v_batch = jax.vmap(read_sequence)(seq_indices, seq_lens)
    
    return k_batch, v_batch


def paged_attention(
    q: jax.Array,
    cache: PagedKVCache,
    seq_indices: jax.Array,
    attention_mask: Optional[jax.Array] = None,
    scale: Optional[float] = None,
) -> jax.Array:
    """Compute attention using paged KV cache.
    
    Args:
        q: Query tensor [batch_size, seq_len, num_heads, head_dim]
        cache: Paged KV cache
        seq_indices: Sequence indices [batch_size]
        attention_mask: Optional attention mask [batch_size, seq_len, kv_len]
        scale: Attention scale factor (default: 1/sqrt(head_dim))
        
    Returns:
        Attention output [batch_size, seq_len, num_heads, head_dim]
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Read K, V from paged cache
    k, v = read_from_paged_cache(cache, seq_indices)
    
    # Handle GQA: repeat KV heads if needed
    num_kv_heads = k.shape[2]
    if num_heads != num_kv_heads:
        num_repeats = num_heads // num_kv_heads
        k = jnp.repeat(k, num_repeats, axis=2)
        v = jnp.repeat(v, num_repeats, axis=2)
    
    # Compute attention
    if scale is None:
        scale = 1.0 / jnp.sqrt(head_dim)
    
    # Compute attention scores: [batch_size, num_heads, seq_len, kv_len]
    scores = jnp.einsum('bthd,bThd->bhtT', q, k) * scale
    
    # Apply mask if provided
    if attention_mask is not None:
        # Expand mask to match attention scores shape
        mask_expanded = attention_mask[:, None, :, :]  # [B, 1, T, T]
        scores = jnp.where(mask_expanded, scores, -1e9)
    
    # Softmax and weighted sum
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum('bhtT,bThd->bthd', attn_weights, v)
    
    return output


def paged_attention_with_update(
    q: jax.Array,
    k_new: jax.Array,
    v_new: jax.Array,
    cache: PagedKVCache,
    seq_indices: jax.Array,
    attention_mask: Optional[jax.Array] = None,
    scale: Optional[float] = None,
) -> Tuple[jax.Array, PagedKVCache]:
    """Compute attention and update paged cache in one pass.
    
    This is the main function to use during generation, combining cache update
    and attention computation.
    
    Args:
        q: Query tensor [batch_size, seq_len, num_heads, head_dim]
        k_new: New keys to add [batch_size, seq_len, num_kv_heads, head_dim]
        v_new: New values to add [batch_size, seq_len, num_kv_heads, head_dim]
        cache: Current paged KV cache
        seq_indices: Sequence indices [batch_size]
        attention_mask: Optional attention mask
        scale: Attention scale factor
        
    Returns:
        Tuple of (attention_output, updated_cache)
    """
    # Update cache with new KV pairs
    updated_cache = write_to_paged_cache(cache, k_new, v_new, seq_indices)
    
    # Compute attention using updated cache
    output = paged_attention(q, updated_cache, seq_indices, attention_mask, scale)
    
    return output, updated_cache
