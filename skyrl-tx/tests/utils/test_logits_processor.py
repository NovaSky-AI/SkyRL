"""Unit tests for LogitsProcessorMixin chunked logprobs computation."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.utils.test_generator import DummyModel


def assert_chunked_matches_nonchunked(
    hidden_states: jnp.ndarray,
    target_ids: jnp.ndarray,
    chunk_size: int,
    adapter_indices: jnp.ndarray | None = None,
    vocab_size: int = 16,
):
    """Assert chunked and non-chunked paths produce identical results."""
    model_chunked = DummyModel(vocab_size=vocab_size, loss_chunk_size=chunk_size)
    model_nonchunked = DummyModel(vocab_size=vocab_size, loss_chunk_size=0)

    logprobs_chunked = model_chunked.compute_logprobs(hidden_states, target_ids, adapter_indices)
    logprobs_nonchunked = model_nonchunked.compute_logprobs(hidden_states, target_ids, adapter_indices)

    B, T = target_ids.shape
    assert logprobs_chunked.shape == (B, T)
    assert logprobs_nonchunked.shape == (B, T)

    np.testing.assert_allclose(
        np.asarray(logprobs_chunked),
        np.asarray(logprobs_nonchunked),
        rtol=1e-5,
        atol=1e-5,
    )


class TestChunkedLogprobs:
    """Tests for chunked vs non-chunked logprobs computation."""

    @pytest.mark.parametrize(
        "B,T,chunk_size",
        [
            (2, 4, 3),  # chunk doesn't divide evenly, needs padding
            (2, 4, 8),  # chunk equals B*T exactly
            (2, 4, 16),  # chunk larger than B*T
            (1, 8, 3),  # single batch element
            (4, 1, 2),  # single token per sequence
            (1, 1, 1),  # minimal case
        ],
    )
    def test_chunk_boundary_cases(self, B, T, chunk_size):
        """Test various chunk size vs total token relationships."""
        V = 16  # vocab_size = hidden_size for identity lm_head
        hidden_states = jnp.arange(B * T * V, dtype=jnp.float32).reshape(B, T, V) / (B * T * V)
        target_ids = jnp.arange(B * T, dtype=jnp.int32).reshape(B, T) % V

        assert_chunked_matches_nonchunked(hidden_states, target_ids, chunk_size, vocab_size=V)

    @pytest.mark.parametrize(
        "B,T,chunk_size,adapter_indices",
        [
            (2, 4, 3, None),  # no adapters
            (2, 4, 3, "arange"),  # different adapter per batch, chunk spans boundary
            (3, 4, 5, "arange"),  # chunk spans multiple batches
            (4, 2, 3, "zeros"),  # all same adapter
        ],
    )
    def test_adapter_indices_handling(self, B, T, chunk_size, adapter_indices):
        """Test adapter indices are correctly mapped across chunk boundaries."""
        V = 16
        hidden_states = jnp.arange(B * T * V, dtype=jnp.float32).reshape(B, T, V) / (B * T * V)
        target_ids = jnp.arange(B * T, dtype=jnp.int32).reshape(B, T) % V

        if adapter_indices == "arange":
            adapter_indices = jnp.arange(B, dtype=jnp.int32)
        elif adapter_indices == "zeros":
            adapter_indices = jnp.zeros(B, dtype=jnp.int32)

        assert_chunked_matches_nonchunked(hidden_states, target_ids, chunk_size, adapter_indices, vocab_size=V)

    def test_gradient_checkpointing_flag(self):
        """Gradient checkpointing should not affect forward pass results."""
        B, T, V, chunk_size = 2, 4, 16, 3
        hidden_states = jnp.arange(B * T * V, dtype=jnp.float32).reshape(B, T, V) / (B * T * V)
        target_ids = jnp.arange(B * T, dtype=jnp.int32).reshape(B, T) % V

        model_no_ckpt = DummyModel(vocab_size=V, loss_chunk_size=chunk_size)
        model_no_ckpt.config.gradient_checkpointing = False

        model_ckpt = DummyModel(vocab_size=V, loss_chunk_size=chunk_size)
        model_ckpt.config.gradient_checkpointing = True

        logprobs_no_ckpt = model_no_ckpt.compute_logprobs(hidden_states, target_ids)
        logprobs_ckpt = model_ckpt.compute_logprobs(hidden_states, target_ids)

        np.testing.assert_allclose(
            np.asarray(logprobs_no_ckpt),
            np.asarray(logprobs_ckpt),
            rtol=1e-5,
            atol=1e-5,
        )
