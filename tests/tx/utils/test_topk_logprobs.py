"""Unit tests for LogitsProcessorMixin top-K logprobs (soft distillation)."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.tx.utils.test_generator import DummyModel


def _np_log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


class TestTopkLogprobs:
    """Tests for compute_topk_logprobs / logits_to_topk_logprobs."""

    @pytest.mark.parametrize("B,T,K", [(2, 4, 3), (1, 8, 5), (3, 2, 1), (1, 1, 1)])
    def test_matches_numpy_reference(self, B, T, K):
        """Top-K gathered logprobs match a numpy log-softmax + gather reference."""
        V = 16  # identity lm_head => logits == hidden_states
        rng = np.random.default_rng(0)
        hidden = rng.standard_normal((B, T, V)).astype(np.float32)
        target_topk = rng.integers(0, V, (B, T, K)).astype(np.int32)

        model = DummyModel(vocab_size=V, loss_chunk_size=0)
        got = np.asarray(model.compute_topk_logprobs(jnp.array(hidden), jnp.array(target_topk)))

        ref = np.take_along_axis(_np_log_softmax(hidden), target_topk, axis=-1)
        assert got.shape == (B, T, K)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)

    def test_k1_equals_compute_logprobs(self):
        """K=1 top-K logprobs equal the standard 1D compute_logprobs."""
        B, T, V = 2, 5, 16
        rng = np.random.default_rng(1)
        hidden = jnp.array(rng.standard_normal((B, T, V)).astype(np.float32))
        target_ids = jnp.array(rng.integers(0, V, (B, T)).astype(np.int32))

        model = DummyModel(vocab_size=V, loss_chunk_size=0)
        topk = model.compute_topk_logprobs(hidden, target_ids[..., None])  # [B, T, 1]
        flat = model.compute_logprobs(hidden, target_ids)  # [B, T]

        np.testing.assert_allclose(np.asarray(topk.squeeze(-1)), np.asarray(flat), rtol=1e-6, atol=1e-6)

    def test_weighted_sum_is_forward_cross_entropy(self):
        """sum_k w_k * logp_k equals the (negative) forward cross-entropy target."""
        B, T, V, K = 2, 4, 16, 4
        rng = np.random.default_rng(2)
        hidden = rng.standard_normal((B, T, V)).astype(np.float32)
        target_topk = rng.integers(0, V, (B, T, K)).astype(np.int32)
        weights = rng.random((B, T, K)).astype(np.float32)
        weights /= weights.sum(axis=-1, keepdims=True)
        weights[0, 0] = 0.0  # masked (non-completion) position

        model = DummyModel(vocab_size=V, loss_chunk_size=0)
        topk_lp = np.asarray(model.compute_topk_logprobs(jnp.array(hidden), jnp.array(target_topk)))
        got = (weights * topk_lp).sum(axis=-1)  # [B, T]

        ref_lp = np.take_along_axis(_np_log_softmax(hidden), target_topk, axis=-1)
        ref = (weights * ref_lp).sum(axis=-1)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)
        assert np.isclose(got[0, 0], 0.0)  # masked position contributes 0
