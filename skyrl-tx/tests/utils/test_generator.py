from typing import cast
from unittest.mock import MagicMock

from flax import nnx
import jax.numpy as jnp
from tx.models.configs import ModelConfig
from tx.models.types import CausalLMOutput
from tx.tinker.types import SamplingParams
from tx.utils.generator import GenerateOutput, GeneratorMixin, KVCache, apply_top_k_batch, apply_top_p_batch
from tx.utils.logits_processor import LogitsProcessorMixin, LMHead


class DummyModel(GeneratorMixin, LogitsProcessorMixin, nnx.Module):
    """Dummy model for testing generator behavior.

    In this dummy model, hidden_states directly equal logits (identity transformation).
    When adapter_indices is provided, it scales logits by (1 + adapter_index).
    """

    def __init__(self, vocab_size: int = 16, loss_chunk_size: int = 0):
        self.config = MagicMock(loss_chunk_size=loss_chunk_size, gradient_checkpointing=False)
        self.vocab_size = vocab_size

        def lm_head(hidden_states, adapter_indices=None):
            # Scale logits by (1 + adapter_index) so different adapters give different log-softmax results
            if adapter_indices is not None:
                scale = (1 + adapter_indices[:, None, None]).astype(jnp.float32)
                return hidden_states * scale
            return hidden_states

        self.lm_head = lm_head

    def get_lm_head(self) -> LMHead:
        """Return the lm_head callable for logits computation."""
        return self.lm_head

    def get_model_config(self) -> ModelConfig:
        """Return the model configuration required by LogitsProcessorMixin."""
        return cast(ModelConfig, self.config)

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        positions=None,
        kv_cache=None,
        adapter_indices=None,
    ):
        batch_size, seq_len = input_ids.shape
        base = jnp.arange(self.vocab_size, dtype=jnp.float32)

        if kv_cache is None:
            # Prefill: deterministic hidden_states (which equal logits)
            hidden_states = jnp.tile(base[None, None, :], (batch_size, seq_len, 1))
            keys = [jnp.zeros((batch_size, seq_len, 1, 1), dtype=jnp.float32)]
            values = [jnp.zeros((batch_size, seq_len, 1, 1), dtype=jnp.float32)]
            # Per-sequence cache_position (all same length in this test)
            cache_position = (
                attention_mask.sum(axis=1) if attention_mask is not None else jnp.full((batch_size,), seq_len)
            )
            kv_cache = KVCache(keys=keys, values=values, cache_position=cache_position)
        else:
            # Step: hidden_states vary with cache_position (use mean for batched position)
            mean_pos = kv_cache.cache_position.mean()
            hidden_states = jnp.tile(base[None, None, :] + mean_pos, (batch_size, 1, 1))
            kv_cache = KVCache(keys=kv_cache.keys, values=kv_cache.values, cache_position=kv_cache.cache_position + 1)

        return CausalLMOutput(last_hidden_state=hidden_states, kv_cache=kv_cache)


def make_inputs(batch_size: int, prompt_length: int):
    input_ids = jnp.tile(jnp.arange(prompt_length, dtype=jnp.int32)[None, :], (batch_size, 1))
    attention_mask = jnp.ones((batch_size, prompt_length), dtype=jnp.int32)
    return input_ids, attention_mask


def generator_outputs_equal(output1: GenerateOutput, index1: int, output2: GenerateOutput, index2: int) -> bool:
    """Check if two GenerateOutput objects are equal at the given indices."""
    return (
        output1.generated_ids[index1] == output2.generated_ids[index2]
        and jnp.allclose(jnp.array(output1.logprobs[index1]), jnp.array(output2.logprobs[index2]))
        and output1.stop_reasons[index1] == output2.stop_reasons[index2]
    )


def test_deterministic_generation():
    """Repeated generation with same seed should be deterministic."""
    model = DummyModel(vocab_size=8)
    input_ids, attention_mask = make_inputs(batch_size=1, prompt_length=3)
    sampling = SamplingParams(max_tokens=4, temperature=1.0, seed=12345)

    res1 = model.generate(input_ids, attention_mask, sampling_params=[sampling])
    res2 = model.generate(input_ids, attention_mask, sampling_params=[sampling])

    assert generator_outputs_equal(res1, 0, res2, 0)


def test_batch_independence():
    """Batch generation should be equivalent to individual generation with same seeds."""
    model = DummyModel(vocab_size=12)
    input_ids, attention_mask = make_inputs(batch_size=2, prompt_length=4)

    sp1 = SamplingParams(max_tokens=5, temperature=1.0, seed=111)
    sp2 = SamplingParams(max_tokens=5, temperature=1.0, seed=222)

    batch_result = model.generate(input_ids, attention_mask, sampling_params=[sp1, sp2])

    res_a = model.generate(input_ids[:1], attention_mask[:1], sampling_params=[sp1])
    res_b = model.generate(input_ids[1:], attention_mask[1:], sampling_params=[sp2])

    assert generator_outputs_equal(batch_result, 0, res_a, 0)
    assert generator_outputs_equal(batch_result, 1, res_b, 0)


def test_greedy_vs_sampled():
    """Greedy and sampled generation should be independent in batch."""
    model = DummyModel(vocab_size=10)
    input_ids, attention_mask = make_inputs(batch_size=2, prompt_length=2)

    sp_greedy = SamplingParams(max_tokens=3, temperature=0.0, seed=999)
    sp_sample = SamplingParams(max_tokens=3, temperature=1.0, seed=2020)

    batch_result = model.generate(input_ids, attention_mask, sampling_params=[sp_greedy, sp_sample])

    single_greedy = model.generate(input_ids[:1], attention_mask[:1], sampling_params=[sp_greedy])
    single_sample = model.generate(input_ids[1:], attention_mask[1:], sampling_params=[sp_sample])

    assert generator_outputs_equal(batch_result, 0, single_greedy, 0)
    assert generator_outputs_equal(batch_result, 1, single_sample, 0)


def test_prompt_logprobs():
    """Test prompt logprobs computation."""
    model = DummyModel(vocab_size=16)
    prompt_length = 5
    expected_length = prompt_length - 1  # We skip the first token

    # Test with single sequence (batch_size=1)
    input_ids, attention_mask = make_inputs(batch_size=1, prompt_length=prompt_length)
    sampling = SamplingParams(max_tokens=4, temperature=0.0, seed=42)

    # Test with prompt_logprobs=True
    result_with = model.generate(input_ids, attention_mask, sampling_params=[sampling], prompt_logprobs=True)
    assert result_with.prompt_logprobs is not None, "prompt_logprobs should not be None when enabled"
    assert len(result_with.prompt_logprobs) == 1, "Should have prompt_logprobs for 1 sequence in batch"
    assert (
        len(result_with.prompt_logprobs[0]) == expected_length
    ), f"prompt_logprobs should have length {expected_length} (prompt_length - 1)"

    # Test with prompt_logprobs=False
    result_without = model.generate(input_ids, attention_mask, sampling_params=[sampling], prompt_logprobs=False)
    assert result_without.prompt_logprobs is None, "prompt_logprobs should be None when disabled"

    # Test with batched generation
    batch_size = 3
    input_ids_batch, attention_mask_batch = make_inputs(batch_size=batch_size, prompt_length=prompt_length)
    result_batch = model.generate(
        input_ids_batch, attention_mask_batch, sampling_params=[sampling] * batch_size, prompt_logprobs=True
    )

    assert result_batch.prompt_logprobs is not None
    assert len(result_batch.prompt_logprobs) == batch_size, f"Should have prompt_logprobs for {batch_size} sequences"
    for i in range(batch_size):
        assert (
            len(result_batch.prompt_logprobs[i]) == expected_length
        ), f"Sequence {i}: expected prompt_logprobs length {expected_length}"

    # Test that adapter_indices affects prompt_logprobs (verifies adapter_indices is passed to compute_logprobs)
    adapter_0 = jnp.array([0], dtype=jnp.int32)
    adapter_1 = jnp.array([1], dtype=jnp.int32)
    result_adapter_0 = model.generate(
        input_ids, attention_mask, sampling_params=[sampling], adapter_indices=adapter_0, prompt_logprobs=True
    )
    result_adapter_1 = model.generate(
        input_ids, attention_mask, sampling_params=[sampling], adapter_indices=adapter_1, prompt_logprobs=True
    )
    assert not jnp.allclose(
        jnp.array(result_adapter_0.prompt_logprobs[0]), jnp.array(result_adapter_1.prompt_logprobs[0])
    ), "prompt_logprobs should differ when adapter_indices differ"


def test_top_k_filtering():
    """Test apply_top_k_batch function directly."""
    # Create test logits [batch_size, vocab_size]
    logits = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

    # Test k=2: should keep only top 2 values (4.0 and 5.0)
    filtered = apply_top_k_batch(logits, k_values=jnp.array([2]), max_k=2)
    # Values below threshold should be -inf, and top 2 values should be unchanged
    expected = jnp.array([[-jnp.inf, -jnp.inf, -jnp.inf, 4.0, 5.0]])
    assert jnp.array_equal(filtered, expected)

    # Test k<=0 with max_k>0: should not filter that example
    filtered = apply_top_k_batch(logits, k_values=jnp.array([-1]), max_k=5)
    assert jnp.array_equal(filtered, logits)

    # Test per-example k values in batch (second row has ties: two 3.0s)
    logits_batch = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 3.0, 1.0]])
    filtered = apply_top_k_batch(logits_batch, k_values=jnp.array([2, 3]), max_k=3)
    # Second row keeps exactly 3 values despite ties (5.0, 4.0, and first 3.0)
    expected = jnp.array(
        [
            [-jnp.inf, -jnp.inf, -jnp.inf, 4.0, 5.0],
            [5.0, 4.0, 3.0, -jnp.inf, -jnp.inf],
        ]
    )
    assert jnp.array_equal(filtered, expected)


def test_top_p_filtering():
    """Test apply_top_p_batch function directly."""
    # probs = [0.002, 0.006, 0.015, 0.041, 0.112, 0.825]
    logits = jnp.array([[0.0, 1.0, 2.0, 3.0, 4.0, 6.0]])

    # Test p=1.0: should not filter anything
    filtered = apply_top_p_batch(logits, p_values=jnp.array([1.0]))
    assert jnp.array_equal(filtered, logits)

    # Test p=0.0: should only keep the top token (index 5)
    filtered = apply_top_p_batch(logits, p_values=jnp.array([0.0]))
    expected = jnp.array([[-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 6.0]])
    assert jnp.array_equal(filtered, expected)

    # Test p=0.9: cumsum_exclusive < 0.9 keeps top 2 tokens
    filtered = apply_top_p_batch(logits, p_values=jnp.array([0.9]))
    expected = jnp.array([[-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 4.0, 6.0]])
    assert jnp.array_equal(filtered, expected)

    # Test per-example p values in batch
    logits_batch = jnp.array([[0.0, 1.0, 2.0, 3.0, 4.0, 6.0], [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]])
    filtered = apply_top_p_batch(logits_batch, p_values=jnp.array([1.0, 0.0]))
    expected = jnp.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 6.0],
            [-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 6.0],
        ]
    )
    assert jnp.array_equal(filtered, expected)
