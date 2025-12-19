from flax import nnx
import jax.numpy as jnp
from tx.models.types import CausalLMOutput
from tx.tinker.types import SamplingParams
from tx.utils.generator import GenerateOutput, GeneratorMixin, KVCache


class DummyModel(GeneratorMixin, nnx.Module):
    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size

    def __call__(self, input_ids, attention_mask=None, positions=None, kv_cache=None, adapter_indices=None):
        """Simple dummy model for testing generator behavior."""
        batch_size, seq_len = input_ids.shape
        base = jnp.arange(self.vocab_size, dtype=jnp.float32)

        if kv_cache is None:
            # Prefill: deterministic logits
            logits = jnp.tile(base[None, None, :], (batch_size, seq_len, 1))
            keys = [jnp.zeros((batch_size, seq_len, 1, 1), dtype=jnp.float32)]
            values = [jnp.zeros((batch_size, seq_len, 1, 1), dtype=jnp.float32)]
            kv_cache = KVCache(keys=keys, values=values, cache_position=seq_len)
        else:
            # Step: logits vary with cache_position
            logits = jnp.tile(base[None, None, :] + kv_cache.cache_position, (batch_size, 1, 1))
            kv_cache = KVCache(keys=kv_cache.keys, values=kv_cache.values, cache_position=kv_cache.cache_position + 1)

        return CausalLMOutput(logits=logits, last_hidden_state=logits, kv_cache=kv_cache)


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
