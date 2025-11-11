from flax import nnx
import jax.numpy as jnp
from tx.models.outputs import CausalLMOutput
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
