from types import SimpleNamespace

import jax.numpy as jnp

from tx.utils.generator import GeneratorMixin, KVCache


class DummyModel(GeneratorMixin):
    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size

    def __call__(self, input_ids, attention_mask=None, positions=None, kv_cache=None, adapter_indices=None):
        """Simple dummy model for testing generator behavior."""
        batch_size, seq_len = input_ids.shape

        if kv_cache is None:
            # Prefill: deterministic logits
            base = jnp.arange(self.vocab_size, dtype=jnp.float32)
            logits = jnp.tile(base[None, None, :], (batch_size, seq_len, 1)).astype(jnp.float32)
            keys = [jnp.zeros((batch_size, seq_len, 1, 1), dtype=jnp.float32)]
            values = [jnp.zeros((batch_size, seq_len, 1, 1), dtype=jnp.float32)]
            cache_position = seq_len
            return SimpleNamespace(
                logits=logits, kv_cache=KVCache(keys=keys, values=values, cache_position=cache_position)
            )
        else:
            # Step: logits vary with cache_position
            pos = kv_cache.cache_position
            base = jnp.arange(self.vocab_size, dtype=jnp.float32) + pos.astype(jnp.float32)
            logits = jnp.tile(base[None, None, :], (batch_size, 1, 1)).astype(jnp.float32)
            return SimpleNamespace(
                logits=logits, kv_cache=KVCache(keys=kv_cache.keys, values=kv_cache.values, cache_position=pos + 1)
            )


def make_attention_mask(batch_size: int, prompt_length: int):
    return jnp.ones((batch_size, prompt_length), dtype=jnp.int32)


def make_input_ids(batch_size: int, prompt_length: int):
    ids = jnp.arange(prompt_length, dtype=jnp.int32)
    ids = jnp.tile(ids[None, :], (batch_size, 1))
    return ids


def make_sampling_param(max_tokens: int, temperature: float, seed: int, stop=None):
    return SimpleNamespace(max_tokens=max_tokens, temperature=temperature, seed=seed, stop=stop)


def extract_generated_lists(result):
    return [list(map(int, seq)) for seq in result.generated_ids]


def test_deterministic_generation():
    """Repeated generation with same seed should be deterministic."""
    model = DummyModel(vocab_size=8)

    prompt_len = 3
    input_ids = make_input_ids(batch_size=1, prompt_length=prompt_len)
    attention_mask = make_attention_mask(batch_size=1, prompt_length=prompt_len)

    sampling = make_sampling_param(max_tokens=4, temperature=1.0, seed=12345)

    res1 = model.generate(input_ids, attention_mask, sampling_params=[sampling])
    res2 = model.generate(input_ids, attention_mask, sampling_params=[sampling])

    assert extract_generated_lists(res1) == extract_generated_lists(res2)
    assert res1.logprobs == res2.logprobs
    assert res1.stop_reasons == res2.stop_reasons


def test_batch_independence():
    """Batch generation should be equivalent to individual generation with same seeds."""
    model = DummyModel(vocab_size=12)

    prompt_len = 4
    batch_size = 2
    input_ids = make_input_ids(batch_size=batch_size, prompt_length=prompt_len)
    attention_mask = make_attention_mask(batch_size=batch_size, prompt_length=prompt_len)

    sp1 = make_sampling_param(max_tokens=5, temperature=1.0, seed=111)
    sp2 = make_sampling_param(max_tokens=5, temperature=1.0, seed=222)

    batch_result = model.generate(input_ids, attention_mask, sampling_params=[sp1, sp2])

    res_a = model.generate(input_ids[:1], attention_mask[:1], sampling_params=[sp1])
    res_b = model.generate(input_ids[1:], attention_mask[1:], sampling_params=[sp2])

    assert extract_generated_lists(batch_result)[0] == extract_generated_lists(res_a)[0]
    assert extract_generated_lists(batch_result)[1] == extract_generated_lists(res_b)[0]


def test_greedy_vs_sampled():
    """Greedy and sampled generation should be independent in batch."""
    model = DummyModel(vocab_size=10)

    prompt_len = 2
    input_ids = make_input_ids(batch_size=2, prompt_length=prompt_len)
    attention_mask = make_attention_mask(batch_size=2, prompt_length=prompt_len)

    sp_greedy = make_sampling_param(max_tokens=3, temperature=0.0, seed=999)
    sp_sample = make_sampling_param(max_tokens=3, temperature=1.0, seed=2020)

    batch_result = model.generate(input_ids, attention_mask, sampling_params=[sp_greedy, sp_sample])

    single_greedy = model.generate(input_ids[:1], attention_mask[:1], sampling_params=[sp_greedy])
    single_sample = model.generate(input_ids[1:], attention_mask[1:], sampling_params=[sp_sample])

    assert extract_generated_lists(batch_result)[0] == extract_generated_lists(single_greedy)[0]
    assert extract_generated_lists(batch_result)[1] == extract_generated_lists(single_sample)[0]
