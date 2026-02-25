import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import PretrainedConfig
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig as HFQwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM as HFQwen3NextForCausalLM

from tx.models.configs import Qwen3NextConfig
from tx.models.qwen3_next import Qwen3NextForCausalLM
from tx.tinker.types import SamplingParams
from tx.utils.models import load_safetensors


def make_small_hf_config() -> HFQwen3NextConfig:
    return HFQwen3NextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        num_experts=0,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
    )


def make_small_tx_config(base_config: HFQwen3NextConfig, *, shard_attention_heads: bool = False) -> Qwen3NextConfig:
    return Qwen3NextConfig(
        base_config,
        max_lora_adapters=2,
        max_lora_rank=8,
        shard_attention_heads=shard_attention_heads,
    )


@pytest.mark.parametrize("tp", [1, 2])
def test_qwen3_next_end_to_end(tp: int):
    if jax.device_count() < tp:
        pytest.skip(f"Need at least {tp} JAX devices for tp={tp}, found {jax.device_count()}")

    hf_config = make_small_hf_config()
    hf_model = HFQwen3NextForCausalLM(hf_config)
    hf_model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)

        config = make_small_tx_config(hf_config, shard_attention_heads=tp > 1)
        mesh = jax.make_mesh((1, 1, tp), ("fsdp", "ep", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 3)
        with jax.set_mesh(mesh):
            model = Qwen3NextForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
            load_safetensors(tmp, config, model)
            outputs = model(
                input_ids.numpy(),
                attention_mask=attention_mask.numpy(),
                output_hidden_states=True,
            )
            logits = model.compute_logits(outputs.last_hidden_state)

    assert outputs.hidden_states is not None
    assert np.allclose(hf_outputs.hidden_states[0], outputs.hidden_states[0], rtol=1e-6, atol=1e-6)
    assert np.allclose(hf_outputs.hidden_states[1], outputs.hidden_states[1], rtol=1e-3, atol=1e-3)
    assert np.allclose(hf_outputs.hidden_states[-1], outputs.hidden_states[-1], rtol=8e-2, atol=8e-2)
    assert np.allclose(hf_outputs.logits, logits, rtol=1e-1, atol=1e-1)


def test_qwen3_next_prefill_cache_shapes():
    config = make_small_tx_config(make_small_hf_config())
    mesh = jax.make_mesh((1, 1, 1), ("fsdp", "ep", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 3)

    with jax.set_mesh(mesh):
        model = Qwen3NextForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        input_ids = jnp.array([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]], dtype=jnp.int32)
        attention_mask = jnp.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=jnp.int32)
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

    assert outputs.last_hidden_state.shape == (2, 5, config.hidden_size)
    assert outputs.hidden_states is not None
    assert len(outputs.hidden_states) == config.num_hidden_layers + 1
    assert outputs.kv_cache is not None
    assert len(outputs.kv_cache.keys) == config.num_hidden_layers
    assert outputs.kv_cache.conv_states is not None
    assert outputs.kv_cache.recurrent_states is not None
    assert outputs.kv_cache.keys[0].shape[1] == 0
    assert outputs.kv_cache.keys[1].shape[1] == 5
    assert outputs.kv_cache.keys[3].shape[1] == 5
    assert outputs.kv_cache.conv_states[0].shape[-1] == config.linear_conv_kernel_dim
    assert outputs.kv_cache.recurrent_states[0].shape[1] == config.linear_num_value_heads


def test_qwen3_next_decode_updates_cache_position():
    config = make_small_tx_config(make_small_hf_config())
    mesh = jax.make_mesh((1, 1, 1), ("fsdp", "ep", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 3)

    with jax.set_mesh(mesh):
        model = Qwen3NextForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        input_ids = jnp.array([[1, 2, 3], [4, 5, 0]], dtype=jnp.int32)
        attention_mask = jnp.array([[1, 1, 1], [1, 1, 0]], dtype=jnp.int32)
        prefill = model(input_ids, attention_mask=attention_mask)
        assert prefill.kv_cache is not None

        cache = prefill.kv_cache.pad_to_length(8)
        decode_attention_mask = jnp.pad(attention_mask, ((0, 0), (0, 5)))
        batch_idx = jnp.arange(decode_attention_mask.shape[0])
        decode_attention_mask = decode_attention_mask.at[batch_idx, cache.cache_position].set(1)

        next_token = jnp.array([[9], [10]], dtype=jnp.int32)
        positions = cache.cache_position[:, None]
        decode_out = model(
            next_token,
            attention_mask=decode_attention_mask,
            positions=positions,
            kv_cache=cache,
        )

    assert decode_out.kv_cache is not None
    assert jnp.all(decode_out.kv_cache.cache_position == cache.cache_position + 1)
    assert decode_out.kv_cache.keys[1].shape[1] == 8
    assert decode_out.kv_cache.conv_states is not None
    assert decode_out.kv_cache.recurrent_states is not None


def test_qwen3_next_generate():
    config = make_small_tx_config(make_small_hf_config())
    mesh = jax.make_mesh((1, 1, 1), ("fsdp", "ep", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 3)

    with jax.set_mesh(mesh):
        model = Qwen3NextForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        attention_mask = jnp.array([[1, 1, 1]], dtype=jnp.int32)
        out = model.generate(
            input_ids,
            attention_mask,
            sampling_params=[SamplingParams(max_tokens=2, temperature=0.0, seed=0)],
        )

    assert len(out.generated_ids) == 1
    assert len(out.logprobs) == 1
    assert len(out.generated_ids[0]) == 2


def test_qwen3_next_nested_rope_parameters_without_top_level_rope_theta():
    text_config = {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
        "linear_conv_kernel_dim": 3,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
        "num_experts": 0,
        "num_experts_per_tok": 1,
        "decoder_sparse_step": 1,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
        },
    }
    base_config = PretrainedConfig(
        architectures=["Qwen3NextForCausalLM"],
        model_type="qwen3_5_moe",
        text_config=text_config,
    )
    config = Qwen3NextConfig(
        base_config,
        max_lora_adapters=2,
        max_lora_rank=8,
        shard_attention_heads=False,
    )

    mesh = jax.make_mesh((1, 1, 1), ("fsdp", "ep", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 3)
    with jax.set_mesh(mesh):
        model = Qwen3NextForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        attention_mask = jnp.array([[1, 1, 1]], dtype=jnp.int32)
        outputs = model(input_ids, attention_mask=attention_mask)

    assert outputs.last_hidden_state.shape == (1, 3, config.hidden_size)
