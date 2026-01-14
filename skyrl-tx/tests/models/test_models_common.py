import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tx.models.configs import Llama3Config, Qwen3Config
from tx.models.llama3 import Llama3ForCausalLM
from tx.models.qwen3 import Qwen3ForCausalLM
from tx.tinker.types import SamplingParams
from tx.utils.models import load_safetensors


@pytest.mark.parametrize(
    "model_name,config_cls,model_cls,mesh_axes",
    [
        ("unsloth/Llama-3.2-1B", Llama3Config, Llama3ForCausalLM, ("dp", "tp")),
        ("Qwen/Qwen3-0.6B", Qwen3Config, Qwen3ForCausalLM, ("fsdp", "tp")),
    ],
    ids=["llama3", "qwen3"],
)
def test_last_token_logits_only(model_name, config_cls, model_cls, mesh_axes):
    """Test that last_token_logits_only returns correct shape and values."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)

    inputs = ["The capital of France is", "Hello world"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    batch_size, seq_len = batch.input_ids.shape

    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)

        base_config = AutoConfig.from_pretrained(model_name)
        config = config_cls(base_config, max_lora_adapters=1, max_lora_rank=1, shard_attention_heads=True)
        mesh = jax.make_mesh((1, 1), mesh_axes)
        with jax.set_mesh(mesh):
            model = model_cls(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        # Get full logits
        outputs_full = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy())
        assert outputs_full.logits.shape == (batch_size, seq_len, config.vocab_size)

        # Get last token logits only
        outputs_last = model(
            batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy(), last_token_logits_only=True
        )
        assert outputs_last.logits.shape == (
            batch_size,
            1,
            config.vocab_size,
        ), f"Expected shape ({batch_size}, 1, {config.vocab_size}), got {outputs_last.logits.shape}"

        # Last token logits should match
        assert np.allclose(outputs_full.logits[:, -1:, :], outputs_last.logits, rtol=1e-5, atol=1e-5)

        # Test generation equivalence with and without prompt_logprobs
        input_ids = jnp.array(batch.input_ids.numpy())
        attention_mask = jnp.array(batch.attention_mask.numpy())
        sampling_params = [SamplingParams(max_tokens=8, temperature=0.0, seed=42)] * batch_size

        result_with = model.generate(input_ids, attention_mask, sampling_params=sampling_params, prompt_logprobs=True)
        result_without = model.generate(
            input_ids, attention_mask, sampling_params=sampling_params, prompt_logprobs=False
        )

        for i in range(batch_size):
            assert (
                result_with.generated_ids[i] == result_without.generated_ids[i]
            ), f"Generated tokens should match for seq {i}"
            assert (
                result_with.stop_reasons[i] == result_without.stop_reasons[i]
            ), f"Stop reasons should match for seq {i}"
            assert np.allclose(
                result_with.logprobs[i], result_without.logprobs[i]
            ), f"Logprobs should match for seq {i}"
