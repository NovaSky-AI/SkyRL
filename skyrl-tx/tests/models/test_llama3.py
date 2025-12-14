import os
import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tx.models.llama3 import Llama3ForCausalLM
from tx.models.configs import Llama3Config
from tx.utils.models import load_safetensors


@pytest.mark.parametrize("tp", [1, 2])
def test_llama3(tp: int):
    """Test LLama3 model against HuggingFace reference implementation."""
    if not jax._src.xla_bridge.backends_are_initialized():  # type: ignore
        jax.config.update("jax_num_cpu_devices", 2)

    if os.getenv("CI"):
        pytest.skip("Test currently runs out of memory in the CI")

    # Use a small LLama model for testing
    model_name = "unsloth/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)

    inputs = ["The capital of France is", "The most popular programming language is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    with torch.no_grad():
        hf_outputs = hf_model(
            batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True, return_dict=True
        )

    # Save the HF model checkpoint so we can load our model from it
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)

        base_config = AutoConfig.from_pretrained(model_name)
        config = Llama3Config(base_config, max_lora_adapters=1, max_lora_rank=1, shard_attention_heads=True)
        mesh = jax.make_mesh((1, tp), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Llama3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        outputs = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy(), output_hidden_states=True)

        assert outputs.hidden_states is not None
        assert np.allclose(hf_outputs.hidden_states[0], outputs.hidden_states[0], rtol=1e-6)
        assert np.allclose(hf_outputs.hidden_states[1], outputs.hidden_states[1], rtol=1e-3, atol=1e-3)
        # Higher tolerance for final layer due to accumulated numerical differences between PyTorch and JAX
        assert np.allclose(hf_outputs.hidden_states[-1], outputs.hidden_states[-1], rtol=5e-2, atol=5e-2)
