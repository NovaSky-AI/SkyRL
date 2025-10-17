import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tx.models import Qwen3ForCausalLM
from tx.utils.models import load_safetensors


def test_qwen3_generate():
    """Test batched text generation with KV caching matches HuggingFace."""
    model_name = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)

    inputs = ["The capital of France is ", "The future of AI is "]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    # Generate with HuggingFace (reference)
    with torch.no_grad():
        hf_output = hf_model.generate(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            max_new_tokens=10,
            do_sample=False,
        )

    # Generate with our implementation
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)
        config = AutoConfig.from_pretrained(model_name)

        mesh = jax.make_mesh((1, 1), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        output = model.generate(
            batch.input_ids.numpy(), batch.attention_mask.numpy(), max_new_tokens=10, temperature=0.01, seed=42
        )

        assert jnp.array_equal(output, hf_output.numpy()), "Generated tokens don't match HuggingFace"
