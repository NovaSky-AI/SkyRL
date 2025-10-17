import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tx.models import Qwen3ForCausalLM
from tx.utils.models import load_safetensors


def test_qwen3_generate():
    """Test batched text generation with KV caching matches HuggingFace."""
    import torch

    model_name = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)

    # Prepare batched input (left-padded for generation)
    inputs = ["The capital of France is ", "The future of AI is "]
    tokenizer.padding_side = "left"
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    input_ids = batch.input_ids.numpy()
    attention_mask = batch.attention_mask.numpy()

    # Generate with HuggingFace (reference)
    with torch.no_grad():
        hf_output = hf_model.generate(
            torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            max_new_tokens=10,
            do_sample=False
        )
    hf_decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in hf_output]

    # Load our implementation
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)
        config = AutoConfig.from_pretrained(model_name)

        mesh = jax.make_mesh((1, 1), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        # Generate with our implementation
        generated = model.generate(input_ids, attention_mask, max_new_tokens=10, temperature=0.01, seed=42)
        our_decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]

        # Verify match
        print("\nComparison:")
        for i in range(len(inputs)):
            print(f"  {i+1}. {our_decoded[i]}")
            assert hf_decoded[i] == our_decoded[i], f"Mismatch: HF={hf_decoded[i]}, Ours={our_decoded[i]}"

        print("✓ All outputs match HuggingFace")
