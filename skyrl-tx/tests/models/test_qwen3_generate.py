import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tx.models import Qwen3ForCausalLM
from tx.utils.models import load_safetensors


def test_qwen3_generate():
    """Test basic text generation with GeneratorMixin."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", attn_implementation="eager", use_safetensors=True
    )

    # Prepare input
    inputs = ["The capital of France is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    input_ids = batch.input_ids.numpy()

    # Save and load model
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)

        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        mesh = jax.make_mesh((1, 1), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        # Test generate method
        max_new_tokens = 10
        generated = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=1.0, seed=42)

        # Decode and print for visual verification
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\nInput: {inputs[0]}")
        print(f"Generated: {decoded}")
