import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tx.models import Qwen3ForCausalLM
from tx.utils.models import load_safetensors


def test_qwen3_generate():
    """Test batched text generation with KV caching matches HuggingFace."""
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)

    inputs = ["My name is", "The capital of France is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    # Generate with HuggingFace (reference)
    with torch.no_grad():
        hf_output = hf_model.generate(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            max_new_tokens=10,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Generate with our implementation
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)
        config = AutoConfig.from_pretrained(model_name)

        mesh = jax.make_mesh((1, 1), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        output, our_scores, _, _ = model.generate(
            batch.input_ids.numpy(),
            batch.attention_mask.numpy(),
            max_new_tokens=10,
            temperature=0.0,
            seed=42,
            return_scores=True,
            prompt_logprobs=False
        )

        assert jnp.array_equal(output, hf_output.sequences.numpy()), "Generated tokens don't match HuggingFace"

        # Compare scores (logits) for each generated token
        for step_idx, (hf_score, our_score) in enumerate(zip(hf_output.scores, our_scores)):
            assert np.allclose(
                hf_score.numpy(), our_score, rtol=1e-3, atol=1e-3
            ), f"Step {step_idx}: Logits don't match HuggingFace. Max diff: {np.abs(hf_score.numpy() - our_score).max()}"
