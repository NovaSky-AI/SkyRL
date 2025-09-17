import os
from pathlib import Path
import tempfile

import jax.numpy as jnp
from flax import nnx
import numpy as np
import safetensors.numpy
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from xtrain.models import Qwen3ForCausalLM


def load_checkpoint(filename: str | os.PathLike, model: nnx.Module) -> None:

    def get_key(path: tuple) -> str:
        if path[-1] in {"embedding", "kernel"}:
            path = (*path[:-1], "weight")
        return ".".join(map(str, path))
    
    tensors = safetensors.numpy.load_file(filename)
    model_params = nnx.to_flat_state(nnx.state(model))
    updates = []
    for path, param in model_params:
        key = get_key(path)
        if path[-2] == "lm_head" and key not in tensors:
            # This model uses tied embeddings
            key = "model.embed_tokens.weight"
        tensors[key] = tensors[key].T
        if path[-2] in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            tensors[key] = tensors[key].reshape(param.shape)
        assert param.shape == tensors[key].shape, f"shape mismatch for {key}"
        updates.append((path, tensors[key]))
    nnx.update(model, nnx.from_flat_state(updates))


def test_qwen3():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", attn_implementation="eager", use_safetensors=True)

    input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        hf_outputs = hf_model(input_ids, output_hidden_states=True, output_attentions=True, return_dict=True)

    # Save the HF model checkpoint so we can load our model from it
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)

        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        model = Qwen3ForCausalLM(config, rngs=nnx.Rngs(0))
        load_checkpoint(Path(tmp) / "model.safetensors", model)
        
        outputs = model(input_ids.numpy(), output_hidden_states=True, output_attentions=True)
        assert np.allclose(hf_outputs.hidden_states[0], outputs["hidden_states"][0], rtol=1e-6)
        assert np.allclose(hf_outputs.attentions[0], outputs["attentions"][0], rtol=1e-4)
        assert np.allclose(hf_outputs.hidden_states[1], outputs["hidden_states"][1], rtol=1e-3, atol=1e-3)
        assert np.allclose(hf_outputs.hidden_states[-1], outputs["hidden_states"][-1], rtol=1e-3, atol=1e-3)
