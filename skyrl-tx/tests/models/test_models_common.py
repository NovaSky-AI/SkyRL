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
from tx.utils.models import load_safetensors

MODEL_PARAMS = [
    ("unsloth/Llama-3.2-1B", Llama3Config, Llama3ForCausalLM, ("fsdp", "tp")),
    ("Qwen/Qwen3-0.6B", Qwen3Config, Qwen3ForCausalLM, ("fsdp", "tp")),
]
MODEL_IDS = ["llama3", "qwen3"]


def load_model(tmp_dir, model_name, config_cls, model_cls, mesh_axes, *, loss_chunk_size=0):
    """Load model from pre-saved weights directory."""
    base_config = AutoConfig.from_pretrained(model_name)
    config = config_cls(
        base_config,
        max_lora_adapters=1,
        max_lora_rank=1,
        shard_attention_heads=True,
        loss_chunk_size=loss_chunk_size,
        gradient_checkpointing=False,
    )
    mesh = jax.make_mesh((1, 1), mesh_axes, axis_types=(jax.sharding.AxisType.Auto,) * 2)
    with jax.set_mesh(mesh):
        model = model_cls(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
    load_safetensors(tmp_dir, config, model)
    return model


@pytest.mark.parametrize("model_name,config_cls,model_cls,mesh_axes", MODEL_PARAMS, ids=MODEL_IDS)
def test_compute_logits(model_name, config_cls, model_cls, mesh_axes):
    """Test that model.compute_logits matches HuggingFace logits."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = ["The capital of France is", "Hello world"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    with tempfile.TemporaryDirectory() as tmp:
        # Load HF model, get logits, save weights, then delete to free memory
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
        hf_outputs = hf_model(batch.input_ids, attention_mask=batch.attention_mask)
        hf_logits = hf_outputs.logits.detach().numpy()
        hf_model.save_pretrained(tmp, safe_serialization=True)
        del hf_model, hf_outputs

        # Load our model from saved weights
        model = load_model(tmp, model_name, config_cls, model_cls, mesh_axes)

        # Get our logits via compute_logits
        outputs = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy())
        our_logits = np.asarray(model.compute_logits(outputs.last_hidden_state))

        np.testing.assert_allclose(our_logits, hf_logits, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("model_name,config_cls,model_cls,mesh_axes", MODEL_PARAMS, ids=MODEL_IDS)
@pytest.mark.parametrize("chunk_size", [8, 16, 32])
def test_chunked_logprobs(model_name, config_cls, model_cls, mesh_axes, chunk_size):
    """Test that chunked and non-chunked compute_logprobs produce identical results."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = ["The capital of France is", "Hello world"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    input_ids = jnp.array(batch.input_ids.numpy())
    attention_mask = jnp.array(batch.attention_mask.numpy())
    target_ids = jnp.roll(input_ids, -1, axis=1)

    with tempfile.TemporaryDirectory() as tmp:
        # Save HF weights once
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
        hf_model.save_pretrained(tmp, safe_serialization=True)
        del hf_model

        # Load non-chunked model, compute logprobs, then delete
        model = load_model(tmp, model_name, config_cls, model_cls, mesh_axes, loss_chunk_size=0)
        outputs = model(input_ids, attention_mask=attention_mask)
        logprobs_nonchunked = np.asarray(model.compute_logprobs(outputs.last_hidden_state, target_ids))
        del model, outputs

        # Load chunked model, compute logprobs
        model = load_model(tmp, model_name, config_cls, model_cls, mesh_axes, loss_chunk_size=chunk_size)
        outputs = model(input_ids, attention_mask=attention_mask)
        logprobs_chunked = np.asarray(model.compute_logprobs(outputs.last_hidden_state, target_ids))

    np.testing.assert_allclose(
        logprobs_chunked,
        logprobs_nonchunked,
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Chunked vs non-chunked logprobs mismatch for chunk_size={chunk_size}",
    )
