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
from tx.utils.models import get_dtype, load_safetensors

MODEL_PARAMS = [
    ("unsloth/Llama-3.2-1B", Llama3Config, Llama3ForCausalLM, ("dp", "tp")),
    ("Qwen/Qwen3-0.6B", Qwen3Config, Qwen3ForCausalLM, ("fsdp", "tp")),
]
MODEL_IDS = ["llama3", "qwen3"]


def make_model(model_name, config_cls, model_cls, mesh_axes, *, loss_chunk_size=0, gradient_checkpointing=False):
    """Create a model with the given config."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)

    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)

        base_config = AutoConfig.from_pretrained(model_name)
        config = config_cls(
            base_config,
            max_lora_adapters=1,
            max_lora_rank=1,
            shard_attention_heads=True,
            loss_chunk_size=loss_chunk_size,
            gradient_checkpointing=gradient_checkpointing,
        )
        mesh = jax.make_mesh((1, 1), mesh_axes)
        with jax.set_mesh(mesh):
            model = model_cls(config, dtype=get_dtype(config.dtype), rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

    return model, tokenizer, hf_model


@pytest.mark.parametrize("model_name,config_cls,model_cls,mesh_axes", MODEL_PARAMS, ids=MODEL_IDS)
def test_compute_logits(model_name, config_cls, model_cls, mesh_axes):
    """Test that model.compute_logits matches HuggingFace logits."""
    model, tokenizer, hf_model = make_model(model_name, config_cls, model_cls, mesh_axes)

    inputs = ["The capital of France is", "Hello world"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    # Get HF logits
    hf_outputs = hf_model(batch.input_ids, attention_mask=batch.attention_mask)
    hf_logits = hf_outputs.logits.detach().numpy()

    # Get our logits via compute_logits
    outputs = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy())
    our_logits = model.compute_logits(outputs.last_hidden_state)

    np.testing.assert_allclose(our_logits, hf_logits, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("model_name,config_cls,model_cls,mesh_axes", MODEL_PARAMS, ids=MODEL_IDS)
@pytest.mark.parametrize("chunk_size", [8, 16, 32])
def test_chunked_logprobs(model_name, config_cls, model_cls, mesh_axes, chunk_size):
    """Test that chunked and non-chunked compute_logprobs produce identical results."""
    model_chunked, tokenizer, _ = make_model(
        model_name, config_cls, model_cls, mesh_axes, loss_chunk_size=chunk_size
    )
    model_nonchunked, _, _ = make_model(
        model_name, config_cls, model_cls, mesh_axes, loss_chunk_size=0
    )

    inputs = ["The capital of France is", "Hello world"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    input_ids = jnp.array(batch.input_ids.numpy())
    attention_mask = jnp.array(batch.attention_mask.numpy())
    target_ids = jnp.roll(input_ids, -1, axis=1)

    # Get hidden states
    outputs_chunked = model_chunked(input_ids, attention_mask=attention_mask)
    outputs_nonchunked = model_nonchunked(input_ids, attention_mask=attention_mask)

    # Compute logprobs with both methods
    logprobs_chunked = model_chunked.compute_logprobs(outputs_chunked.last_hidden_state, target_ids)
    logprobs_nonchunked = model_nonchunked.compute_logprobs(outputs_nonchunked.last_hidden_state, target_ids)

    np.testing.assert_allclose(
        np.asarray(logprobs_chunked),
        np.asarray(logprobs_nonchunked),
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Chunked vs non-chunked logprobs mismatch for chunk_size={chunk_size}",
    )
