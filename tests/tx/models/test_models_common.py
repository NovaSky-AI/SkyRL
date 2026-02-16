import jax.numpy as jnp
import numpy as np
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from skyrl.tx.models.configs import Llama3Config, ModelConfig, Qwen3Config
from skyrl.tx.models.llama3 import Llama3ForCausalLM
from skyrl.tx.models.qwen3 import Qwen3ForCausalLM
from skyrl.tx.models.types import ModelForCausalLM

from tests.tx.models.conftest import load_model

MODEL_PARAMS = [
    ("unsloth/Llama-3.2-1B", Llama3Config, Llama3ForCausalLM, ("fsdp", "tp")),
    ("Qwen/Qwen3-0.6B", Qwen3Config, Qwen3ForCausalLM, ("fsdp", "tp")),
]
MODEL_IDS = ["llama3", "qwen3"]


@pytest.mark.parametrize("model_name,config_cls,model_cls,mesh_axes", MODEL_PARAMS, ids=MODEL_IDS)
def test_compute_logits(
    model_name: str,
    config_cls: type[ModelConfig],
    model_cls: type[ModelForCausalLM],
    mesh_axes: tuple[str, str],
) -> None:
    """Test that model.compute_logits matches HuggingFace logits."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = ["The capital of France is", "Hello world"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    # Load HF model, get logits, then delete to free memory
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
    hf_outputs = hf_model(batch.input_ids, attention_mask=batch.attention_mask)
    hf_logits = hf_outputs.logits.detach().numpy()
    del hf_model, hf_outputs

    _, model = load_model(
        model_name,
        config_cls,
        model_cls,
        mesh_axes,
        max_lora_adapters=1,
        max_lora_rank=1,
        gradient_checkpointing=False,
    )

    # Get our logits via compute_logits
    outputs = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy())
    our_logits = np.asarray(model.compute_logits(outputs.last_hidden_state))

    np.testing.assert_allclose(our_logits, hf_logits, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("model_name,config_cls,model_cls,mesh_axes", MODEL_PARAMS, ids=MODEL_IDS)
@pytest.mark.parametrize("chunk_size", [8, 16, 32])
def test_chunked_logprobs(
    model_name: str,
    config_cls: type[ModelConfig],
    model_cls: type[ModelForCausalLM],
    mesh_axes: tuple[str, str],
    chunk_size: int,
) -> None:
    """Test that chunked and non-chunked compute_logprobs produce identical results."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = ["The capital of France is", "Hello world"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    input_ids = jnp.array(batch.input_ids.numpy())
    attention_mask = jnp.array(batch.attention_mask.numpy())
    target_ids = jnp.roll(input_ids, -1, axis=1)

    common_kwargs = dict(max_lora_adapters=1, max_lora_rank=1, gradient_checkpointing=False)

    # Load non-chunked model, compute logprobs, then delete
    _, model = load_model(model_name, config_cls, model_cls, mesh_axes, loss_chunk_size=0, **common_kwargs)
    outputs = model(input_ids, attention_mask=attention_mask)
    logprobs_nonchunked = np.asarray(model.compute_logprobs(outputs.last_hidden_state, target_ids))
    del model, outputs

    # Load chunked model, compute logprobs
    _, model = load_model(model_name, config_cls, model_cls, mesh_axes, loss_chunk_size=chunk_size, **common_kwargs)
    outputs = model(input_ids, attention_mask=attention_mask)
    logprobs_chunked = np.asarray(model.compute_logprobs(outputs.last_hidden_state, target_ids))

    np.testing.assert_allclose(
        logprobs_chunked,
        logprobs_nonchunked,
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Chunked vs non-chunked logprobs mismatch for chunk_size={chunk_size}",
    )
