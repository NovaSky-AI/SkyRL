from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from tx.models.configs import Llama3Config, ModelConfig, Qwen3Config
from tx.models.llama3 import Llama3ForCausalLM
from tx.models.qwen3 import Qwen3ForCausalLM
from tx.models.types import CausalLMOutput, ModelForCausalLM

from tests.models.conftest import create_model, load_model

MODEL_PARAMS = [
    ("unsloth/Llama-3.2-1B", Llama3Config, Llama3ForCausalLM, ("fsdp", "tp")),
    ("Qwen/Qwen3-0.6B", Qwen3Config, Qwen3ForCausalLM, ("fsdp", "tp")),
    # ("olm/olmo-3-mini", Olmo3Config, Olmo3ForCausalLM, ("fsdp", "tp")),
]
MODEL_IDS = ["llama3", "qwen3"]  # , "olmo3"]


@pytest.mark.parametrize("model_name,config_cls,model_cls,mesh_axes", MODEL_PARAMS, ids=MODEL_IDS)
class TestGradientCheckpointing:

    def _forward(
        self,
        model_name: str,
        config_cls: type[ModelConfig],
        model_cls: type[ModelForCausalLM],
        mesh_axes: tuple[str, str],
        gradient_checkpointing: bool,
        **forward_kwargs: Any,
    ) -> tuple[ModelForCausalLM, ModelConfig, CausalLMOutput]:
        """Create model, run forward pass, and return (model, config, out)."""
        batch_size, seq_len = 2, 8
        config, model = create_model(
            model_name,
            config_cls,
            model_cls,
            mesh_axes,
            max_lora_adapters=1,
            max_lora_rank=1,
            gradient_checkpointing=gradient_checkpointing,
        )
        input_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, config.vocab_size)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        out = model(input_ids, attention_mask=attention_mask, **forward_kwargs)
        return model, config, out

    def test_output_and_hidden_states_match(
        self,
        model_name: str,
        config_cls: type[ModelConfig],
        model_cls: type[ModelForCausalLM],
        mesh_axes: tuple[str, str],
    ) -> None:
        """Forward pass should produce identical outputs and hidden states with/without checkpointing."""
        results = {}
        for use_checkpointing in (False, True):
            model, config, out = self._forward(
                model_name,
                config_cls,
                model_cls,
                mesh_axes,
                gradient_checkpointing=use_checkpointing,
                output_hidden_states=True,
            )
            results[use_checkpointing] = {
                "logits": np.asarray(model.compute_logits(out.last_hidden_state)),
                "hidden_states": [np.asarray(hs) for hs in out.hidden_states],
                "num_hidden_layers": config.num_hidden_layers,
            }
            del model, config, out

        np.testing.assert_allclose(results[False]["logits"], results[True]["logits"], rtol=1e-4, atol=1e-6)

        hidden_states_no_ckpt = results[False]["hidden_states"]
        hidden_states_ckpt = results[True]["hidden_states"]
        assert len(hidden_states_no_ckpt) == len(hidden_states_ckpt) == results[False]["num_hidden_layers"] + 1
        for i, (hs_no_ckpt, hs_ckpt) in enumerate(zip(hidden_states_no_ckpt, hidden_states_ckpt)):
            np.testing.assert_allclose(
                hs_no_ckpt, hs_ckpt, rtol=1e-4, atol=1e-6, err_msg=f"Mismatch at hidden state {i}"
            )

    def test_kv_cache_with_checkpointing(
        self,
        model_name: str,
        config_cls: type[ModelConfig],
        model_cls: type[ModelForCausalLM],
        mesh_axes: tuple[str, str],
    ) -> None:
        """KV cache should be populated even with gradient checkpointing enabled."""
        _, config, out = self._forward(model_name, config_cls, model_cls, mesh_axes, gradient_checkpointing=True)

        # keys is a list with one entry per layer
        assert len(out.kv_cache.keys) == config.num_hidden_layers


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
