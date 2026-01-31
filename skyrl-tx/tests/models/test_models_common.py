import tempfile
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tx.models.configs import Llama3Config, ModelConfig, Qwen3Config
from tx.models.llama3 import Llama3ForCausalLM
from tx.models.qwen3 import Qwen3ForCausalLM
from tx.models.types import CausalLMOutput, ModelForCausalLM
from tx.utils.models import load_safetensors

MODEL_PARAMS = [
    ("unsloth/Llama-3.2-1B", Llama3Config, Llama3ForCausalLM, ("fsdp", "tp")),
    ("Qwen/Qwen3-0.6B", Qwen3Config, Qwen3ForCausalLM, ("fsdp", "tp")),
]
MODEL_IDS = ["llama3", "qwen3"]


def create_model(
    model_name: str,
    config_cls: type[ModelConfig],
    model_cls: type[ModelForCausalLM],
    mesh_axes: tuple[str, str],
    *,
    mesh_axis_types: tuple[jax.sharding.AxisType, ...] | None = None,
    seed: int = 0,
    **config_kwargs: Any,
) -> tuple[ModelForCausalLM, ModelConfig]:
    """Create model with random weights for testing."""
    base_config = AutoConfig.from_pretrained(model_name)
    config = config_cls(base_config, max_lora_adapters=1, max_lora_rank=1, shard_attention_heads=True, **config_kwargs)
    # Default to Auto axis types to avoid sharding resolution errors
    if mesh_axis_types is None:
        mesh_axis_types = (jax.sharding.AxisType.Auto,) * len(mesh_axes)
    mesh = jax.make_mesh((1, 1), mesh_axes, axis_types=mesh_axis_types)
    with jax.set_mesh(mesh):
        model = model_cls(config, dtype=jnp.float32, rngs=nnx.Rngs(seed))
    return model, config


def load_model(
    tmp_dir: str,
    model_name: str,
    config_cls: type[ModelConfig],
    model_cls: type[ModelForCausalLM],
    mesh_axes: tuple[str, str],
    *,
    loss_chunk_size: int = 0,
) -> ModelForCausalLM:
    """Load model from pre-saved weights directory."""
    model, config = create_model(
        model_name,
        config_cls,
        model_cls,
        mesh_axes,
        loss_chunk_size=loss_chunk_size,
        gradient_checkpointing=False,
    )
    load_safetensors(tmp_dir, config, model)
    return model


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
        model, config = create_model(
            model_name, config_cls, model_cls, mesh_axes, gradient_checkpointing=gradient_checkpointing
        )
        input_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, config.vocab_size)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        out = model(input_ids, attention_mask=attention_mask, **forward_kwargs)
        return model, config, out

    def test_output_matches_non_checkpointed(
        self,
        model_name: str,
        config_cls: type[ModelConfig],
        model_cls: type[ModelForCausalLM],
        mesh_axes: tuple[str, str],
    ) -> None:
        """Forward pass should produce identical outputs with/without checkpointing."""
        model, _, out = self._forward(model_name, config_cls, model_cls, mesh_axes, gradient_checkpointing=False)
        logits_no_ckpt = model.compute_logits(out.last_hidden_state)
        del model, out

        model, _, out = self._forward(model_name, config_cls, model_cls, mesh_axes, gradient_checkpointing=True)
        logits_ckpt = model.compute_logits(out.last_hidden_state)
        del model, out

        np.testing.assert_allclose(logits_no_ckpt, logits_ckpt, rtol=1e-4, atol=1e-6)

    def test_hidden_states_length_matches(
        self,
        model_name: str,
        config_cls: type[ModelConfig],
        model_cls: type[ModelForCausalLM],
        mesh_axes: tuple[str, str],
    ) -> None:
        """Both paths should return same number of hidden states."""
        _, config, out = self._forward(
            model_name, config_cls, model_cls, mesh_axes, gradient_checkpointing=False, output_hidden_states=True
        )
        hidden_states_no_ckpt = out.hidden_states
        num_hidden_layers = config.num_hidden_layers
        del out

        _, _, out = self._forward(
            model_name, config_cls, model_cls, mesh_axes, gradient_checkpointing=True, output_hidden_states=True
        )
        hidden_states_ckpt = out.hidden_states
        del out

        assert len(hidden_states_no_ckpt) == len(hidden_states_ckpt) == num_hidden_layers + 1
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
        model, config = create_model(model_name, config_cls, model_cls, mesh_axes)
        config.gradient_checkpointing = True

        batch_size, seq_len = 2, 8
        input_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, config.vocab_size)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        out = model(input_ids, attention_mask=attention_mask)

        # keys is a stacked array with shape (num_layers, batch, seq, heads, dim)
        assert out.kv_cache.keys.shape[0] == config.num_hidden_layers


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
