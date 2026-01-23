import tempfile

from flax import nnx
import jax
import numpy as np
import pytest
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tx.models.configs import Llama3Config, Qwen3Config
from tx.models.llama3 import Llama3ForCausalLM
from tx.models.qwen3 import Qwen3ForCausalLM
from tx.utils.models import load_safetensors


@pytest.mark.parametrize(
    "model_name,config_cls,model_cls,mesh_axes",
    [
        ("unsloth/Llama-3.2-1B", Llama3Config, Llama3ForCausalLM, ("dp", "tp")),
        ("Qwen/Qwen3-0.6B", Qwen3Config, Qwen3ForCausalLM, ("fsdp", "tp")),
    ],
    ids=["llama3", "qwen3"],
)
def test_compute_logits(model_name, config_cls, model_cls, mesh_axes):
    """Test that model.compute_logits matches HuggingFace logits."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load HF model in float32 for the comparison (our model will also use float32)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager", use_safetensors=True
    )

    inputs = ["The capital of France is", "Hello world"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)

        base_config = AutoConfig.from_pretrained(model_name)
        config = config_cls(base_config, max_lora_adapters=1, max_lora_rank=1, shard_attention_heads=True)
        mesh = jax.make_mesh((1, 1), mesh_axes)
        with jax.set_mesh(mesh):
            import jax.numpy as jnp
            # Use float32 to match HF model for accurate comparison
            model = model_cls(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        # Get HF logits
        hf_outputs = hf_model(batch.input_ids, attention_mask=batch.attention_mask)
        hf_logits = hf_outputs.logits.detach().numpy()

        # Get our logits via compute_logits
        outputs = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy())
        our_logits = np.asarray(model.compute_logits(outputs.last_hidden_state))

        np.testing.assert_allclose(our_logits, hf_logits, rtol=3e-2, atol=3e-2)
