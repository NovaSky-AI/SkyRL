from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from cloudpathlib import CloudPath, implementation_registry
from cloudpathlib.local import local_s3_implementation
from flax import nnx
from peft import PeftModel
from transformers import AutoModelForCausalLM

from tx.layers.lora import update_adapter_config
from tx.models import Qwen3Config, Qwen3ForCausalLM
from tx.tinker.types import LoraConfig
from tx.utils import models
from tx.utils.storage import download_and_unpack


def create_test_model(rank: int, alpha: int, adapter_index: int):
    """Create a small Qwen3 model for testing with LoRA enabled."""
    config = Qwen3Config.from_pretrained_with_lora(
        "Qwen/Qwen3-0.6B",
        max_lora_adapters=5,
        max_lora_rank=32
    )
    # Make it smaller for testing
    config.num_hidden_layers = 1
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 2
    config.num_key_value_heads = 2

    mesh = jax.make_mesh((1, 1), ("dp", "tp"))
    with jax.set_mesh(mesh):
        model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        update_adapter_config(model, adapter_index=adapter_index, lora_rank=rank, lora_alpha=alpha)

    return config, model


@pytest.mark.parametrize("storage_type", ["local", "cloud"])
def test_save_load_lora_checkpoint(storage_type: str, monkeypatch, tmp_path: Path):
    # Setup output path for tar.gz file based on storage type
    if storage_type == "cloud":
        monkeypatch.setitem(implementation_registry, "s3", local_s3_implementation)
        client = local_s3_implementation.client_class(local_storage_dir=tmp_path)
        output_path = CloudPath("s3://bucket/checkpoint.tar.gz", client=client)
    else:
        output_path = tmp_path / "checkpoint.tar.gz"

    rank, alpha, adapter_index = 8, 16, 2
    config, model = create_test_model(rank, alpha, adapter_index)
    adapter_config = LoraConfig(rank=rank, alpha=alpha)

    # Set LoRA weights to random values for testing (to catch transpose bugs)
    q_proj = model.model.layers[0].self_attn.q_proj
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(42))
    q_proj.lora_A.value = jax.random.normal(rng1, q_proj.lora_A.value.shape)
    q_proj.lora_B.value = jax.random.normal(rng2, q_proj.lora_B.value.shape)

    # Store expected values (trimmed to rank and transposed)
    expected_lora_A = np.array(q_proj.lora_A.value[adapter_index, :, :rank].T)
    expected_lora_B = np.array(q_proj.lora_B.value[adapter_index, :rank, :].T)

    # Save and verify checkpoint exists
    models.save_lora_checkpoint(model, adapter_config, adapter_index, output_path)
    assert output_path.exists()

    # Load with peft and verify
    with download_and_unpack(output_path) as extracted_dir:
        # Convert to base Qwen3Config for HuggingFace compatibility
        from transformers import Qwen3Config
        config_dict = config.to_dict()
        # Remove LoRA-specific fields that Qwen3Config doesn't know about
        config_dict.pop("max_lora_adapters", None)
        config_dict.pop("max_lora_rank", None)
        config_dict.pop("shard_attention_heads", None)
        # Fix layer_types to match num_hidden_layers
        if config_dict.get("num_hidden_layers") != len(config_dict.get("layer_types", [])):
            config_dict["layer_types"] = None
        base_config = Qwen3Config(**config_dict)
        base_model = AutoModelForCausalLM.from_config(base_config)
        peft_model = PeftModel.from_pretrained(base_model, extracted_dir)

        assert peft_model.peft_config["default"].r == rank
        assert peft_model.peft_config["default"].lora_alpha == alpha

        q_proj_adapter = peft_model.base_model.model.model.layers[0].self_attn.q_proj
        lora_A = q_proj_adapter.lora_A["default"].weight
        lora_B = q_proj_adapter.lora_B["default"].weight

        assert torch.allclose(lora_A, torch.from_numpy(expected_lora_A), atol=1e-6)
        assert torch.allclose(lora_B, torch.from_numpy(expected_lora_B), atol=1e-6)
