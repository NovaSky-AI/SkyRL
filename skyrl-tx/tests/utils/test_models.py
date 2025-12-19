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
from transformers import AutoConfig, AutoModelForCausalLM

from tx.layers.lora import update_adapter_config
from tx.models.configs import Qwen3Config
from tx.models.qwen3 import Qwen3ForCausalLM
from tx.tinker.types import LoraConfig
from tx.utils import models
from tx.utils.storage import download_and_unpack


def create_test_model(base_model_name: str, rank: int, alpha: int, adapter_index: int):
    """Create a small Qwen3 model for testing with LoRA enabled."""
    base_config = AutoConfig.from_pretrained(base_model_name)
    # Make it smaller for testing
    base_config.num_hidden_layers = 1
    base_config.hidden_size = 64
    base_config.intermediate_size = 128
    base_config.num_attention_heads = 2
    base_config.num_key_value_heads = 2

    config = Qwen3Config(base_config, max_lora_adapters=5, max_lora_rank=32, shard_attention_heads=True)

    mesh = jax.make_mesh((1, 1), ("dp", "tp"))
    with jax.set_mesh(mesh):
        model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        update_adapter_config(model, adapter_index=adapter_index, lora_config=LoraConfig(rank=rank, alpha=alpha))

    return config, base_config, model


@pytest.mark.parametrize("storage_type", ["local", "cloud"])
def test_save_load_lora_checkpoint(storage_type: str, monkeypatch, tmp_path: Path):
    base_model_name = "Qwen/Qwen3-0.6B"
    # Setup output path for tar.gz file based on storage type
    if storage_type == "cloud":
        monkeypatch.setitem(implementation_registry, "s3", local_s3_implementation)
        client = local_s3_implementation.client_class(local_storage_dir=tmp_path)
        output_path = CloudPath("s3://bucket/checkpoint.tar.gz", client=client)
    else:
        output_path = tmp_path / "checkpoint.tar.gz"

    rank, alpha, adapter_index = 8, 16, 2
    config, base_config, model = create_test_model(base_model_name, rank, alpha, adapter_index)
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
    models.save_lora_checkpoint(model, base_model_name, adapter_config, adapter_index, output_path)
    assert output_path.exists()

    # Load with peft and verify
    with download_and_unpack(output_path) as extracted_dir:
        base_model = AutoModelForCausalLM.from_config(base_config)
        peft_model = PeftModel.from_pretrained(base_model, extracted_dir)

        assert peft_model.peft_config["default"].r == rank
        assert peft_model.peft_config["default"].lora_alpha == alpha

        q_proj_adapter = peft_model.base_model.model.model.layers[0].self_attn.q_proj
        lora_A = q_proj_adapter.lora_A["default"].weight
        lora_B = q_proj_adapter.lora_B["default"].weight

        assert torch.allclose(lora_A, torch.from_numpy(expected_lora_A), atol=1e-6)
        assert torch.allclose(lora_B, torch.from_numpy(expected_lora_B), atol=1e-6)
