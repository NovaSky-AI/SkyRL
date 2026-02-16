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

from jax.tree_util import DictKey

from skyrl.tx.layers.lora import init_lora_adapter
from skyrl.tx.models.configs import Qwen3Config
from skyrl.tx.models.qwen3 import Qwen3ForCausalLM
from skyrl.tinker.types import LoraConfig
from skyrl.tx.utils import models
from skyrl.tx.utils.models import extract_adapter_state, insert_adapter_state, is_stacked_path
from skyrl.utils.storage import download_and_unpack


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

    mesh = jax.make_mesh((1, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
    with jax.set_mesh(mesh):
        model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        init_lora_adapter(model, adapter_index=adapter_index, lora_config=LoraConfig(rank=rank, alpha=alpha, seed=0))

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
    adapter_config = LoraConfig(rank=rank, alpha=alpha, seed=0)

    # Set LoRA weights to random values for testing (to catch transpose bugs)
    q_proj = model.model.layers[0].self_attn.q_proj
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(42))
    q_proj.lora_A[...] = jax.random.normal(rng1, q_proj.lora_A[...].shape)
    q_proj.lora_B[...] = jax.random.normal(rng2, q_proj.lora_B[...].shape)

    # Store expected values (trimmed to rank and transposed)
    expected_lora_A = np.array(q_proj.lora_A[...][adapter_index, :, :rank].T)
    expected_lora_B = np.array(q_proj.lora_B[...][adapter_index, :rank, :].T)

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


@pytest.mark.parametrize(
    "path,expected",
    [
        # Stacked paths (DictKey) — real NNX paths include _stacked
        (
            (
                DictKey(key="model"),
                DictKey(key="layers"),
                DictKey(key="_stacked"),
                DictKey(key="self_attn"),
                DictKey(key="lora_A"),
            ),
            True,
        ),
        # Non-stacked paths (DictKey)
        ((DictKey(key="model"), DictKey(key="embed_tokens"), DictKey(key="lora_A")), False),
        ((DictKey(key="lm_head"), DictKey(key="lora_A")), False),
        # String paths
        (("model", "layers", "_stacked", "self_attn", "lora_A"), True),
        (("model", "embed_tokens", "lora_A"), False),
    ],
    ids=["stacked_layers", "embed_tokens", "lm_head", "str_stacked", "str_embed"],
)
def test_is_stacked_path(path, expected):
    """Test is_stacked_path correctly identifies stacked vs non-stacked paths."""
    assert is_stacked_path(path) is expected


def test_extract_insert_adapter_state_roundtrip():
    """Test that extract_adapter_state and insert_adapter_state are inverses."""
    base_model_name = "Qwen/Qwen3-0.6B"
    rank, alpha, adapter_index = 8, 16, 2
    _, _, model = create_test_model(base_model_name, rank, alpha, adapter_index)

    # Set LoRA weights to random values
    q_proj = model.model.layers[0].self_attn.q_proj
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(123))
    q_proj.lora_A[...] = jax.random.normal(rng1, q_proj.lora_A[...].shape)
    q_proj.lora_B[...] = jax.random.normal(rng2, q_proj.lora_B[...].shape)

    # Split model to get lora_params
    _, lora_params, _ = nnx.split(model, model.is_lora_param, ...)

    # Store original values for comparison
    original_lora_A = np.array(q_proj.lora_A[...][adapter_index, :, :rank])
    original_lora_B = np.array(q_proj.lora_B[...][adapter_index, :rank, :])

    # Extract adapter state
    extracted = extract_adapter_state(adapter_index, lora_params, rank)

    # Verify extracted shape is correct (no adapter dimension)
    for path, leaf in jax.tree.leaves_with_path(extracted):
        key = path[-2].key if hasattr(path[-2], "key") else str(path[-2])
        if key in {"lora_A", "lora_B"}:
            # Stacked: should have (num_layers, ...) not (num_layers, num_adapters, ...)
            if is_stacked_path(path):
                assert leaf.shape[0] == 1  # num_layers
                assert leaf.ndim == 3  # (layers, in_dim, rank) or (layers, rank, out_dim)

    # Zero out the adapter's weights
    q_proj.lora_A[...] = q_proj.lora_A[...].at[adapter_index].set(0)
    q_proj.lora_B[...] = q_proj.lora_B[...].at[adapter_index].set(0)

    # Verify weights are zeroed
    assert np.allclose(q_proj.lora_A[...][adapter_index], 0)
    assert np.allclose(q_proj.lora_B[...][adapter_index], 0)

    # Re-split to get updated lora_params
    _, lora_params, _ = nnx.split(model, model.is_lora_param, ...)

    # Insert extracted state back (modifies lora_params in-place via nnx.update)
    insert_adapter_state(adapter_index, lora_params, extracted, rank)

    # Verify weights are restored by checking lora_params directly
    for path, leaf in jax.tree.leaves_with_path(lora_params):
        key = path[-2].key if hasattr(path[-2], "key") else str(path[-2])
        # leaf is a state wrapper with .value, or can be an array directly
        arr = leaf.value if hasattr(leaf, "value") else leaf
        if "q_proj" in str(path) and key == "lora_A":
            restored_lora_A = np.array(arr[0, adapter_index, :, :rank])
        elif "q_proj" in str(path) and key == "lora_B":
            restored_lora_B = np.array(arr[0, adapter_index, :rank, :])

    assert np.allclose(original_lora_A, restored_lora_A), "lora_A not restored correctly"
    assert np.allclose(original_lora_B, restored_lora_B), "lora_B not restored correctly"
