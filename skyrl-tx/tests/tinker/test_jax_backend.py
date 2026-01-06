"""Unit tests for JaxBackend."""

import pytest

from tx.tinker.backends.jax import JaxBackend, JaxBackendConfig
from tx.tinker.types import LoraConfig
from tx.layers.lora import LoRALinear


BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
MAX_LORA_ADAPTERS = 4


def create_backend():
    """Create a JaxBackend."""
    config = JaxBackendConfig(max_lora_adapters=MAX_LORA_ADAPTERS, max_lora_rank=32)
    return JaxBackend(BASE_MODEL, config)


def create_model(backend: JaxBackend, model_id: str) -> int:
    """Create a model and return its adapter index."""
    lora_config = LoraConfig(rank=8, alpha=16)
    backend.create_model(model_id, lora_config)
    return backend.models[model_id].adapter_index


def test_delete_model_basic():
    """Test basic model deletion."""
    backend = create_backend()
    model_id = "test_model"

    # Create model
    _ = create_model(backend, model_id)
    assert backend.has_model(model_id)

    # Delete model
    backend.delete_model(model_id)
    assert not backend.has_model(model_id)


def test_delete_non_existent_model():
    """Test deleting a non-existent model raises ValueError."""
    backend = create_backend()
    with pytest.raises(ValueError, match="not found"):
        backend.delete_model("nonexistent_model")


def test_adapter_slot_reuse():
    """Test that deleted adapter slots are reused."""
    backend = create_backend()

    # Create 3 models
    idx1 = create_model(backend, "model_1")
    idx2 = create_model(backend, "model_2")
    idx3 = create_model(backend, "model_3")
    assert idx1 == 1
    assert idx2 == 2
    assert idx3 == 3

    # Delete first model, new model should reuse index 1
    backend.delete_model("model_1")
    idx4 = create_model(backend, "model_4")
    assert idx4 == 1

    # Delete middle model, new model should fill gap at index 1
    backend.delete_model("model_2")
    idx5 = create_model(backend, "model_5")
    assert idx5 == 2


def test_max_adapters_limit():
    """Test that creating more than max_lora_adapters raises ValueError."""
    backend = create_backend()
    # Create max number of models
    for i in range(MAX_LORA_ADAPTERS):
        _ = create_model(backend, f"model_{i}")

    # Try to create one more - should fail
    with pytest.raises(ValueError, match="Maximum number of LoRA adapters"):
        _ = create_model(backend, "model_overflow")


def test_max_adapters_after_delete():
    """Test that deleting a model frees a slot for new models."""
    backend = create_backend()
    # Create max number of models
    for i in range(MAX_LORA_ADAPTERS):
        _ = create_model(backend, f"model_{i}")

    # Delete one model
    backend.delete_model("model_0")

    # Now we should be able to create a new model
    idx = create_model(backend, "model_new")
    assert idx == 1  # Should reuse the freed slot


def test_clear_adapter_config():
    """Test that clear_adapter_config zeros out adapter state."""
    backend = create_backend()
    model_id = "test_model"
    adapter_idx = create_model(backend, model_id)

    # Verify adapter has non-zero rank after creation
    model = backend.model
    lora_layer: LoRALinear = model.model.layers[0].self_attn.q_proj
    assert lora_layer.lora_ranks[adapter_idx] > 0

    # Delete the model (calls clear_adapter_config internally)
    backend.delete_model(model_id)

    # Verify adapter state is zeroed
    assert lora_layer.lora_ranks[adapter_idx] == 0
    assert lora_layer.lora_scaling[adapter_idx] == 0.0
    assert (lora_layer.lora_A[adapter_idx] == 0.0).all()
    assert (lora_layer.lora_B[adapter_idx] == 0.0).all()
