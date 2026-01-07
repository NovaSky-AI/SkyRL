"""Unit tests for JaxBackend."""

import pytest

from tx.tinker.backends.jax import JaxBackend, JaxBackendConfig
from tx.tinker.types import LoraConfig
from tx.layers.lora import LoRALinear


BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
MAX_LORA_ADAPTERS = 4
LORA_RANK = 8


def create_backend(max_lora_adapters: int = MAX_LORA_ADAPTERS):
    """Create a JaxBackend."""
    config = JaxBackendConfig(max_lora_adapters=max_lora_adapters, max_lora_rank=32)
    return JaxBackend(BASE_MODEL, config)


def create_model(backend: JaxBackend, model_id: str) -> int:
    """Create a model and return its adapter index."""
    lora_config = LoraConfig(rank=LORA_RANK, alpha=16)
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

    # Create 3 models and check adapter indices
    assert create_model(backend, "model_1") == 1
    assert create_model(backend, "model_2") == 2
    assert create_model(backend, "model_3") == 3

    # Delete first model, new model should reuse index 1
    backend.delete_model("model_1")
    assert create_model(backend, "model_4") == 1

    # Delete middle model, new model should fill gap at index 1
    backend.delete_model("model_2")
    assert create_model(backend, "model_5") == 2


def test_max_adapters_limit():
    """Test that creating more than available adapters raises ValueError."""
    backend = create_backend()

    # Index 0 is reserved for base model, so we have max_lora_adapters - 1 slots
    num_available = MAX_LORA_ADAPTERS - 1
    for i in range(num_available):
        _ = create_model(backend, f"model_{i}")

    # Try to create one more - should fail
    with pytest.raises(ValueError, match="Maximum number of LoRA adapters"):
        _ = create_model(backend, "model_overflow")


def test_max_adapters_after_delete():
    """Test that deleting a model frees a slot for new models."""
    backend = create_backend()
    # Index 0 is reserved for base model, so we have max_lora_adapters - 1 slots
    num_available = MAX_LORA_ADAPTERS - 1
    for i in range(num_available):
        _ = create_model(backend, f"model_{i}")

    # Delete one model
    backend.delete_model("model_0")

    # Now we should be able to create a new model which should reuse the freed slot
    assert create_model(backend, "model_new") == 1


def test_clear_adapter_config():
    """Test that clear_adapter zeros out adapter state."""
    backend = create_backend()
    model_id = "test_model"
    adapter_idx = create_model(backend, model_id)

    # Verify adapter has non-zero rank after creation
    model = backend.model
    lora_layer: LoRALinear = model.model.layers[0].self_attn.q_proj
    assert lora_layer.lora_ranks[adapter_idx] > 0

    # Delete the model (calls clear_adapter internally)
    backend.delete_model(model_id)

    # Verify adapter state is zeroed
    assert lora_layer.lora_ranks[adapter_idx] == 0
    assert lora_layer.lora_scaling[adapter_idx] == 0.0
    assert (lora_layer.lora_A[adapter_idx] == 0.0).all()
    assert (lora_layer.lora_B[adapter_idx] == 0.0).all()


def test_adapter_reuse_initializes_lora_adapter():
    """Test that reusing an adapter slot initializes the lora adapter properly."""
    # Use max_lora_adapters=2 so only slot 1 is available
    # (slot 0 is reserved for base model)
    backend = create_backend(max_lora_adapters=2)
    model = backend.model
    lora_layer: LoRALinear = model.model.layers[0].self_attn.q_proj

    # Create first model
    model_id_1 = "model_1"
    adapter_idx = create_model(backend, model_id_1)

    # Verify lora_A is non-zero after creation
    assert not (
        lora_layer.lora_A[adapter_idx, ..., :LORA_RANK] == 0.0
    ).all(), "lora_A should be initialized with he_uniform (non-zero)"

    # Delete the model (clears both lora_A and lora_B to zeros)
    backend.delete_model(model_id_1)
    assert (lora_layer.lora_A[adapter_idx] == 0.0).all(), "lora_A should be zeroed after clear_lora_adapter"

    # Create a new model that reuses the same adapter slot
    model_id_2 = "model_2"
    new_adapter_idx = create_model(backend, model_id_2)
    assert new_adapter_idx == adapter_idx, "Should reuse the same adapter slot"

    # Verify lora_A is initialized (non-zero)
    assert not (
        lora_layer.lora_A[adapter_idx, ..., :LORA_RANK] == 0.0
    ).all(), "lora_A should be initialized with he_uniform after adapter reuse"

    # Verify lora_B is zeros
    assert (lora_layer.lora_B[adapter_idx] == 0.0).all(), "lora_B should be zeros"
