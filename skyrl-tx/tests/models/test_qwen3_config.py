"""Tests for Qwen3Config."""

from tx.models.configs import Qwen3Config
from transformers import Qwen3Config as HfQwen3Config


def test_config_inherits_from_qwen3config():
    """Test that Qwen3Config properly inherits from HuggingFace Qwen3Config."""
    config = Qwen3Config()
    assert isinstance(config, HfQwen3Config)


def test_config_has_lora_defaults():
    """Test that default LoRA parameters are set correctly."""
    config = Qwen3Config()
    assert config.max_lora_adapters == 32
    assert config.max_lora_rank == 32
    assert config.shard_attention_heads == True


def test_from_pretrained_with_lora():
    """Test factory method for loading from HuggingFace."""
    # Note: This test requires network access
    # Using a small model for faster testing
    config = Qwen3Config.from_pretrained_with_lora(
        "Qwen/Qwen3-0.6B",
        max_lora_adapters=8,
        max_lora_rank=16,
        shard_attention_heads=False
    )

    # Check LoRA params were set
    assert config.max_lora_adapters == 8
    assert config.max_lora_rank == 16
    assert config.shard_attention_heads == False

    # Check base config was loaded
    assert config.vocab_size > 0
    assert config.hidden_size > 0
    assert config.num_hidden_layers > 0
