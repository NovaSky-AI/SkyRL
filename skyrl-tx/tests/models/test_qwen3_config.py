"""Tests for Qwen3Config."""

from tx.models.configs import Qwen3Config
from transformers import PretrainedConfig


def test_config_wraps_pretrained_config():
    """Test that Qwen3Config wraps a PretrainedConfig and adds LoRA params."""
    hf_config = PretrainedConfig.from_pretrained("Qwen/Qwen3-0.6B")
    config = Qwen3Config(hf_config, max_lora_adapters=8, max_lora_rank=16, shard_attention_heads=False)

    # Check LoRA params were set
    assert config.max_lora_adapters == 8
    assert config.max_lora_rank == 16
    assert config.shard_attention_heads is False

    # Check base config attributes were copied
    assert config.vocab_size > 0
    assert config.hidden_size > 0
    assert config.num_hidden_layers > 0


def test_config_preserves_moe_config():
    """Test that MoE-specific configs are preserved."""
    hf_config = PretrainedConfig.from_pretrained("trl-internal-testing/tiny-Qwen3MoeForCausalLM")
    config = Qwen3Config(hf_config, max_lora_adapters=3, max_lora_rank=4, shard_attention_heads=True)

    # Check that MoE-specific attributes are preserved
    assert config.num_experts > 0
