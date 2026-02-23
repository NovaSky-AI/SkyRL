"""Model output dataclasses."""

from __future__ import annotations
from dataclasses import dataclass

import jax

from tx.models.configs import ModelConfig
from tx.utils.generator import KVCache


class ModelForCausalLM:

    config: ModelConfig

    def get_model_config(self) -> ModelConfig:
        return self.config

    def is_lora_param(self, path: tuple, _value) -> bool:
        """Return True if a parameter path corresponds to trainable LoRA/connector weights."""
        is_lora = any(name in path for name in ("lora_A", "lora_B"))
        is_connector = self.config.expansion_rate > 1 and any(
            name in path for name in ("attn_connector", "mlp_connector")
        )
        return is_lora or is_connector


@jax.tree_util.register_dataclass
@dataclass
class ModelOutput:
    """Output type for models like Qwen3Model.

    Attributes:
        last_hidden_state: The last hidden state from the model.
        kv_cache: The updated key-value cache (None during training).
        hidden_states: All hidden states if output_hidden_states=True.
    """

    last_hidden_state: jax.Array
    kv_cache: KVCache | None
    hidden_states: list[jax.Array] | None = None


@jax.tree_util.register_dataclass
@dataclass
class CausalLMOutput:
    """Output type for causal language models like Qwen3ForCausalLM.

    Attributes:
        last_hidden_state: The last hidden state from the model.
        kv_cache: The updated key-value cache (None during training).
        hidden_states: All hidden states, if output_hidden_states=True.
    """

    last_hidden_state: jax.Array
    kv_cache: KVCache | None
    hidden_states: list[jax.Array] | None = None
