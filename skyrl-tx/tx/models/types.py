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


@jax.tree_util.register_dataclass
@dataclass
class ModelOutput:
    """Output type for models like Qwen3Model.

    Attributes:
        last_hidden_state: The last hidden state from the model.
        kv_cache: The updated key-value cache.
        hidden_states: All hidden states if output_hidden_states=True.
    """

    last_hidden_state: jax.Array
    kv_cache: KVCache
    hidden_states: list[jax.Array] | None = None


@jax.tree_util.register_dataclass
@dataclass
class CausalLMOutput:
    """Output type for causal language models like Qwen3ForCausalLM.

    Attributes:
        last_hidden_state: The last hidden state from the model.
        kv_cache: The updated key-value cache.
        hidden_states: All hidden states, if output_hidden_states=True.
    """

    last_hidden_state: jax.Array
    kv_cache: KVCache
    hidden_states: list[jax.Array] | None = None
