"""Model output dataclasses."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

import jax
from transformers import PretrainedConfig

from tx.utils.generator import KVCache


class ModelForCausalLM(Protocol):
    config: PretrainedConfig


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
        logits: The language modeling logits.
        last_hidden_state: The last hidden state from the model.
        kv_cache: The updated key-value cache.
        hidden_states: All hidden states, if output_hidden_states=True.
    """

    logits: jax.Array
    last_hidden_state: jax.Array
    kv_cache: KVCache
    hidden_states: list[jax.Array] | None = None
